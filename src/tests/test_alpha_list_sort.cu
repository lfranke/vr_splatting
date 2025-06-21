/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"
#include "rendering/AlphaListSort.h"
#include "rendering/NeuralPointCloudCuda.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderInfo.h"
using namespace Saiga;

#include "gtest/gtest.h"

struct SortAlphaTest
{
    SortAlphaTest(int image_size, int max_segment_length)
    {
        torch::manual_seed(362346);
        Random::setSeed(93467);
        info.images.push_back({});
        info.num_layers = 1;
        info.train      = false;
        info.scene      = nullptr;
        cache.info      = &info;

        LayerCuda layer;
        layer.size = ivec2(image_size, image_size);

        // [batch, channels, h, w]
        layer.counting = torch::randint(1, max_segment_length + 1, {1, 1, layer.size(1), layer.size(0)},
                                        torch::TensorOptions(torch::kCUDA).dtype<int>());
        layer.per_image_atomic_counters = layer.counting.clone();
        layer.scanned_counting          = torch::zeros_like(layer.counting);


        layer.depth     = torch::zeros({1, 1, layer.size(1), layer.size(0)}, torch::TensorOptions(torch::kCUDA));
        layer.weight    = torch::zeros({1, 1, layer.size(1), layer.size(0)}, torch::TensorOptions(torch::kCUDA));
        layer.max_depth = torch::zeros({1, 1, layer.size(1), layer.size(0)}, torch::TensorOptions(torch::kCUDA));
        cache.output_forward.push_back(
            torch::zeros({1, 1, layer.size(1), layer.size(0)}, torch::TensorOptions(torch::kCUDA)));

        auto countings         = layer.BatchViewCounting(0);
        auto scanned_countings = layer.BatchViewScannedCounting(0);

        thrust::device_ptr<int> thrust_offset_ptr         = thrust::device_pointer_cast(countings.data);
        thrust::device_ptr<int> thrust_scanned_counts_ptr = thrust::device_pointer_cast(scanned_countings.data);
        thrust::exclusive_scan(thrust_offset_ptr, thrust_offset_ptr + layer.size.x() * layer.size.y(),
                               thrust_scanned_counts_ptr);

        int num_elements = layer.counting.sum().item<int>();

        std::cout << "\n\nSortAlphaTest " << image_size << "x" << image_size << " length: " << max_segment_length
                  << std::endl;
        std::cout << "counting " << TensorInfo(layer.counting) << std::endl;
        std::cout << "scanned_counting " << TensorInfo(layer.scanned_counting) << std::endl;
        std::cout << "num elements: " << num_elements << std::endl;


        cache.layers_cuda.push_back(layer);
        SAIGA_ASSERT(cache.layers_cuda.size() == 1);

        std::vector<std::pair<float, int>> data;
        for (int i = 0; i < num_elements; ++i)
        {
            data.push_back({Random::sampleDouble(0, 1), Random::uniformInt(0, 100)});
        }


        // auto collection = torch::randint(100, {2, num_elements}, torch::TensorOptions(torch::kCUDA).dtype<int>());

        auto collection =
            torch::from_blob(data.data(), {num_elements, 2}, torch::TensorOptions().dtype<int>()).clone().cuda();
        collection = collection.transpose(1, 0).contiguous();
        collections.push_back(collection);

        cache.PushParametersForward();
    }

    void PrintData(torch::Tensor data)
    {
        std::cout << ">> Printing data " << TensorInfo(data) << std::endl;
        auto data_cpu = data.transpose(1, 0).cpu().contiguous();

        std::pair<float, int>* data_cpu_ptr = (std::pair<float, int>*)data_cpu.data_ptr();
        auto counts_cpu                     = cache.layers_cuda[0].counting.cpu().reshape({-1});
        auto counts_scan_cpu                = cache.layers_cuda[0].scanned_counting.cpu().reshape({-1});
        for (int k = 0; k < counts_cpu.size(0); ++k)
        {
            int offset = counts_scan_cpu.data_ptr<int>()[k];
            int count  = counts_cpu.data_ptr<int>()[k];

            std::cout << "list " << k << ": " << offset << " " << count << "\n";

            for (int i = offset; i < offset + count; ++i)
            {
                std::cout << data_cpu_ptr[i].first << " " << data_cpu_ptr[i].second << std::endl;
            }
        }

        // std::cout << counts_cpu << std::endl;
        // std::cout << counts_scan_cpu << std::endl;
    }

    bool CheckSortedCPU(torch::Tensor data)
    {
        auto data_cpu = data.transpose(1, 0).cpu().contiguous();

        std::pair<float, int>* data_cpu_ptr = (std::pair<float, int>*)data_cpu.data_ptr();
        auto counts_cpu                     = cache.layers_cuda[0].counting.cpu().reshape({-1});
        auto counts_scan_cpu                = cache.layers_cuda[0].scanned_counting.cpu().reshape({-1});

        // std::cout << "\nCheckSortedCPU\n";
        // std::cout << data_cpu.transpose(1, 0) << std::endl;

        bool is_sorted = true;

        for (int k = 0; k < counts_cpu.size(0); ++k)
        {
            int offset = counts_scan_cpu.data_ptr<int>()[k];
            int count  = counts_cpu.data_ptr<int>()[k];

            for (int i = offset; i < offset + count; ++i)
            {
                // std::cout << data_cpu_ptr[i].first << " " << data_cpu_ptr[i].second << std::endl;

                float current_value = data_cpu_ptr[i].first;

                // make sure we don't get our padding value in here
                EXPECT_LE(current_value, 1);

                if (i > offset)
                {
                    float last_value = data_cpu_ptr[i - 1].first;

                    if (last_value > current_value)
                    {
                        is_sorted = false;
                        std::cout << "> Sort fail at " << k << " " << i << " with values " << last_value << " > "
                                  << current_value << std::endl;
                    }
                }
            }
        }

        if (!is_sorted) PrintData(data);
        std::cout << "is_sorted: " << is_sorted << std::endl;
        return is_sorted;
    }

    torch::Tensor SortBitonic()
    {
        auto data = collections.front().clone();
        SegmentedSortBitonicHelper(cache.layers_cuda.front().counting.view({-1}),
                                   cache.layers_cuda.front().scanned_counting.view({-1}), data, timer);
        return data;
    }
    torch::Tensor SortBitonic2()
    {
        auto data = collections.front().clone();
        SegmentedSortBitonicHelper2(cache.layers_cuda.front().counting.view({-1}),
                                    cache.layers_cuda.front().scanned_counting.view({-1}), data, timer);
        return data;
    }
    torch::Tensor SortCub()
    {
        auto data = collections.front().clone();
        SegmentedSortCubHelper(cache.layers_cuda.front().counting.view({-1}),
                               cache.layers_cuda.front().scanned_counting.view({-1}), data, timer);
        return data;
    }
    std::vector<torch::Tensor> collections;
    NeuralRenderInfo info;
    PointRendererCache cache;
    CUDA::CudaTimerSystem* timer = nullptr;
};


std::vector<std::pair<int, int>> sort_test_cases = {{1, 8}, {1, 32}, {1, 1024}, {16, 16}, {16, 256}, {16, 4096}};


TEST(SortAlphaTest, SortBitonic2)
{
    // std::vector<std::pair<int, int>> sort_test_cases = {{16, 16}};
    for (auto [size, length] : sort_test_cases)
    {
        SortAlphaTest test(size, length);
        {
            auto data     = test.SortBitonic2();
            auto data_ref = test.SortCub();


            EXPECT_TRUE(test.CheckSortedCPU(data));
            EXPECT_TRUE(test.CheckSortedCPU(data_ref));

            PrintTensorInfo(data);
            PrintTensorInfo(data_ref);
            EXPECT_EQ((data - data_ref).abs().sum().item<int>(), 0);

            // EXPECT_TRUE(test.CheckSortedCPU(data));
        }
    }
}


TEST(SortAlphaTest, SortCub)
{
    for (auto [size, length] : sort_test_cases)
    {
        SortAlphaTest test(size, length);
        {
            auto data = test.SortCub();
            EXPECT_TRUE(test.CheckSortedCPU(data));
        }
    }
}

TEST(SortAlphaTest, SortBitonic)
{
    for (auto [size, length] : sort_test_cases)
    {
        SortAlphaTest test(size, length);
        {
            auto data = test.SortBitonic();
            EXPECT_TRUE(test.CheckSortedCPU(data));
        }
    }
}

TEST(SortAlphaTest, Benchmark)
{
    SortAlphaTest test(256, 1500);

    CUDA::CudaTimerSystem timer;
    auto timer_ptr = &timer;
    test.timer     = &timer;


    for (int i = 0; i < 10; ++i)
    {
        timer.BeginFrame();
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SortBitonic", timer_ptr);
            test.SortBitonic();
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SortBitonic2", timer_ptr);
            test.SortBitonic2();
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SortCub", timer_ptr);
            test.SortCub();
        }
        timer.EndFrame();
    }
    timer.PrintTable(std::cout);
}