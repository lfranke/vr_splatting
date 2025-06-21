/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "../lib/config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"
#include "gtest/gtest.h"
#include "models/mlp.h"
#include "rendering/NeuralPointCloudCuda.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderInfo.h"

#include "models/spherical_harmonics.h"
#include <tiny-cuda-nn/torch_wrapper.h>

using namespace Saiga;
using namespace tcnn;
using namespace nlohmann;

TORCH_LIBRARY(asdfasdf, m)
{
    std::cout << "register TnnInfo" << std::endl;
    m.class_<TnnInfo>("TnnInfo").def(torch::init());
}


TEST(TCNN, spherical_harmonics)
{
    std::cout << "SPHERICAL HARMONICS TEST" << std::endl;
    auto sh         = SphericalHarmonicsEncoding(3);
    torch::Tensor t = torch::ones({16 * 65536, 3}).to(torch::kCUDA);
    std::cout << TensorInfo(t) << std::endl;
    auto res = sh.forward(t);
    std::cout << TensorInfo(res) << std::endl;
    std::cout << "SPHERICAL HARMONICS TEST DONE" << std::endl;
}



TEST(TCNN, work)
{
    // Configure the model
    json config = {
        {"loss", {{"otype", "RelativeL2"}}},
        {"optimizer",
         {
             {"otype", "Adam"},
             // {"otype", "Shampoo"},
             {"learning_rate", 1e-2},
             {"beta1", 0.9f},
             {"beta2", 0.99f},
             {"l2_reg", 0.0f},
             // The following parameters are only used when the optimizer is "Shampoo".
             {"beta3", 0.9f},
             {"beta_shampoo", 0.0f},
             {"identity", 0.0001f},
             {"cg_on_momentum", false},
             {"frobenius_normalization", true},
         }},
        {"encoding",
         {
             {"otype", "OneBlob"},
             {"n_bins", 32},
         }},
        //{"encoding",
        // {
        //     {"otype", "Frequency"},
        //     {"n_frequencies", 16},
        // }},
        {"network",
         {
             {"otype", "FullyFusedMLP"},
             // {"otype", "CutlassMLP"},
             {"n_neurons", 64},
             {"n_hidden_layers", 1},
             {"activation", "ReLU"},
             {"output_activation", "None"},
         }},
    };

    std::cout << "MLP TEST" << std::endl;

    int n_input_dims     = 3;
    int n_output_dims    = 3;
    int batch_size       = 65536;
    int n_training_steps = 500;

    json encoding_opts = config.value("encoding", json::object());
    std::cout << encoding_opts << std::endl;
    json network_opts = config.value("network", json::object());
    std::cout << network_opts << std::endl;

    tcnn::cpp::Module* net_module =
        tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding_opts, network_opts);

    std::cout << "network input/output: " << net_module->n_input_dims() << "/" << net_module->n_output_dims()
              << std::endl;

    TcnnTorchModule test_module = nullptr;
    test_module                 = TcnnTorchModule(TorchTcnnWrapperModule(net_module));

    torch::Tensor target = torch::full({batch_size, n_output_dims}, 10.f).to(torch::kCUDA);
    torch::Tensor pred   = torch::rand({batch_size, n_input_dims}).to(torch::kCUDA);
    // pred.requires_grad_();

    std::cout << TensorInfo(test_module->params) << std::endl;

    auto opt = std::make_unique<torch::optim::AdamOptions>(0.001);
    std::vector<torch::optim::OptimizerParamGroup> g_optim;
    g_optim.emplace_back(test_module->parameters(), std::move(opt));

    std::cout << "PARAMETERS: " << TensorInfo(test_module->params) << std::endl;

    auto adam_optimizer = std::make_shared<torch::optim::Adam>(g_optim, torch::optim::AdamOptions(1));

    adam_optimizer->zero_grad();
    for (int i = 0; i < n_training_steps; ++i)
    {
        std::cout << "before: " << TensorInfo(pred) << std::endl;

        auto x = test_module->forward(pred).to(torch::kFloat).slice(1, 0, n_output_dims);
        // for (int dim = 0; dim < x.size(1); ++dim) std::cout << TensorInfo(x.slice(1, dim, dim + 1)) << std::endl;
        std::cout << "after: " << TensorInfo(x) << std::endl;
        auto lt_l1 = torch::abs(x - target).sum() / 1000.f;
        std::cout << "Loss: " << TensorInfo(lt_l1) << std::endl;

        std::cout << "===================" << std::endl;
        lt_l1.backward();
        // for (int dim = 0; dim < x.size(1); ++dim) std::cout << TensorInfo(x.grad().slice(1, dim, dim + 1)) <<
        // std::endl;
        std::cout << "PARAMETERS GRAD: " << TensorInfo(test_module->params.grad()) << std::endl;

        adam_optimizer->step();
        adam_optimizer->zero_grad();
        std::cout << "PARAMETERS: " << TensorInfo(test_module->params) << std::endl;
    }
    std::cout << "MLP TEST DONE" << std::endl;
}


TEST(TCNN, own_mlp)
{
    // Configure the model
    json config = {
        // {"encoding",
        //  {
        //      {"otype", "OneBlob"},
        //      {"n_bins", 32},
        //  }},
        {"encoding",
         {
             {"otype", "Frequency"},
             {"n_frequencies", 8},
         }},
    };

    int n_input_dims     = 4;
    int n_output_dims    = 3;
    int batch_size       = 65536;
    int n_training_steps = 500;

    json encoding_opts = config.value("encoding", json::object());
    std::cout << encoding_opts << std::endl;

    auto precision                  = tcnn::cpp::EPrecision::Fp32;
    tcnn::cpp::Module* enc_module   = tcnn::cpp::create_encoding(n_input_dims, encoding_opts, precision);
    TcnnTorchModule encoding_module = nullptr;
    encoding_module                 = TcnnTorchModule(TorchTcnnWrapperModule(enc_module));

    FCBlock mlp(64, 3, 3, 256);
    mlp->to(torch::kCUDA);
    for (auto& p : mlp->parameters()) std::cout << "PARAMETERS: " << TensorInfo(p) << std::endl;

    torch::Tensor target = torch::full({batch_size, n_output_dims}, 10.5f).to(torch::kCUDA);
    torch::Tensor pred   = torch::rand({batch_size, n_input_dims}).to(torch::kCUDA);


    std::vector<torch::optim::OptimizerParamGroup> g_optim;
    {
        auto opt = std::make_unique<torch::optim::AdamOptions>(0.01);
        g_optim.emplace_back(mlp->parameters(), std::move(opt));
    }
    auto adam_optimizer = std::make_shared<torch::optim::Adam>(g_optim, torch::optim::AdamOptions(1));
    for (int i = 0; i < n_training_steps; ++i)
    {
        std::cout << TensorInfo(pred) << std::endl;
        auto encoded_f = encoding_module->forward(pred);
        std::cout << TensorInfo(encoded_f) << std::endl;
        auto x = mlp->forward(encoded_f);
        std::cout << TensorInfo(x) << std::endl;

        auto lt_l1 = torch::abs(x - target).sum();
        std::cout << "Loss: " << TensorInfo(lt_l1) << std::endl;

        lt_l1.backward();

        adam_optimizer->step();
        adam_optimizer->zero_grad();
    }
}