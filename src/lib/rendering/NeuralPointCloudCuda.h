/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/cuda/cuda.h"
#include "saiga/cuda/thrust_helper.h"
#include "saiga/normal_packing.h"
#include "saiga/vision/torch/TorchHelper.h"

#include "NeuralPointCloud.h"

#include <torch/torch.h>

#include "cuda_fp16.h"



class NeuralPointCloudCudaImpl : public NeuralPointCloud, public torch::nn::Module
{
   public:
    NeuralPointCloudCudaImpl(const Saiga::UnifiedMesh& model, bool use_grid_loss = false, float cell_ws_size = 1.f,
                             AABB custom_point_manip_aabb = AABB(), bool use_pointsize = true);

    std::vector<int> Indices();
    void SetIndices(std::vector<int>& indices);

    /* Following [Schütz et al. 22], 80*128 conseq points */
    void UpdateCellStructureForRendering(size_t conseq_points = 10240);

    void UpdateCellStructureForPointOptim(float size_of_box_in_ws, AABB custom_aabb = AABB());

    torch::Tensor GetPerPointBBIndex();
    torch::Tensor GetPerPointBBValue();
    torch::Tensor DebugBBIndexToCol();

    std::vector<vec3> DebugColorsPerBoxCPU();


    void RemoveSelected(torch::Tensor to_keep)
    {
        torch::NoGradGuard ngg;
        std::cout << TensorInfo(to_keep) << std::endl;
        std::cout << TensorInfo(t_position) << std::endl;
        t_position = t_position.index({to_keep});
        std::cout << TensorInfo(t_position) << std::endl;
        t_point_size = t_point_size.index({to_keep});
        std::cout << TensorInfo(t_point_size) << std::endl;
        // std::cout << TensorInfo(t_index) << std::endl;
        // t_index = t_index.index({to_keep});
        // std::cout << TensorInfo(t_index) << std::endl;
        //  std::cout << TensorInfo(t_original_index) << std::endl;
        //  t_original_index = t_original_index.index({to_keep});
        //  std::cout << TensorInfo(t_original_index) << std::endl;
    }

    torch::Tensor DebugColorsPerBox();
    // void Reorder(torch::Tensor indices);

    int Size();
    Saiga::UnifiedMesh Mesh();

    // [n, 4]
    torch::Tensor t_position;

    //[n,1]
    torch::Tensor t_point_size;

    // [n, 1]
    torch::Tensor t_index;

    using PointType  = vec4;
    using NormalType = vec4;
};


TORCH_MODULE(NeuralPointCloudCuda);


// A simple helper class to make the kernels more compact.
struct DevicePointCloud
{
    float4* __restrict__ position;
    // int* __restrict__ index;
    float* __restrict__ point_size;

    int n;

    DevicePointCloud() = default;

    DevicePointCloud(NeuralPointCloudCuda pc)
    {
        // SAIGA_ASSERT(pc->t_position.size(0) == pc->t_index.size(0));
        SAIGA_ASSERT(pc->t_position.size(0) == pc->Size());

        position = (float4*)pc->t_position.data_ptr<float>();

        // index = (int*)pc->t_index.data_ptr();

        if (pc->t_point_size.defined())
        {
            point_size = pc->t_point_size.data_ptr<float>();
        }
        else
        {
            point_size = nullptr;
        }

        n = pc->Size();
    }

    HD inline thrust::tuple<vec3, float> GetPointWoNormal(int point_index) const
    {
        vec4 p;

        // float4 global memory loads are vectorized!
        float4 p_f4 = position[point_index];
        p           = reinterpret_cast<vec4*>(&p_f4)[0];

        float drop_out_radius = p(3);

        vec3 pos = p.head<3>();

        return {pos, drop_out_radius};
    }

    // HD inline int GetIndex(int tid) const { return index[tid]; }
    HD inline float GetPointSize(int tid) const { return point_size[tid]; }

    // HD inline void SetIndex(int tid, int value) const { index[tid] = value; }

    HD inline int Size() const { return n; }
};