/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "NeuralPointCloudCuda.h"

#include "saiga/normal_packing.h"

inline float _softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    //  return x * beta > threshold ? x : logf(1 + expf(beta * x));
    if (x > threshold) return x;
    return logf(1.f + expf(x * beta)) / beta;
}


inline float inverse_softplus(float x, float beta = 1.f, float threshold = 20.f)
{
    if (x > threshold) return x;

    return log(exp(x * beta) - 1) / beta;
}

//
// #define USE_LAYER_EXPL


NeuralPointCloudCudaImpl::NeuralPointCloudCudaImpl(const UnifiedMesh& model, bool use_grid_loss, float cell_ws_size,
                                                   AABB custom_point_manip_aabb, bool use_pointsize)
    : NeuralPointCloud(model)
{
    std::vector<vec4> data_position;
    std::vector<int> data_normal_compressed;
    // std::vector<vec4> original_colors;

    std::vector<float> data_point_size;

    std::vector<int32_t> indices;
    for (int i = 0; i < points.size(); ++i)
    {
        indices.push_back(points[i].index);

        float drop_out_radius = 0;
        if (data.size() == points.size())
        {
            drop_out_radius = data[i](3);
        }
        data_point_size.push_back(inverse_softplus(data[i](0) * 0.5f));

        //      original_colors.push_back(color[i]);
        data_position.push_back(make_vec4(points[i].position, drop_out_radius));
    }
    t_position = torch::from_blob(data_position.data(), {(long)data_position.size(), 4},
                                  torch::TensorOptions().dtype(torch::kFloat32))
                     .contiguous()
                     .cuda()
                     .clone();

    // t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1},
    // torch::TensorOptions().dtype(torch::kInt32))
    //               .contiguous()
    //               .cuda()
    //               .clone();
    t_index      = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32)).contiguous().cuda().clone();
    t_point_size = torch::from_blob(data_point_size.data(), {(long)data_point_size.size(), 1},
                                    torch::TensorOptions().dtype(torch::kFloat32))
                       .contiguous()
                       .cuda()
                       .clone();

    register_parameter("t_position", t_position);
    register_buffer("t_index", t_index);

    if (use_pointsize) register_parameter("t_point_size", t_point_size);

    SAIGA_ASSERT(t_position.isfinite().all().item().toBool());

    // size_t total_mem = t_position.nbytes() + t_index.nbytes();

    // std::cout << "GPU memory - Point Cloud: " << total_mem / 1000000.0 << "MB" << std::endl;
}

// void NeuralPointCloudCudaImpl::Reorder(torch::Tensor indices){
//     torch::NoGradGuard ngg;
//
//     SAIGA_ASSERT(indices.sizes().size() == 1);
//
//     t_position = t_position.index({indices}).contiguous();
//     t_index = t_index.index({indices}).contiguous();
//     t_normal = t_normal.index({indices}).contiguous();
//
//
// }


Saiga::UnifiedMesh NeuralPointCloudCudaImpl::Mesh()
{
    Saiga::UnifiedMesh mesh;

    std::cout << "Extracing Point Cloud from device data" << std::endl;
    std::cout << "Tensors included:" << std::endl;
    // Position
    PrintTensorInfo(t_position);
    std::vector<vec4> data_position(t_position.size(0), vec4(-1, -1, -1, -1));
    torch::Tensor cp_position = t_position.contiguous().cpu();
    memcpy(data_position[0].data(), cp_position.data_ptr(), sizeof(vec4) * data_position.size());

    for (auto p : data_position)
    {
        mesh.position.push_back(p.head<3>());
        mesh.data.push_back(vec4(0, 0, 0, p(3)));
    }
    std::cout << "End of tensors included:" << std::endl;

    return mesh;
}
std::vector<int> NeuralPointCloudCudaImpl::Indices()
{
    std::vector<int> indices(t_index.size(0));

    torch::Tensor cp_index = t_index.contiguous().cpu();

    memcpy(indices.data(), cp_index.data_ptr(), sizeof(int) * indices.size());

    return indices;
}
void NeuralPointCloudCudaImpl::SetIndices(std::vector<int>& indices)
{
#if 1
    // torch::NoGradGuard  ngg;
    t_index.set_data(
        torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
            .contiguous()
            .cuda()
            .clone());
#else
    t_index = torch::from_blob(indices.data(), {(long)indices.size(), 1}, torch::TensorOptions().dtype(torch::kFloat32))
                  .contiguous()
                  .cuda()
                  .clone();
#endif
}
int NeuralPointCloudCudaImpl::Size()
{
    SAIGA_ASSERT(t_position.defined());
    return t_position.size(0);
}
