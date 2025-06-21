/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/math/math.h"
#include "saiga/core/sophus/Sophus.h"
#include "saiga/core/util/ProgressBar.h"

#include "data/Dataset.h"
#include "models/Pipeline.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>

std::shared_ptr<CombinedParams> params;
using namespace Saiga;
using namespace torch::indexing;

int num_points_passing_fuzzy_dp = 2000;
int max_frames                  = 308;

int num_texture_channels                       = 4;
int num_additional_direction_based_descriptors = 4;


double all_facs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


inline Saiga::vec6 get_oct_factors(vec3 dir)
{
    vec6 oct_factors;
    // l1 norm on appearence vector
    float l1 = abs(dir(0)) + abs(dir(1)) + abs(dir(2));

    dir = dir / l1;  // dir.lpNorm<1>();
    // project on octahedron
    oct_factors[0] = fmax(0.f, dir(0));
    oct_factors[1] = fmax(0.f, -dir(0));
    oct_factors[2] = fmax(0.f, dir(1));
    oct_factors[3] = fmax(0.f, -dir(1));
    oct_factors[4] = fmax(0.f, dir(2));
    oct_factors[5] = fmax(0.f, -dir(2));

    return oct_factors;
}

inline Saiga::vec3 get_reflection_dir(vec3 dir, vec3 normal)
{
    vec3 d = dir.normalized();
    vec3 n = normal.normalized();
    vec3 r = d - 2 * (d.dot(n)) * n;
    return r;
}

inline Saiga::mat3 getTBN(vec3 normal)
{
    vec3 n            = normal.normalized();
    vec3 bi_tangent_f = vec3(0, 1, 0);
    if (bi_tangent_f == n) bi_tangent_f = vec3(1, 0, 0);
    vec3 tangent   = (n.cross(bi_tangent_f)).normalized();
    vec3 bitangent = (n.cross(tangent)).normalized();

    mat3 TBN;  //= mat3(tangent, bitangent, normal);
    TBN.col(0) = tangent;
    TBN.col(1) = bitangent;
    TBN.col(2) = normal;
    return TBN;
}



torch::Tensor get_contrib_from_dir(std::vector<torch::Tensor>& neural_descs, std::vector<vec3>& pos_vertices,
                                   vec3 wpos_cam, std::vector<vec3>& normal_vertices)
{
    float weight         = 1.f / float(num_points_passing_fuzzy_dp);
    torch::Tensor result = torch::zeros({num_texture_channels + num_additional_direction_based_descriptors});
    for (int x = 0; x < num_points_passing_fuzzy_dp; ++x)
    {
        //   std::cout << pos_vertives[x] << std::endl<< std::endl;
        result.index({Slice(0, num_texture_channels)}) += neural_descs[x].index({Slice(0, num_texture_channels)});

        // mat3 tbn = getTBN(normal_vertices[x]);
        // std::cout << (tbn.inverse() * normal_vertices[x]).normalized() << std::endl;

        vec3 wdir_to_p = pos_vertices[x] - wpos_cam;
        {
            vec6 oct_factors = get_oct_factors(wdir_to_p);


            for (int i = 0; i < 6; ++i)
            {
                all_facs[i] += oct_factors[i] / (max_frames * num_points_passing_fuzzy_dp);
                // std::cout << "oct _factor forward" << i << ": " << oct_factors[i] << std::endl;
            }


            for (int i = 0; i < 6; ++i)
            {
                int ith_channel_start = num_texture_channels + i * num_additional_direction_based_descriptors;
                int ith_channel_end   = ith_channel_start + num_additional_direction_based_descriptors;
                // if(oct_factors[i]>0){
                for (int ci = ith_channel_start; ci < ith_channel_end; ++ci)
                {
                    int write_out_channel = num_texture_channels + (ci % num_additional_direction_based_descriptors);
                    auto t                = oct_factors[i] * neural_descs[x].index({Slice(ci, ci + 1)});
                    result.index({Slice(write_out_channel, write_out_channel + 1)}) += t;
                }
            }
        }
    }
    result *= weight;
    return result;
}

std::vector<torch::Tensor> get_contrib_from_dir_backwards(std::vector<torch::Tensor>& neural_descs,
                                                          std::vector<vec3>& pos_vertices, vec3 wpos_cam,
                                                          torch::Tensor in_gradients)
{
    float iw = 1.f / float(num_points_passing_fuzzy_dp);
    std::vector<torch::Tensor> grads_ret;
    for (int x = 0; x < num_points_passing_fuzzy_dp; ++x)
    {
        vec3 wdir_to_p = pos_vertices[x] - wpos_cam;


        torch::Tensor grads_result =
            torch::zeros({num_texture_channels + 6 * num_additional_direction_based_descriptors});
        for (int ci = 0; ci < num_texture_channels; ++ci)
        {
            float g = iw * in_gradients[ci].item<float>();
            // atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);

            grads_result[ci] += g;
        }
        // l1 norm on appearence vector

        vec6 oct_factors = get_oct_factors(wdir_to_p);

        for (int i = 0; i < 6; ++i)
        {
            int ith_channel_start = num_texture_channels + i * num_additional_direction_based_descriptors;
            int ith_channel_end   = ith_channel_start + num_additional_direction_based_descriptors;
            for (int ci = ith_channel_start; ci < ith_channel_end; ++ci)
            {
                int grad_in_channel = num_texture_channels + (ci % num_additional_direction_based_descriptors);
                float g             = iw * oct_factors[i] * in_gradients[grad_in_channel].item<float>();
                // atomicAdd(&d_backward_params.out_gradient_texture(ci, texture_index), g);
                grads_result[ci] += g;
                //    std::cout << ci << " " <<g <<  " " << grads_result.index({(ci)}).item<float>() <<std::endl;
            }
        }
        grads_ret.push_back(grads_result);
    }
    return grads_ret;
}



int main(int argc, char* argv[])
{
    std::cout << "test oct grad" << std::endl;

    /// LOAD SCENE
    std::string config_file;
    CLI::App app{"test oct gradient", "oct_grad_test"};
    app.add_option("--config", config_file)->required();
    CLI11_PARSE(app, argc, argv);
    console << "Testing Config: " << config_file << std::endl;
    SAIGA_ASSERT(std::filesystem::exists(config_file));
    params = std::make_shared<CombinedParams>(config_file);
    params->Check();
    auto scene = std::make_shared<SceneData>(params->train_params.scene_base_dir + params->train_params.scene_names[0]);
    //\ LOAD SCENE



    std::cout << scene->Frame(0).pose << std::endl;
    std::cout << scene->Frame(0).pose.matrix() << std::endl;
    std::cout << scene->Frame(0).pose.inverse() << std::endl;
    std::cout << scene->Frame(0).pose.inverse().matrix() << std::endl;

    bool all_ok = true;

    Saiga::ProgressBar bar(std::cout, "Points  " + std::to_string(0) + " |", max_frames, 30, false, 5000);

#pragma omp parallel for
    for (int frame_id = 0; frame_id < max_frames; ++frame_id)
    {
        std::vector<torch::Tensor> neural_descs;
        for (int i = 0; i < num_points_passing_fuzzy_dp; ++i)
        {
            neural_descs.push_back(torch::randn({num_texture_channels + 6 * num_additional_direction_based_descriptors},
                                                torch::requires_grad()));
        }

        auto pos_vertices = std::vector<vec3>(scene->point_cloud.position.begin(),
                                              scene->point_cloud.position.begin() + num_points_passing_fuzzy_dp);

        auto normal_vertices = std::vector<vec3>(scene->point_cloud.normal.begin(),
                                                 scene->point_cloud.normal.begin() + num_points_passing_fuzzy_dp);



        // std::sample(
        //     scene->point_cloud.position.begin(),
        //     scene->point_cloud.position.end(),
        //     std::back_inserter(pos_vertices),
        //     num_points_passing_fuzzy_dp,
        //     std::mt19937{std::random_device{}()}
        //);


        auto V        = scene->Frame(frame_id).pose.cast<float>().inverse();
        vec3 wpos_cam = V.inverse() * vec3(0, 0, 0);
#if 0

        std::cout << "wpos_cam " << wpos_cam << std::endl<< std::endl;
#endif
        torch::Tensor result = torch::zeros({num_texture_channels + num_additional_direction_based_descriptors});


        torch::Tensor input_gradient = torch::rand({num_texture_channels + num_additional_direction_based_descriptors});

        result = get_contrib_from_dir(neural_descs, pos_vertices, wpos_cam, normal_vertices);
        result.backward(input_gradient);
        auto own_res = get_contrib_from_dir_backwards(neural_descs, pos_vertices, wpos_cam, input_gradient);
#if 0
        for(int i=0; i<num_points_passing_fuzzy_dp; ++i){
            std::cout << "torch "<<i <<":" << neural_descs[i].grad() << std::endl;
        }
        for(int i=0; i<num_points_passing_fuzzy_dp; ++i){
            std::cout << "own "<<i <<":" << own_res[i] << std::endl;
        }
#endif
        bool grad_ok = true;
        for (int i = 0; i < num_points_passing_fuzzy_dp; ++i)
        {
            auto sum = torch::abs(own_res[i] - neural_descs[i].grad());
            //  std::cout << "sum" << sum << std::endl;
            grad_ok &= torch::all(torch::less_equal(sum, 0.0001)).item<bool>();
            if (!grad_ok)
            {
                std::cout << "grad not ok " << std::endl;  // << own_res[i] << neural_descs[i].grad() << std::endl;
            }
        }
#if 0
        if(grad_ok)
            std::cout << "grad ok" << std::endl;
#endif

        all_ok &= grad_ok;
        bar.addProgress(1);
        bar.SetPostfix(" x");
    }
    if (all_ok) std::cout << "all grad ok" << std::endl;

    for (int i = 0; i < 6; ++i)
    {
        std::cout << "accumulated oct factor " << i << ": " << all_facs[i] << std::endl;
        std::cout << "averaged oct factor " << i << ": " << all_facs[i] << std::endl;
    }
}

/*
torch 0: 0.1000
 0.0024
 0.0000
 0.0109
 0.0000
 0.0000
 0.0867
[ CPUFloatType{7} ]
torch 1: 0.1000
 0.0024
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 2: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0867
[ CPUFloatType{7} ]
torch 3: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 4: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 5: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 6: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 7: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 8: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
torch 9: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 0: 0.1000
 0.0024
 0.0000
 0.0109
 0.0000
 0.0000
 0.0867
[ CPUFloatType{7} ]
own 1: 0.1000
 0.0024
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 2: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0867
[ CPUFloatType{7} ]
own 3: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 4: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 5: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 6: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 7: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 8: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
own 9: 0.1000
 0.0025
 0.0000
 0.0109
 0.0000
 0.0000
 0.0866
[ CPUFloatType{7} ]
*/