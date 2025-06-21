/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/core/Core.h"

#include "SceneData.h"
#include "Settings.h"
#include "config.h"
#include "data/NeuralStructure.h"
#include "gaussian/gaussian.cuh"
#include "models/NeuralCamera.h"
#include "models/NeuralTexture.h"
#include "rendering/EnvironmentMap.h"
#include "rendering/NeuralPointCloudCuda.h"

#include "gaussian/cuda_rasterizer/rasterizer.h"


using namespace Saiga;

class NeuralScene
{
   public:
    NeuralScene(std::shared_ptr<SceneData> scene, std::shared_ptr<CombinedParams> params, bool eval_only = false);

    void BuildOutlierCloud(int n);

    void Train(int epoch_id, bool train);

    void to(torch::Device device)
    {
        if (environment_map)
        {
            environment_map->to(device);
        }
        texture->to(device);

        camera->to(device);
        intrinsics->to(device);
        poses->to(device);
        point_cloud_cuda->to(device);
    }

    void SaveCheckpoint(const std::string& dir, bool reduced);

    void LoadCheckpoint(const std::string& dir, bool eval_only = false);

    void CreateTextureOptimizer();

    void CreateTexture(bool eval_only = false);

    void CreateStructureOptimizer();

    void ShrinkTextureOptimizer(torch::Tensor indices_to_keep);

    void AppendToTextureOptimizer(int new_size);

    void Log(const std::string& log_dir);

    void OptimizerStep(int epoch_id, bool structure_only);

    void OptimizerClear(int epoch_id, bool structure_only);

    void UpdateLearningRate(int epoch_id, double factor);

    void ShrinkPCOptimizer(torch::Tensor indices_to_keep);
    void RemovePointsWithConfUnder(float conv_threshold, bool reset_optimizer);
    // Download + Save in 'scene'
    void DownloadIntrinsics();

    void DownloadPoses();

    void AddPointsViaPointGrowing(int factor = 2, float distance = 1.f, bool update_optimizer = true);

    void AddNewPoints(std::vector<vec3> positions, std::vector<vec3> normal);

    void AddNewPoints(torch::Tensor random_values, bool update_optimizer = true);
    void AddNewPoints(std::vector<vec3>& positions, bool update_optimizer = true);

    void AddNewRandomForEnvSphere(int num_spheres, float inner_radius, float env_radius_factor, int num_points,
                                  bool update_optimizer = true);

    void AddNewRandomPoints(float factor);

    void AddNewRandomPointsInValuefilledBB(int num_points_to_add, float percent_of_boxes = 0.05);

    void AddNewRandomPointsFromCTStack(int num_points_to_add, std::string path, float ct_volume_scale = 5.f,
                                       vec3 ct_volume_translation = vec3(0, 0, 1));

    void AddNewRandomPointsFromCTHdr(torch::Tensor hdr_img_stack, int max_num_points_to_add, float ct_volume_scale,
                                     vec3 ct_volume_translation, AABB aabb);

    void RemovePoints(torch::Tensor indices, bool update_optimizer = true);

    void Print()
    {
        std::cout << "Pos" << TensorInfo(point_cloud_cuda->t_position) << std::endl;

        std::cout << "Pointsize" << TensorInfo(point_cloud_cuda->t_point_size) << std::endl;
        std::cout << "texture" << TensorInfo(texture->texture) << std::endl;

        std::cout << "conf" << TensorInfo(texture->confidence_value_of_point) << std::endl;
    }

   public:
    friend class NeuralPipeline;

    std::shared_ptr<SceneData> scene;
    bool baked_for_inference = false;

    NeuralPointCloudCuda point_cloud_cuda = nullptr;
    NeuralPointTexture texture            = nullptr;

    std::shared_ptr<GaussianModel> gaussian_model = nullptr;
    torch::Tensor gaussian_background;
    gs::param::OptimizationParameters optimParams;
    gs::param::ModelParameters modelParams;
    CudaRasterizer::SplattingSettings stp_splatting_args;
    float cameras_extent = 1.f;

    EnvironmentMap environment_map = nullptr;
    NeuralCamera camera            = nullptr;
    PoseModule poses               = nullptr;
    IntrinsicsModule intrinsics    = nullptr;

    std::shared_ptr<torch::optim::Optimizer> camera_adam_optimizer, camera_sgd_optimizer;
    std::shared_ptr<torch::optim::Optimizer> texture_optimizer;
    std::shared_ptr<torch::optim::Optimizer> structure_optimizer;

    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<CombinedParams> params;
};
