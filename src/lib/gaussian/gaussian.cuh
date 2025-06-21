// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and
// MPII.

#pragma once

#include "saiga/core/Core.h"
#include "saiga/core/camera/HDR.h"
#include "saiga/core/math/CoordinateSystems.h"
#include "saiga/core/sophus/Sophus.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/cameraModel/Intrinsics4.h"
#include "saiga/vision/cameraModel/OCam.h"

#include "parameters.cuh"

#include <memory>
#include <string>
#include <torch/torch.h>

#include "general_utils.cuh"
#include "simple-knn/spatial.h"
// #include "point_cloud.cuh"
#include "sh_utils_gs.cuh"

using namespace Saiga;

class GaussianModel
{
   public:
    explicit GaussianModel(int sh_degree);
    // Copy constructor
    GaussianModel(const GaussianModel& other) = delete;
    // Copy assignment operator
    GaussianModel& operator=(const GaussianModel& other) = delete;
    // Move constructor
    GaussianModel(GaussianModel&& other) = default;
    // Move assignment operator
    GaussianModel& operator=(GaussianModel&& other) = default;

   public:
    // Getters
    inline torch::Tensor Get_xyz() const { return _xyz; }
    inline torch::Tensor Get_opacity() const { return inference_baked ? _opacity : torch::sigmoid(_opacity); }
    inline torch::Tensor Get_rotation() const
    {
        return inference_baked ? _rotation : torch::nn::functional::normalize(_rotation);
    }
    torch::Tensor Get_features() const;
    torch::Tensor Get_covariance(float scaling_modifier = 1.0);
    int Get_active_sh_degree() const { return _active_sh_degree; }
    int Get_max_sh_degree() const { return _max_sh_degree; }
    torch::Tensor Get_scaling() { return inference_baked ? _scaling : torch::exp(_scaling); }
    void Bake_for_inference();

    // Methods
    void One_up_sh_degree();
    void Create_from_pcd(const Saiga::UnifiedMesh& pcd, float spatial_lr_scale);
    void Training_setup(const gs::param::OptimizationParameters& params);
    void CreateOptimizer(const gs::param::OptimizationParameters& params);
    void Update_learning_rate(float iteration);
    void Reset_opacity();
    void Decay_opacity(float decay_value = 0.999f);
    void Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter, bool abs_grad);
    void Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size);
    void Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration);
    std::pair<vec3, vec3> Get_aabb();
    void printModel(bool include_optimizer = false);

    void SaveCheckpoint(const std::string& checkpoint_prefix, bool with_optimizer = false);
    void LoadCheckpoint(const std::string& checkpoint_dir, bool with_optimizer = false);
    void LoadCheckpointForTraining(const std::string& checkpoint_dir, const gs::param::OptimizationParameters& params);


    std::shared_ptr<torch::optim::Adam> _optimizer;
    torch::Tensor _max_radii2D;

   private:
    void prune_points(torch::Tensor mask);
    void densification_postfix(torch::Tensor& new_xyz, torch::Tensor& new_features_dc, torch::Tensor& new_features_rest,
                               torch::Tensor& new_scaling, torch::Tensor& new_rotation, torch::Tensor& new_opacity);

    void densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent);
    void densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity,
                           float max_screen_size);
    std::vector<std::string> construct_list_of_attributes();

   private:
    int _active_sh_degree   = 0;
    int _max_sh_degree      = 0;
    float _spatial_lr_scale = 0.f;
    float _percent_dense    = 0.f;
    gs::param::OptimizationParameters _optim_params;

    bool inference_baked = false;
    Expon_lr_func _xyz_scheduler_args;

    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;

    torch::Tensor _xyz_gradient_accum;
    torch::Tensor _xyz_gradient_accum_new;
    torch::Tensor _denom;

    torch::Tensor _features_inference;
    torch::Tensor _cov_inference;
};
