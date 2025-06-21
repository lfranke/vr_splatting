﻿/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once
#include "saiga/core/Core.h"

#include "SceneData.h"
#include "Settings.h"
#include "config.h"

#include <torch/torch.h>


using namespace Saiga;


class PoseModuleImpl : public torch::nn::Module
{
   public:
    PoseModuleImpl(std::shared_ptr<SceneData> scene);

    // Adds only a single pose
    PoseModuleImpl(Sophus::SE3d pose);

    // empty with size amount
    explicit PoseModuleImpl(int amount);

    std::vector<Sophus::SE3d> Download();
    Sophus::SE3d Download(int index_pose);
    void SetPose(int id, Sophus::SE3d pose);
    torch::Tensor GetPose(int id);
    torch::Tensor GetPoseMat4(int id);

    // double: [num_cameras, 8]
    torch::Tensor poses_se3;

    // float: [num_cameras, 4, 4]
    torch::Tensor poses_mat4;

    // double: [num_cameras, 6]
    torch::Tensor tangent_poses;

    void ApplyTangent();
};
TORCH_MODULE(PoseModule);

std::pair<vec3, float> getNerfppNorm(std::shared_ptr<SceneData> scene);

class IntrinsicsModuleImpl : public torch::nn::Module
{
   public:
    IntrinsicsModuleImpl(std::shared_ptr<SceneData> scene);

    // Adds only a single intrinsic
    IntrinsicsModuleImpl(IntrinsicsPinholef K);

    IntrinsicsModuleImpl(int amount);

    std::vector<Distortionf> DownloadDistortion();
    std::vector<IntrinsicsPinholef> DownloadK();


    void SetPinholeIntrinsics(int id, IntrinsicsPinholef K, Distortionf dis);
    void SetOcamIntrinsics(int id, OCam<float> ocam);

    void AddToOpticalCenter(float x);

    // [num_cameras, num_model_params]
    // Pinhole + Distortion: [num_cameras, 5 + 8]
    // OCam:                 [num_cameras, 5 + world2cam_coefficients]
    torch::Tensor intrinsics;
};
TORCH_MODULE(IntrinsicsModule);
