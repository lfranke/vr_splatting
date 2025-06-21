//
// Created by Linus on 27/08/2024.
//
#pragma once
#include "saiga/vision/torch/torch.h"

torch::Tensor rgb_to_xyz(const torch::Tensor rgb);

torch::Tensor xyz_to_rgb(const torch::Tensor xyz);

torch::Tensor rgb_to_lab(const torch::Tensor rgb);

torch::Tensor rgb_to_lab_normalized(const torch::Tensor rgb);

torch::Tensor lab_to_rgb(const torch::Tensor lab);

torch::Tensor srgb_to_lab(const torch::Tensor srgb);

torch::Tensor srgb_to_lab_normalized(const torch::Tensor srgb);
