/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/torch.h>

#ifdef snprintf
#    undef snprintf
#endif

#include "nlohmann/json.hpp"

#include <cstdio>
#include <string>
#include <tuple>

#include "cuda_rasterizer/rasterizer.h"

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeGaussiansCUDA(
    const torch::Tensor& background, const torch::Tensor& means3D, const torch::Tensor& colors,
    const torch::Tensor& opacity, const torch::Tensor& scales, const torch::Tensor& rotations,
    const float scale_modifier, const torch::Tensor& cov3D_precomp, const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix, const torch::Tensor& inv_viewprojmatrix, const float tan_fovx,
    const float tan_fovy, const int image_height, const int image_width, const torch::Tensor& sh, const int degree,
    const torch::Tensor& campos, const bool prefiltered, const CudaRasterizer::SplattingSettings& settings,
    const bool render_depth, const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& background, const torch::Tensor& means3D,
                               const torch::Tensor& radii, const torch::Tensor& opacities, const torch::Tensor& colors,
                               const torch::Tensor& scales, const torch::Tensor& rotations, const float scale_modifier,
                               const torch::Tensor& cov3D_precomp, const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix, const torch::Tensor& inv_viewprojmatrix,
                               const float tan_fovx, const float tan_fovy, const torch::Tensor& pixel_colors,
                               const torch::Tensor& dL_dout_color, const torch::Tensor& sh, const int degree,
                               const torch::Tensor& campos, const torch::Tensor& geomBuffer, const int R,
                               const torch::Tensor& binningBuffer, const torch::Tensor& imageBuffer,
                               const CudaRasterizer::SplattingSettings& settings, const bool debug);

torch::Tensor markVisible(torch::Tensor& means3D, torch::Tensor& viewmatrix, torch::Tensor& projmatrix);