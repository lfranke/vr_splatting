//
// Created by Linus on 26/08/2024.
//
#pragma once

#include "saiga/core/math/Types.h"
using namespace Saiga;
#include "config.h"

#include <torch/torch.h>
#include <vector>

#include "cie_lab_color.h"

class CombineGSNPSettings : public torch::CustomClassHolder
{
   public:
    ivec2 fovea_pos_in_px;
    float fovea_extend;
    bool complex_fovea_merge;
};

torch::autograd::tensor_list combine_fovea_gs_np(torch::Tensor gs_output, torch::Tensor np_output,
                                                 CombineGSNPSettings combine_settings);

torch::Tensor combine_fovea_forward(torch::Tensor gs_output, torch::Tensor np_output,
                                    CombineGSNPSettings combine_settings);

std::tuple<torch::Tensor, torch::Tensor> combine_fovea_backwards(torch::Tensor gs_output, torch::Tensor np_output,
                                                                 CombineGSNPSettings combine_settings,
                                                                 torch::Tensor image_grads);


class _Combine_GS_NP : public torch::autograd::Function<_Combine_GS_NP>
{
   public:
    static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx, torch::Tensor gs_output,
                                                torch::Tensor np_output, torch::IValue settings)
    {
        CombineGSNPSettings* render_settings = settings.toCustomClass<CombineGSNPSettings>().get();

        auto result = combine_fovea_forward(gs_output, np_output, *render_settings);

        // save for backwards:
        ctx->save_for_backward({gs_output, np_output});
        ctx->saved_data["settings"] = settings;

        return {result};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs)
    {
        // return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
        // std::cout << "here" << std::endl;
        // for (auto t : grad_outputs)
        //{
        //    PrintTensorInfo(t);
        //}
        auto grad_out_color = grad_outputs[0];
        auto saved          = ctx->get_saved_variables();
        auto gs_output      = saved[0];
        auto np_output      = saved[1];

        torch::IValue settings              = ctx->saved_data["settings"];
        CombineGSNPSettings* gs_np_settings = settings.toCustomClass<CombineGSNPSettings>().get();

        auto [grad_gs, grad_np] = combine_fovea_backwards(gs_output, np_output, *gs_np_settings, grad_out_color);

        return {grad_gs, grad_np, torch::Tensor()};
        // return {gs_output, np_output, torch::Tensor()};
    }
};