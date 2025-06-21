//
// Created by Linus on 26/08/2024.
//
#include "saiga/core/imgui/imgui.h"
#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "fovea_combine.h"
/// \Boilerplate
TORCH_LIBRARY(combine_gs_np_settings_for_custom_modulefunc, m)
{
    std::cout << "register combine_gs_np_settings" << std::endl;
    m.class_<CombineGSNPSettings>("CombineGSNPSettings").def(torch::init());
}
torch::autograd::tensor_list combine_fovea_gs_np(torch::Tensor gs_output, torch::Tensor np_output,
                                                 CombineGSNPSettings combine_settings)
{
    torch::intrusive_ptr<CombineGSNPSettings> gs_np_comb_data =
        torch::make_intrusive<CombineGSNPSettings>(combine_settings);

    return _Combine_GS_NP::apply(gs_output, np_output, gs_np_comb_data);
}
/// \Boilerplate_end

__device__ float smootherstep_func(const float x)
{
    return 6 * x * x * x * x * x - 15 * x * x * x * x + 10 * x * x * x;
}
__device__ float smootherstep_func_bwd(const float x)
{
    return 30 * x * x * x * x - 60 * x * x * x + 30 * x * x;
}


template <int CHANNELS, typename T>
__global__ void CombineFoveaForward(StaticDeviceTensor<T, 3> gs_tensor, StaticDeviceTensor<T, 3> np_tensor,
                                    StaticDeviceTensor<T, 3> out, ivec2 fovea_pos_in_px, float fovea_extend_f,
                                    StaticDeviceTensor<T, 3> gs_edge_map, StaticDeviceTensor<T, 3> np_edge_map,
                                    bool complex_fovea_merge, bool debug_mask_only, bool debug_fovea_only)
{
    const int fovea_local_gx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int fovea_local_gy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int fovea_extend     = static_cast<int>(ceilf(fovea_extend_f));
    const ivec2 start_fovea_xy = fovea_pos_in_px - ivec2(fovea_extend / 2, fovea_extend / 2);
    const int gx               = start_fovea_xy.x() + fovea_local_gx;
    const int gy               = start_fovea_xy.y() + fovea_local_gy;

    if (fovea_local_gx >= fovea_extend || fovea_local_gy >= fovea_extend) return;



    bool outside_merge_region =
        !((gx > fovea_pos_in_px.x() - fovea_extend / 2) && (gx < fovea_pos_in_px.x() + fovea_extend / 2) &&
          (gy > fovea_pos_in_px.y() - fovea_extend / 2) && (gy < fovea_pos_in_px.y() + fovea_extend / 2));

    ivec2 start_coords_np = ivec2(fovea_pos_in_px.x() - fovea_extend / 2, fovea_pos_in_px.y() - fovea_extend / 2);

    // e [1.41,0]
    float norm_distance_to_center = sqrtf((gy - fovea_pos_in_px.y()) * (gy - fovea_pos_in_px.y()) +
                                          (gx - fovea_pos_in_px.x()) * (gx - fovea_pos_in_px.x())) /
                                    (0.5f * fovea_extend);

    const float outside_fovea_radial_cutoff = 1;
    const float inside_fovea_radial_cutoff  = 0.75;
    outside_merge_region &= (norm_distance_to_center > outside_fovea_radial_cutoff);

// #define ONLY_GS
#ifdef ONLY_GS
    return;
#endif

    // copy result to output
    if (outside_merge_region)
    {
        // for (int i = 0; i < CHANNELS; ++i)
        //{
        //     out(i, gy, gx) = gs_tensor(i, gy, gx);
        // }
// #define DEBUG_COMBINE_MASK
#ifdef DEBUG_COMBINE_MASK
        for (int i = 0; i < CHANNELS; ++i) out(i, gy, gx) = 0.f;
#endif
        return;
    }
    // only threads merging result remain
    for (int i = 0; i < CHANNELS; ++i)
    {
        float col_np = float(np_tensor(i, fovea_local_gy, fovea_local_gx));
        // out(i, gy, gx) = col_np;
        // continue;

        float col_gs     = float(gs_tensor(i, gy, gx));
        float fac_smooth = (norm_distance_to_center - inside_fovea_radial_cutoff) /
                           (outside_fovea_radial_cutoff - inside_fovea_radial_cutoff);
        float np_factor  = 1.f;
        float smooth_fac = 0.f;
        if (fac_smooth > 0) smooth_fac = smootherstep_func(fac_smooth);
        if (complex_fovea_merge)
        {
            float edge_fac = 1.f - float(np_edge_map(0, fovea_local_gy, fovea_local_gx));
            if (fac_smooth > 0) smooth_fac = smootherstep_func(fac_smooth + 0.2 * edge_fac);
        }
        np_factor = clamp((1.f - smooth_fac), 0, 1);

        out(i, gy, gx) = T(col_np * np_factor + col_gs * (1.f - np_factor));
        // if (fovea_local_gx > fovea_extend / 2) out(i, gy, gx) = np_factor;
        if (debug_mask_only)
            out(i, gy, gx) = np_factor;
        else if (debug_fovea_only)
            out(i, gy, gx) = col_np;
            // else if (debug_fovea_only) out(i, gy, gx) = col_np;
// #define ONLY_NP
#ifdef ONLY_NP
        out(i, gy, gx) = T(col_np);
#endif

#ifdef DEBUG_COMBINE_MASK
        out(i, gy, gx) = np_factor;
#endif
#if 0
        if (fovea_local_gx < fovea_extend / 2)
            // out(i, gy, gx) = col_np * clamp(np_factor, 0.f, 1.f) + col_gs * clamp(1.f - np_factor, 0.f, 1.f);
            if (fovea_local_gy < fovea_extend / 2)
                out(i, gy, gx) = np_edge_map(0, fovea_local_gy, fovea_local_gx);
            else
                out(i, gy, gx) = gs_edge_map(0, gy, gx);
        else
        {
            // out(i, gy, gx) = (1.f - smooth_fac * (1.f - np_edge_map(0, fovea_local_gy, fovea_local_gx)));
            if (fovea_local_gy < fovea_extend / 2)
            {
                float diff     = gs_edge_map(0, gy, gx) - np_edge_map(0, fovea_local_gy, fovea_local_gx);
                out(i, gy, gx) = diff * diff;
            }
            else
                out(i, gy, gx) = abs(gs_edge_map(0, gy, gx) - np_edge_map(0, fovea_local_gy, fovea_local_gx));
            // out(i, gy, gx) = np_edge_map(0, fovea_local_gy, fovea_local_gx);

            //
        }
#endif
    }
}

template <int BLOCKSIZE, int CHANNELS, typename T>
__global__ void EdgeDetectTensor(StaticDeviceTensor<T, 3> in_tensor, StaticDeviceTensor<T, 3> out_tensor)
{
    constexpr int BOARDER         = 2;
    constexpr int MAX_ELEMENTS    = (BLOCKSIZE + 2 * BOARDER) * (BLOCKSIZE + 2 * BOARDER);
    const int gx                  = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy                  = blockIdx.y * blockDim.y + threadIdx.y;
    const int blockstart_gx       = blockIdx.x * blockDim.x;
    const int blockstart_gy       = blockIdx.y * blockDim.y;
    const int thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ float shared_grey_arr[MAX_ELEMENTS];
    auto lin_to_twodim = [&](int linearized)
    {
        return ivec2((linearized % (BLOCKSIZE + 2 * BOARDER)) - BOARDER,
                     (linearized / (BLOCKSIZE + 2 * BOARDER)) - BOARDER);
    };
    auto twodim_to_lin = [&](ivec2 xy) { return (xy.y() + BOARDER) * (BLOCKSIZE + 2 * BOARDER) + xy.x() + BOARDER; };


    for (int running_load = thread_idx_in_block; running_load < MAX_ELEMENTS; running_load += blockDim.x * blockDim.y)
    {
        ivec2 loadidx = lin_to_twodim(running_load) + ivec2(blockstart_gx, blockstart_gy);
        if (loadidx.x() >= 0 && loadidx.x() < in_tensor.size(2) && loadidx.y() >= 0 && loadidx.y() < in_tensor.size(1))
        {
            float elem = 0.f;
            for (int i = 0; i < CHANNELS; ++i) elem += float(in_tensor(i, loadidx.y(), loadidx.x()));
            shared_grey_arr[running_load] = elem / float(CHANNELS);
        }
        else
        {
            shared_grey_arr[running_load] = 0.f;
        }
    }
    __syncthreads();

    float* sobel_input = &shared_grey_arr[0];
#if 1
    // gauss
    const int GAUSS_RANGE = 1;

    const float gauss_kernel[(2 * GAUSS_RANGE + 1) * (2 * GAUSS_RANGE + 1)] = {
        1.f / 16.f, 1 / 8.f, 1.f / 16.f, 1.f / 8.f, 1.f / 4.f, 1.f / 8.f, 1.f / 16.f, 1.f / 8.f, 1.f / 16.f,
    };
    __shared__ float shared_blurred_arr[MAX_ELEMENTS];

    for (int running_load = thread_idx_in_block; running_load < MAX_ELEMENTS; running_load += blockDim.x * blockDim.y)
    {
        ivec2 loadidx = lin_to_twodim(running_load);
        // leave 1 px boarder
        if (loadidx.x() >= (-BOARDER + 1) && loadidx.x() < BLOCKSIZE + BOARDER - 1 && loadidx.y() >= (-BOARDER + 1) &&
            loadidx.y() < BLOCKSIZE + BOARDER - 1)
        {
            float blurred = 0.f;

            for (int y = -GAUSS_RANGE; y <= GAUSS_RANGE; ++y)
            {
                for (int x = -GAUSS_RANGE; x <= GAUSS_RANGE; ++x)
                {
                    const float filter_weight =
                        gauss_kernel[(y + GAUSS_RANGE) * (2 * GAUSS_RANGE + 1) + (x + GAUSS_RANGE)];
                    blurred += filter_weight * shared_grey_arr[twodim_to_lin(ivec2(loadidx.x() + x, loadidx.y() + y))];
                }
            }
            shared_blurred_arr[twodim_to_lin(ivec2(loadidx.x(), loadidx.y()))] = blurred;
        }
        else
        {
            shared_blurred_arr[twodim_to_lin(ivec2(loadidx.x(), loadidx.y()))] = 0.f;
        }
    }
    __syncthreads();

    sobel_input = &shared_blurred_arr[0];

#endif

    constexpr int SOBEL_RANGE = 1;

    const float sobel_x[(2 * SOBEL_RANGE + 1) * (2 * SOBEL_RANGE + 1)] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    const float sobel_y[(2 * SOBEL_RANGE + 1) * (2 * SOBEL_RANGE + 1)] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    // static const float sobel_x[(2 * SOBEL_RANGE + 1) * (2 * SOBEL_RANGE + 1)] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    // static const float sobel_y[(2 * SOBEL_RANGE + 1) * (2 * SOBEL_RANGE + 1)] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    float G_x = 0.f;
    float G_y = 0.f;
    for (int y = -SOBEL_RANGE; y <= SOBEL_RANGE; ++y)
    {
        for (int x = -SOBEL_RANGE; x <= SOBEL_RANGE; ++x)
        {
            const float val = sobel_input[twodim_to_lin(ivec2(threadIdx.x + x, threadIdx.y + y))];
            G_x += sobel_x[(y + SOBEL_RANGE) * (2 * SOBEL_RANGE + 1) + (x + SOBEL_RANGE)] * val;
            G_y += sobel_y[(y + SOBEL_RANGE) * (2 * SOBEL_RANGE + 1) + (x + SOBEL_RANGE)] * val;
        }
    }
    out_tensor(0, gy, gx) = T(sqrtf(G_x * G_x + G_y * G_y));
    // out_tensor(0, gy, gx) = shared_grey_arr[twodim_to_lin(ivec2(threadIdx.x, threadIdx.y))];
}


torch::Tensor combine_fovea_forward(torch::Tensor gs_output, torch::Tensor np_output,
                                    CombineGSNPSettings combine_settings)
{
    auto edge_map_gs = torch::empty({1, gs_output.size(1), gs_output.size(2)}, gs_output.options());
    auto edge_map_np = torch::empty({1, np_output.size(1), np_output.size(2)}, np_output.options());
    if (combine_settings.complex_fovea_merge)
    {
        const int bx = iDivUp(static_cast<int>(gs_output.size(2)), 16);
        const int by = iDivUp(static_cast<int>(gs_output.size(1)), 16);
        if (gs_output.dtype() == torch::kFloat)
            ::EdgeDetectTensor<16, 3, float><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(gs_output, edge_map_gs);
        else
            ::EdgeDetectTensor<16, 3, __half><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(gs_output, edge_map_gs);

        const int bx_np = iDivUp(static_cast<int>(np_output.size(2)), 16);
        const int by_np = iDivUp(static_cast<int>(np_output.size(1)), 16);
        if (gs_output.dtype() == torch::kFloat)
            ::EdgeDetectTensor<16, 3, float><<<dim3(bx_np, by_np, 1), dim3(16, 16, 1)>>>(np_output, edge_map_np);
        else
            ::EdgeDetectTensor<16, 3, __half><<<dim3(bx_np, by_np, 1), dim3(16, 16, 1)>>>(np_output, edge_map_np);

        CUDA_SYNC_CHECK_ERROR();
    }
    torch::Tensor result = gs_output.clone().detach();

    const int bx = iDivUp(static_cast<int>(std::ceil(combine_settings.fovea_extend)), 16);
    const int by = iDivUp(static_cast<int>(std::ceil(combine_settings.fovea_extend)), 16);
    SAIGA_ASSERT(bx > 0 && by > 0);


    static bool debug_mask_only     = false;
    static bool debug_fovea_only    = false;
    static bool debug_gaussian_only = false;

    /// ImGui::Checkbox("debug_fovea_only", &debug_fovea_only);
    /// ImGui::Checkbox("debug_mask_only", &debug_mask_only);
    /// ImGui::Checkbox("debug_gaussian_only", &debug_gaussian_only);
    if (debug_mask_only || debug_fovea_only) result.zero_();

    // PrintTensorInfo(gs_output);
    // PrintTensorInfo(np_output);
    // PrintTensorInfo(result);
    if (gs_output.dtype() == torch::kFloat)
        ::CombineFoveaForward<3, float><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
            gs_output, np_output, result, combine_settings.fovea_pos_in_px, combine_settings.fovea_extend, edge_map_gs,
            edge_map_np, combine_settings.complex_fovea_merge, debug_mask_only, debug_fovea_only);
    else
        ::CombineFoveaForward<3, __half><<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(
            gs_output, np_output, result, combine_settings.fovea_pos_in_px, combine_settings.fovea_extend, edge_map_gs,
            edge_map_np, combine_settings.complex_fovea_merge, debug_mask_only, debug_fovea_only);


    if (debug_gaussian_only) result = gs_output.clone().detach();
    CUDA_SYNC_CHECK_ERROR();
    return result;
}


template <int CHANNELS>
__global__ void CombineFoveaBackwards(StaticDeviceTensor<float, 3> gs_tensor, StaticDeviceTensor<float, 3> np_tensor,
                                      ivec2 fovea_pos_in_px, float fovea_extend_f, StaticDeviceTensor<float, 3> grad_in,
                                      StaticDeviceTensor<float, 3> grad_out_gs,
                                      StaticDeviceTensor<float, 3> grad_out_np)
{
    const int fovea_local_gx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int fovea_local_gy   = blockIdx.y * blockDim.y + threadIdx.y;
    const int fovea_extend     = static_cast<int>(ceilf(fovea_extend_f));
    const ivec2 start_fovea_xy = fovea_pos_in_px - ivec2(fovea_extend / 2, fovea_extend / 2);
    const int gx               = start_fovea_xy.x() + fovea_local_gx;
    const int gy               = start_fovea_xy.y() + fovea_local_gy;

    if (fovea_local_gx >= fovea_extend || fovea_local_gy >= fovea_extend) return;

    bool outside_merge_region =
        !((gx > fovea_pos_in_px.x() - fovea_extend / 2) && (gx < fovea_pos_in_px.x() + fovea_extend / 2) &&
          (gy > fovea_pos_in_px.y() - fovea_extend / 2) && (gy < fovea_pos_in_px.y() + fovea_extend / 2));

    ivec2 start_coords_np = ivec2(fovea_pos_in_px.x() - fovea_extend / 2, fovea_pos_in_px.y() - fovea_extend / 2);

    // e [1.41,0]
    float norm_distance_to_center = sqrtf((gy - fovea_pos_in_px.y()) * (gy - fovea_pos_in_px.y()) +
                                          (gx - fovea_pos_in_px.x()) * (gx - fovea_pos_in_px.x())) /
                                    (0.5f * fovea_extend);

    const float outside_fovea_radial_cutoff = 0.95;
    const float inside_fovea_radial_cutoff  = 0.5;
    outside_merge_region &= (norm_distance_to_center > outside_fovea_radial_cutoff);

    // copy result to output
    if (outside_merge_region)
    {
        return;
    }
    // only threads merging result remain
    for (int i = 0; i < CHANNELS; ++i)
    {
        // float col_np     = np_tensor(i, fovea_local_gy, fovea_local_gx);
        // float col_gs     = gs_tensor(i, gy, gx);
        float fac_smooth = (norm_distance_to_center - inside_fovea_radial_cutoff) /
                           (outside_fovea_radial_cutoff - inside_fovea_radial_cutoff);
        float np_factor  = 1.f;
        float smooth_fac = 0.f;
        if (fac_smooth > 0) smooth_fac = smootherstep_func(fac_smooth);
        // float edge_fac = np_edge_map(0, fovea_local_gy, fovea_local_gx);
        // if (fac_smooth > 0) smooth_fac = smootherstep_func(fac_smooth + 0.2 * edge_fac);

        np_factor = clamp((1.f - smooth_fac), 0, 1);

        // forward
        // out(i, gy, gx) = col_np * np_factor + col_gs * (1.f - np_factor);

        float grad                                     = grad_in(i, gy, gx);
        grad_out_gs(i, gy, gx)                         = (1.f - np_factor) * grad;
        grad_out_np(i, fovea_local_gy, fovea_local_gx) = np_factor * grad;
    }
#if 0

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= grad_in.size(2) || gy >= grad_in.size(1)) return;

    bool outside_merge_region =
        !((gx > fovea_pos_in_px.x() - fovea_extend / 2) && (gx < fovea_pos_in_px.x() + fovea_extend / 2) &&
          (gy > fovea_pos_in_px.y() - fovea_extend / 2) && (gy < fovea_pos_in_px.y() + fovea_extend / 2));

    // copy result to output
    if (outside_merge_region)
    {
        for (int i = 0; i < CHANNELS; ++i)
        {
            grad_out_gs(i, gy, gx) = grad_in(i, gy, gx);
        }
        return;
    }
    // only threads merging result remain

    ivec2 start_coords_np = ivec2(fovea_pos_in_px.x() - fovea_extend / 2, fovea_pos_in_px.y() - fovea_extend / 2);

    for (int i = 0; i < CHANNELS; ++i)
    {
        grad_out_np(i, gy - start_coords_np.y(), gx - start_coords_np.x()) = grad_in(i, gy, gx);
    }
#endif
}

std::tuple<torch::Tensor, torch::Tensor> combine_fovea_backwards(torch::Tensor gs_output, torch::Tensor np_output,
                                                                 CombineGSNPSettings combine_settings,
                                                                 torch::Tensor image_grads)
{
    torch::Tensor grad_gs_out = image_grads.clone().detach();
    torch::Tensor grad_np_out = torch::zeros_like(np_output);

    const int bx = iDivUp(static_cast<int>(std::ceil(combine_settings.fovea_extend)), 16);
    const int by = iDivUp(static_cast<int>(std::ceil(combine_settings.fovea_extend)), 16);
    SAIGA_ASSERT(bx > 0 && by > 0);

    ::CombineFoveaBackwards<3>
        <<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(gs_output, np_output, combine_settings.fovea_pos_in_px,
                                               combine_settings.fovea_extend, image_grads, grad_gs_out, grad_np_out);

    CUDA_SYNC_CHECK_ERROR();


    return std::make_tuple(grad_gs_out, grad_np_out);
}