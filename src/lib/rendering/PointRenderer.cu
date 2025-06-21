/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// #undef CUDA_DEBUG
// #define CUDA_NDEBUG

// #include "saiga/colorize.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/torch/CudaHelper.h"

#include "PointRenderer.h"
#include "PointRendererHelper.h"
#include "RenderConstants.h"

#include "cooperative_groups.h"
#include <curand_kernel.h>

// curandState* curand_state_h;



__global__ void ProjectDirectionToWSKernel(StaticDeviceTensor<float, 3> in_directions,
                                           StaticDeviceTensor<float, 3> out_directions, Sophus::SE3d* pose)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= in_directions.size(2) || gy >= in_directions.size(1)) return;

    vec3 dir_vs = vec3(in_directions(0, gy, gx), in_directions(1, gy, gx), in_directions(2, gy, gx));
    //  CUDA_KERNEL_ASSERT(dir_vs.z() != 0);
    Sophus::SE3d inv_pose = pose[0].inverse();

    vec3 point_ws = inv_pose.cast<float>() * dir_vs.normalized();

    vec3 dir_ws = point_ws - inv_pose.cast<float>() * vec3(0, 0, 0);
    dir_ws /= length(dir_ws);

    out_directions(0, gy, gx) = dir_ws.x();
    out_directions(1, gy, gx) = dir_ws.y();
    out_directions(2, gy, gx) = dir_ws.z();
}

torch::Tensor ProjectDirectionsToWS(torch::Tensor directions, torch::Tensor pose)
{
    // Sophus::SE3d inv_pose = pose.inverse();
    torch::Tensor result = torch::empty_like(directions);
    // std::cout << "----" << TensorInfo(directions) << std::endl;
    // std::cout << "----" << TensorInfo(result) << std::endl;

    int bx = iDivUp(directions.size(2), 16);
    int by = iDivUp(directions.size(1), 16);
    SAIGA_ASSERT(bx > 0 && by > 0);

    //  std::cout << "----" << bx << " " << by << " " << directions.size(2) << " " << directions.size(1) << " "
    //            << std::endl;
    ::ProjectDirectionToWSKernel<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(directions, result,
                                                                       (Sophus::SE3d*)pose.data_ptr<double>());
    CUDA_SYNC_CHECK_ERROR();
    // std::cout << "----" << TensorInfo(result) << std::endl;

    return result;
}

__global__ void PaintEyePosOnTensorKernel(StaticDeviceTensor<float, 3> t, vec2 pos, float radius)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    float disx = gx - pos.x();
    float disy = gy - pos.y();

    float d = sqrtf(disx*disx + disy*disy);

    if(d<radius){
        t(0,gy,gx)=1;
        t(1,gy,gx)=0;
        t(2,gy,gx)=0;
    }
}

void PaintEyePosOnTensor(torch::Tensor t, vec2 eye_pos, float radius){

    int bx = iDivUp(t.size(2), 16);
    int by = iDivUp(t.size(1), 16);
    SAIGA_ASSERT(bx > 0 && by > 0);

    ::PaintEyePosOnTensorKernel<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(t, eye_pos, radius);
    CUDA_SYNC_CHECK_ERROR();
}


__global__ void DebugWeightToColor(ImageView<float> weight, StaticDeviceTensor<float, 3> out_neural_image,
                                   float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= weight.width || gy >= weight.height) return;

    auto cou = weight(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        // float t = ::saturate(x);
        // vec3 c  = saturate(vec3(sqrt(t), t * t * t, std::max(sin(3.1415 * 1.75 * t), pow(t, 12.0))));

        vec3 c = colorizeTurbo(x);

        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}

__global__ void DebugDepthToColor(ImageView<float> depth, StaticDeviceTensor<float, 3> out_neural_image,
                                  float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= depth.width || gy >= depth.height) return;

    auto cou = depth(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);
        // divide by weight
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void DebugCountingsToColor(ImageView<int> counting, StaticDeviceTensor<float, 3> out_neural_image,
                                      float debug_max_weight)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= counting.width || gy >= counting.height) return;

    auto cou = counting(gy, gx);
    CUDA_DEBUG_ASSERT(out_neural_image.sizes[0] == 4);

    if (cou == 0)
    {
        // copy background into output
        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = 0;
        }
        out_neural_image(3, gy, gx) = 1;
    }
    else
    {
        float x = cou / debug_max_weight;
        vec3 c  = vec3(x, x, x);

        for (int ci = 0; ci < 3; ++ci)
        {
            out_neural_image(ci, gy, gx) = c(ci);
        }
        out_neural_image(3, gy, gx) = 1;
    }
}


__global__ void CreateMask(StaticDeviceTensor<float, 4> in_weight, StaticDeviceTensor<float, 4> out_mask,
                           float background_value, int b)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;


    if (!in_weight.Image().template inImage(gy, gx)) return;

    auto w = in_weight.At({b, 0, gy, gx});

    if (w == 0)
    {
        out_mask.At({b, 0, gy, gx}) = background_value;
    }
    else
    {
        out_mask(b, 0, gy, gx) = 1;
    }
}


void PointRendererCache::Build(NeuralRenderInfo* info, bool forward)
{
    this->info        = info;
    this->num_batches = info->images.size();

    SAIGA_OPTIONAL_TIME_MEASURE("Build Cache", info->timer_system);
    static_assert(sizeof(Packtype) == 8);

    SAIGA_ASSERT(num_batches > 0);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Allocate", info->timer_system);
        Allocate(info, forward);
    }

    if (forward)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Initialize", info->timer_system);
        InitializeData(forward);
    }
    else
    {
        output_gradient_texture    = torch::zeros_like(info->scene->texture->texture);
        output_gradient_confidence = torch::zeros_like(info->scene->texture->confidence_value_of_point);


        output_gradient_background = torch::zeros_like(info->scene->texture->background_color);

        if (info->scene->point_cloud_cuda->t_point_size.requires_grad())
            output_gradient_layer = torch::zeros_like(info->scene->point_cloud_cuda->t_point_size);

        if (info->scene->point_cloud_cuda->t_position.requires_grad())
        {
            output_gradient_points = torch::zeros_like(info->scene->point_cloud_cuda->t_position);
            // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_point_count =
                    torch::zeros({output_gradient_points.size(0)}, output_gradient_points.options());
        }
#if 0
       if (info->scene->dynamic_refinement_t.sizes().size() > 1)
       {
           output_gradient_dynamic_points = torch::zeros_like(info->scene->dynamic_refinement_t);
           // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
           if (info->params.normalize_grads)

               output_gradient_dynamic_point_count =
                   torch::zeros({output_gradient_dynamic_points.size(0), output_gradient_dynamic_points.size(1)},
                                output_gradient_points.options());
       }
#endif
        if (info->scene->poses->tangent_poses.requires_grad())
        {
            output_gradient_pose_tangent = torch::zeros_like(info->scene->poses->tangent_poses);
            //            if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_pose_tangent_count =
                    torch::zeros({info->scene->poses->tangent_poses.size(0)},
                                 info->scene->poses->tangent_poses.options().dtype(torch::kFloat32));
        }

        if (info->scene->intrinsics->is_training())
        {
            output_gradient_intrinsics = torch::zeros_like(info->scene->intrinsics->intrinsics);
            // if (render_mode != PointRendererCache::RenderMode::TILED_BILINEAR_BLEND)
            if (info->params.normalize_grads)
                output_gradient_intrinsics_count = torch::zeros({info->scene->intrinsics->intrinsics.size(0)},
                                                                info->scene->intrinsics->intrinsics.options());
        }
    }
}

void PointRendererCache::Allocate(NeuralRenderInfo* info, bool forward)
{
    auto& fd = info->images.front();
    int h    = fd.h;
    int w    = fd.w;

    SAIGA_ASSERT(info->scene->point_cloud_cuda);
    SAIGA_ASSERT(info->scene->texture);

    std::vector<int> new_cache_size = {(int)info->scene->texture->texture.size(0),
                                       info->scene->point_cloud_cuda->Size(),
                                       info->num_layers,
                                       num_batches,
                                       h,
                                       w,
                                       info->params.viewer_only};


    bool size_changed = new_cache_size != cache_size;
    // size_changed      = true;
    if (size_changed)
    {
        cache_has_forward  = false;
        cache_has_backward = false;
    }

    bool need_allocate_forward  = !cache_has_forward && forward;
    bool need_allocate_backward = !cache_has_backward && !forward;

    if (!need_allocate_forward && !need_allocate_backward)
    {
        // std::cout << "skip allocate" << std::endl;
        return;
    }

    // std::cout << "allocate render cache " << need_allocate_forward << " " << need_allocate_backward << " "
    //          << size_changed << std::endl;

    /*    if (curand_state_h == nullptr)
        {
            cudaMalloc(&curand_state_h, sizeof(curandState));
            Saiga::CUDA::initRandom(ArrayView<curandState>(curand_state_h, 1), 0);
        }*/
    if (size_changed)
    {
        layers_cuda.resize(info->num_layers);
    }

    const int MAX_ELEM_TILE_EX   = 8192;
    const int TILE_SIZE          = 16;
    int link_list_tile_extension = w / TILE_SIZE * h / TILE_SIZE * MAX_ELEM_TILE_EX;
    float scale                  = 1;
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(w > 0 && h > 0);
        auto& l = layers_cuda[i];

        if (need_allocate_forward || need_allocate_backward)
        {
            {
                int size_of_save_buffer = 6;
                size_of_save_buffer     = 7;
                l.bw_sorted_maxed       = torch::empty({num_batches, h, w, max_pixels_per_list, size_of_save_buffer},
                                                       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));

                //   l.per_pixel_list_heads =
                //       torch::empty({num_batches, h, w},
                //       torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                l.per_pixel_list_lengths =
                    torch::empty({num_batches, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                l.scanned_counts =
                    torch::empty({num_batches, h, w}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

                // l.per_pixel_list_links =
                //     torch::empty({num_batches, (info->scene->point_cloud_cuda->Size() + link_list_tile_extension) *
                //     4},
                //                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
            }
        }



        l.size  = {w, h};
        l.scale = scale;

        if (info->scene->params->net_params.network_version != "MultiScaleUnet2d")
        {
            h = std::ceil(float(h) / 2.f);
            w = std::ceil(float(w) / 2.f);
        }
        else
        {
            h = h / 2;
            w = w / 2;
        }

        scale *= 0.5;
    }



    if (need_allocate_forward && info->train && info->params.dropout > 0)
    {
        dropout_points = torch::empty({num_batches, info->scene->point_cloud_cuda->Size()},
                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    }

    cache_size = new_cache_size;
    if (forward)
    {
        cache_has_forward = true;
    }
    else
    {
        cache_has_backward = true;
    }
}

void PointRendererCache::InitializeData(bool forward)
{
    if (forward)
    {
        for (auto& l : layers_cuda)
        {
            {
                //   l.per_pixel_list_heads.fill_(-1);
                if (info->train)
                {
                    l.bw_sorted_maxed.fill_(-1.f);
                }
                l.per_pixel_list_lengths.zero_();
            }

            // l.per_pixel_list_links.fill_(-1);

            // l.per_pixel_sorted_bilin_lists.fill_(std::numeric_limits<double>::max());
            // PrintTensorInfo(l.per_pixel_sorted_bilin_lists);
        }


        // This is created every frame, because we 'move' it to the output
        output_forward.resize(info->num_layers);
        for (int i = 0; i < info->num_layers; ++i)
        {
            int w                = layers_cuda[i].size(0);
            int h                = layers_cuda[i].size(1);
            int texture_channels = info->params.num_texture_channels;
            if (info->train)
            {
                output_forward[i] = torch::ones({num_batches, texture_channels, h, w},
                                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                for (int bg = 0; bg < info->scene->texture->background_color.size(0); ++bg)
                    output_forward[i].slice(1, bg, bg + 1) *=
                        info->scene->texture->background_color.slice(0, bg, bg + 1);
            }
            else
            {
                // output_forward[i] = torch::ones({num_batches, h, w, texture_channels},
                //                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                // for (int bg = 0; bg < info->scene->texture->background_color.size(0); ++bg)
                //     output_forward[i].slice(3, bg, bg + 1) *=
                //         info->scene->texture->background_color.slice(0, bg, bg + 1);

                output_forward[i] = info->scene->texture->background_color.repeat({num_batches, h, w, 1}).contiguous();
            }

            // else
            //     output_forward[i] = torch::zeros({num_batches, h, w, texture_channels},
            //                                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32))
            //                             .permute({0, 3, 1, 2});
        }

        if (info->params.add_depth_to_network)
        {
            output_forward_depthbuffer.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                int w                = layers_cuda[i].size(0);
                int h                = layers_cuda[i].size(1);
                int texture_channels = 1;
                output_forward_depthbuffer[i] =
                    torch::zeros({num_batches, texture_channels, h, w},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

        if (info->params.output_background_mask)
        {
            output_forward_blend.resize(info->num_layers);
            for (int i = 0; i < info->num_layers; ++i)
            {
                auto& l = layers_cuda[i];
                output_forward_blend[i] =
                    torch::zeros({num_batches, 1, l.size.y(), l.size.x()},
                                 torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            }
        }

#if 0
       for (auto& t : output_forward)
       {
           t.zero_();
       }
#endif
        if (info->train)
        {
            if (info->params.dropout > 0)
            {
                dropout_points.bernoulli_(info->params.dropout);
            }
            else
            {
                //   dropout_points.zero_();
            }
        }
    }
}


DeviceRenderParams PointRendererCache::PrepareDeviceRenderParams()
{
    static DeviceRenderParams drp;

    drp = DeviceRenderParams(info->params);
    if (info->scene)
    {
        drp._poses     = (Sophus::SE3d*)info->scene->poses->poses_se3.data_ptr<double>();
        drp.intrinsics = info->scene->intrinsics->intrinsics;
    }
    drp.num_layers = info->num_layers;


    if (info->params.use_point_adding_and_removing_module)
    {
        if (gradient_of_forward_pass_x.defined())
        {
            drp.gradient_of_forward_pass_x = gradient_of_forward_pass_x;
        }
    }

    // drp.curand_state  = curand_state_h;
    drp.current_epoch = info->current_epoch;

    return drp;
}
DeviceTexture PointRendererCache::PrepareDeviceTexture()
{
    static DeviceTexture d_tex;

    // if (render_mode == RenderMode::TILED_BILINEAR_BLEND)
    //{
    //     d_tex.points_confidence_value = info->scene->texture->confidence_value_of_point.permute({1,
    //     0}).contiguous(); d_tex.in_texture              = info->scene->texture->texture.permute({1,
    //     0}).contiguous();
    // }
    // else
    //{
    d_tex.points_confidence_value = info->scene->texture->confidence_value_of_point;
    // d_tex.points_layer_value      = info->scene->texture->layer_value_of_point;

    d_tex.in_texture = info->scene->texture->texture;
    //}
    // std::cout << TensorInfo(info->scene->texture->texture.contiguous()) << std::endl;

    return d_tex;
}


DeviceBackwardParams PointRendererCache::PrepareDeviceBackwardParams()
{
    DeviceBackwardParams dbp = {0};

    dbp.out_gradient_pose       = nullptr;
    dbp.out_gradient_pose_count = nullptr;
    if (output_gradient_pose_tangent.defined())
    {
        SAIGA_ASSERT(output_gradient_pose_tangent.size(1) == 6);
        dbp.out_gradient_pose = (Vec6*)output_gradient_pose_tangent.data_ptr<double>();
        if (info->params.normalize_grads)
            dbp.out_gradient_pose_count = output_gradient_pose_tangent_count.data_ptr<float>();
    }

    dbp.out_gradient_points       = nullptr;
    dbp.out_gradient_points_count = nullptr;
    if (output_gradient_points.defined())
    {
        SAIGA_ASSERT(output_gradient_points.size(1) == 4);
        dbp.out_gradient_points = (vec4*)output_gradient_points.data_ptr<float>();
        if (info->params.normalize_grads) dbp.out_gradient_points_count = output_gradient_point_count.data_ptr<float>();
    }


    dbp.out_gradient_intrinsics.data  = nullptr;
    dbp.out_gradient_intrinsics_count = nullptr;
    if (output_gradient_intrinsics.defined())
    {
        dbp.out_gradient_intrinsics = output_gradient_intrinsics;
        if (info->params.normalize_grads)
            dbp.out_gradient_intrinsics_count = output_gradient_intrinsics_count.data_ptr<float>();
    }

    if (output_gradient_layer.defined())
    {
        dbp.out_gradient_layer = output_gradient_layer;
    }
    else
    {
        dbp.out_gradient_layer.data = nullptr;
    }
    dbp.out_gradient_texture    = output_gradient_texture;
    dbp.out_gradient_confidence = output_gradient_confidence;

    SAIGA_ASSERT(image_gradients.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        SAIGA_ASSERT(image_gradients[i].dim() == 4);
        dbp.in_gradient_image[i] = image_gradients[i];
    }

    return dbp;
}


void PointRendererCache::CreateMask(int batch, float background_value)
{
    SAIGA_ASSERT(output_forward_blend.size() == info->num_layers);
    for (int i = 0; i < info->num_layers; ++i)
    {
        // Allocate result tensor
        auto& l = layers_cuda[i];
        int bx  = iDivUp(l.size.x(), 16);
        int by  = iDivUp(l.size.y(), 16);
        SAIGA_ASSERT(bx > 0 && by > 0);

        SAIGA_ASSERT(output_forward_blend[i].size(2) == l.size.y());
        ::CreateMask<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(l.weight, output_forward_blend[i], background_value, batch);
    }
    CUDA_SYNC_CHECK_ERROR();
}



std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BlendPointCloudForward(
    torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info)
{
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    PointRendererCache& cache = *info->cache;

    cache.Build(info, true);

    int num_batches = cache.num_batches;

    cache.PushParametersForward();

    // only for blending
    //  tensors are shape [2,max_elems]
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);

    {
        static bool new_impl = true;

        if (new_impl && info->params.use_layer_point_size && info->params.render_points_in_all_lower_resolutions &&
            !info->params.combine_lists)
        {
            cache.UploadLinkedListBuffers();

            for (int b = 0; b < num_batches; ++b)
            {
                cache.RenderFast16(b, scene.point_cloud_cuda, info->train, scene.texture->background_color,
                                   info->timer_system);
            }

            if (!info->train)
            {
                for (int i = 0; i < info->num_layers; ++i)
                {
                    cache.output_forward[i] = cache.output_forward[i].permute({0, 3, 1, 2});
                }
            }
        }
        else
        {
            SAIGA_ASSERT(false);
        }
    }

    if (info->params.debug_weight_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto weights = l.BatchViewWeights(b);
                ::DebugWeightToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(weights, in_out_neural_image,
                                                                           info->params.debug_max_weight);
            }
        }
    }

    if (info->params.debug_depth_color && info->params.num_texture_channels == 4)
    {
        for (int b = 0; b < num_batches; ++b)
        {
            for (int i = 0; i < info->num_layers; ++i)
            {
                // Allocate result tensor
                auto& l = cache.layers_cuda[i];
                int bx  = iDivUp(l.size.x(), 16);
                int by  = iDivUp(l.size.y(), 16);
                SAIGA_ASSERT(bx > 0 && by > 0);
                auto in_out_neural_image = cache.output_forward[i][b];

                auto depths = l.BatchViewDepth(b);
                ::DebugDepthToColor<<<dim3(bx, by, 1), dim3(16, 16, 1)>>>(depths, in_out_neural_image,
                                                                          info->params.debug_max_weight);
            }
        }
    }


    if (info->params.debug_print_num_rendered_points)
    {
        double weight_sum = 0;
        for (int i = 0; i < info->num_layers; ++i)
        {
            // Allocate result tensor
            auto& l = cache.layers_cuda[i];
            weight_sum += l.weight.sum().item().toFloat();
        }
        std::cout << "# Rasterized Points = " << (int)weight_sum << std::endl;
    }

    if (ctx && !info->params.viewer_only)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Save in Graph", info->timer_system);
        std::vector<torch::Tensor> save_variables;

        {
            for (auto l : cache.layers_cuda)
            {
                save_variables.push_back(l.bw_sorted_maxed);
            }
            save_variables.insert(save_variables.end(), cache.output_forward.begin(), cache.output_forward.end());
        }

        save_variables.push_back(cache.dropout_points);
        ctx->save_for_backward(save_variables);
        CUDA_SYNC_CHECK_ERROR();
    }

    if (info->params.add_depth_to_network)
    {
        cache.output_forward.insert(cache.output_forward.end(), cache.output_forward_depthbuffer.begin(),
                                    cache.output_forward_depthbuffer.end());
    }

    // cudaDeviceSynchronize();
    return {std::move(cache.output_forward), std::move(cache.output_forward_blend)};
}

template <typename T, int N>
__global__ void NormalizeGradient(Vector<T, N>* tangent, float* tangent_count, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vector<T, N> t = tangent[tid];
    float c        = tangent_count[tid];

    if (c > 0)
    {
        // if (N == 6)
        //     for (int i = 0; i < 6; ++i) printf("++%f++ ", float(t(i)));
        tangent[tid] = t / c;
        // tangent[tid] = t / T(c);
        // if (N == 6)
        //    for (int i = 0; i < 6; ++i) printf("##%f## ", float(tangent[tid](i)));
    }
}

template <typename T, int N>
__global__ void NormalizeGradientDevTensor(StaticDeviceTensor<T, 2> tangent, StaticDeviceTensor<float, 1> tangent_count,
                                           int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Vector<T, N> t = tangent[tid];
    float c = tangent_count(tid);

    if (c > 0)
    {
        for (int i = 0; i < N; ++i)
        {
            tangent(tid, i) = tangent(tid, i) / c;
        }
    }
}

torch::autograd::variable_list BlendPointCloudBackward(torch::autograd::AutogradContext* ctx, NeuralRenderInfo* info,
                                                       torch::autograd::variable_list _image_gradients)
{
    SAIGA_ASSERT(info->cache);
    for (auto& ig : _image_gradients)
    {
        SAIGA_ASSERT(ig.dtype() == torch::kFloat32);
    }

    // int num_batches     = info->images.size();
    NeuralScene& scene  = *info->scene;
    RenderParams params = info->params;

    // PointRendererCache cache;
    PointRendererCache& cache = *info->cache;

    int num_batches = cache.num_batches;

    /*
     *  These buffers are large buffers including space for exactly all pixels collected.
     *  Accessing can be done with the scanned countings list.
     *  there exists one for each batch and layer (i.e. 4 batches, 4 layers = [4][4])
     *  gradient_sum_back_buffer is an intermediate buffer for the Jacobians, with num_tex_parameters + 1 channels
     */
    std::vector<std::vector<torch::Tensor>> collection_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> per_point_data_buffer(num_batches);
    std::vector<std::vector<torch::Tensor>> gradient_sum_back_buffer(num_batches);


    {
        SAIGA_OPTIONAL_TIME_MEASURE("Prepare Backward", info->timer_system);
        cache.Build(info, false);

        // The first [num_layers] gradients are the actual neural image gradients. After that we get the gradients
        // of the mask which does not help us much
        cache.image_gradients =
            std::vector<torch::Tensor>(_image_gradients.begin(), _image_gradients.begin() + info->num_layers);

        auto save_variables = ctx->get_saved_variables();

        {
            cache.output_forward.resize(info->num_layers);
            int offset_v = 1;
            for (int i = 0; i < info->num_layers; ++i)
            {
                cache.layers_cuda[i].bw_sorted_maxed = save_variables[i * offset_v + 0];
                cache.output_forward[i]              = save_variables[info->num_layers * offset_v + i];
            }
        }
        cache.dropout_points = save_variables.back();

        SAIGA_ASSERT(cache.image_gradients.size() == info->num_layers);

        cache.PushParametersBackward();
    }
    {
        cache.UploadCollectionBuffersBackwardsTiled();
        SAIGA_OPTIONAL_TIME_MEASURE("BlendBackwardsBilinearFast", info->timer_system);
        for (int b = 0; b < num_batches; ++b)
        {
            cache.BlendBackwardsBilinearFast(b, scene.point_cloud_cuda, scene.texture->background_color,
                                             info->params.use_environment_map);
        }
    }


    if (info->params.normalize_grads)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Post Process Gradient", info->timer_system);
        if (cache.output_gradient_pose_tangent.defined())
        {
            // std::cout << "POSE NORMALIZATION" <<
            // TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) <<
            //           std::endl
            //           << std::endl;
            //  Average pose gradient over all measurements
            int n = cache.output_gradient_pose_tangent.size(0);
            int c = iDivUp(n, 128);
            // NormalizeGradient<double, 6><<<c,
            // 128>>>((Vec6*)cache.output_gradient_pose_tangent.data_ptr<double>(),
            //                                           cache.output_gradient_pose_tangent_count.data_ptr<float>(),
            // n);

            NormalizeGradientDevTensor<double, 6>
                <<<c, 128>>>(cache.output_gradient_pose_tangent, cache.output_gradient_pose_tangent_count, n);
            CUDA_SYNC_CHECK_ERROR();

            // std::cout << std::endl
            //           << "END POSE NORMALIZATION" <<
            //           TensorInfo(cache.output_gradient_pose_tangent)
            //           << TensorInfo(cache.output_gradient_pose_tangent_count) <<
            //           std::endl;
        }

        if (cache.output_gradient_points.defined())
        {
            // Average point gradient over all measurements
            int n = cache.output_gradient_points.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 3><<<c, 128>>>((vec3*)cache.output_gradient_points.data_ptr<float>(),
                                                    cache.output_gradient_point_count.data_ptr<float>(), n);
        }

        if (cache.output_gradient_intrinsics.defined())
        {
            // Average intrinsics/distortion gradient over all measurements
            int n = cache.output_gradient_intrinsics.size(0);
            int c = iDivUp(n, 128);
            NormalizeGradient<float, 13>
                <<<c, 128>>>((Vector<float, 13>*)cache.output_gradient_intrinsics.data_ptr<float>(),
                             cache.output_gradient_intrinsics_count.data_ptr<float>(), n);
        }
    }

    CUDA_SYNC_CHECK_ERROR();

    // gradients for displacement field are equal to point gradients for that batch, as:
    //  point_pos = original_point_pos + displacements
    //  thus:
    //  d_point_pos / d_displacements = 1
    //  d_point_pos / d_original_point_pos = 1


    return {std::move(cache.output_gradient_texture),    std::move(cache.output_gradient_background),
            std::move(cache.output_gradient_points),     std::move(cache.output_gradient_pose_tangent),
            std::move(cache.output_gradient_intrinsics), std::move(cache.output_gradient_confidence),
            std::move(cache.output_gradient_layer)};
}
__global__ void ApplyTangent(Vec6* tangent, Sophus::SE3d* pose, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    Vec6 t = tangent[tid];
    auto p = pose[tid];

    // TODO check magic rotation scaling value
    t.template tail<3>() *= 0.1;
#ifdef _WIN32
    Sophus::SE3d p2(Sophus::se3_expd(t) * p);
    for (int i = 0; i < 7; ++i) pose[tid].data()[i] = p2.data()[i];
#else
    p         = Sophus::se3_expd(t) * p;
    pose[tid] = p;
#endif

    tangent[tid] = Vec6::Zero();
}

void ApplyTangentToPose(torch::Tensor tangent, torch::Tensor pose)
{
    SAIGA_ASSERT(tangent.is_contiguous() && pose.is_contiguous());
    int n = tangent.size(0);
    int c = iDivUp(n, 128);
    ApplyTangent<<<c, 128>>>((Vec6*)tangent.data_ptr<double>(), (Sophus::SE3d*)pose.data_ptr<double>(), n);
    CUDA_SYNC_CHECK_ERROR();
}