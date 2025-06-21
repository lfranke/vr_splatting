// Copyright (c) 2023 Janusch Patas.
// Adapted by Linus Franke (2024).
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and
// MPII.
#pragma once

// #include "camera.cuh"
#include "gaussian.cuh"
#include "parameters.cuh"
#include "rasterizer.cuh"

#include <cmath>
#include <torch/torch.h>

#include "sh_utils_gs.cuh"

float focal2fov(float focal, int pixels)
{
    return 2 * std::atan(static_cast<float>(pixels) / (2.f * focal));
}

torch::Tensor getWorld2View2(const mat3& R, const vec3& t, const vec3& translate /*= Eigen::Vector3d::Zero()*/,
                             float scale /*= 1.0*/)
{
    mat4 Rt              = mat4::Zero();
    Rt.block<3, 3>(0, 0) = R.transpose();
    Rt.block<3, 1>(0, 3) = t;
    Rt(3, 3)             = 1.0;

    mat4 C2W              = Rt.inverse();
    vec3 cam_center       = C2W.block<3, 1>(0, 3);
    cam_center            = (cam_center + translate) * scale;
    C2W.block<3, 1>(0, 3) = cam_center;
    Rt                    = C2W.inverse();
    // Here we create a torch::Tensor from the Eigen::Matrix
    // Note that the tensor will be on the CPU, you may want to move it to the desired device later
    auto RtTensor = torch::from_blob(Rt.data(), {4, 4}, torch::kFloat);
    // clone the tensor to allocate new memory, as from_blob shares the same memory
    // this step is important if Rt will go out of scope and the tensor will be used later
    return RtTensor.clone();
}


torch::Tensor getProjectionMatrix(float znear, float zfar, float fovX, float fovY)
{
    float tanHalfFovY = std::tan((fovY / 2.f));
    float tanHalfFovX = std::tan((fovX / 2.f));

    float top    = tanHalfFovY * znear;
    float bottom = -top;
    float right  = tanHalfFovX * znear;
    float left   = -right;

    mat4 P = mat4::Zero();

    float z_sign = 1.f;

    P(0, 0) = 2.f * znear / (right - left);
    P(1, 1) = 2.f * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);

    // create torch::Tensor from Eigen::Matrix
    auto PTensor = torch::from_blob(P.data(), {4, 4}, torch::kFloat);

    // clone the tensor to allocate new memory
    return PTensor.clone();
}

inline mat3 qvec2rotmat(const quat& q)
{
    vec4 qvec = q.coeffs();  // [x, y, z, w]

    mat3 rotmat;
    rotmat(0, 0) = 1.f - 2.f * qvec[2] * qvec[2] - 2.f * qvec[3] * qvec[3];
    rotmat(0, 1) = 2.f * qvec[1] * qvec[2] - 2.f * qvec[0] * qvec[3];
    rotmat(0, 2) = 2.f * qvec[3] * qvec[1] + 2.f * qvec[0] * qvec[2];
    rotmat(1, 0) = 2.f * qvec[1] * qvec[2] + 2.f * qvec[0] * qvec[3];
    rotmat(1, 1) = 1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[3] * qvec[3];
    rotmat(1, 2) = 2.f * qvec[2] * qvec[3] - 2.f * qvec[0] * qvec[1];
    rotmat(2, 0) = 2.f * qvec[3] * qvec[1] - 2.f * qvec[0] * qvec[2];
    rotmat(2, 1) = 2.f * qvec[2] * qvec[3] + 2.f * qvec[0] * qvec[1];
    rotmat(2, 2) = 1.f - 2.f * qvec[1] * qvec[1] - 2.f * qvec[2] * qvec[2];

    return rotmat;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(
    NeuralScene& scene, const NeuralTrainData& batch, std::shared_ptr<GaussianModel>& gaussianModel,
    torch::Tensor& bg_color, bool viewer_only = false, CUDA::CudaTimerSystem* timer_system = nullptr,
    float scaling_modifier = 1.0, torch::Tensor override_color = torch::empty({}))
{
    // SAIGA_ASSERT(batch.size() == 1);
    //  get camera data to opengl/3Dgs
    SAIGA_OPTIONAL_TIME_MEASURE("Gaussian Render", timer_system);

    GaussianRasterizer rasterizer;
    auto scales    = torch::Tensor();
    auto rotations = torch::Tensor();
    torch::Tensor means3D;
    torch::Tensor means2D;
    torch::Tensor opacity;
    torch::Tensor shs            = torch::Tensor();
    torch::Tensor colors_precomp = torch::Tensor();
    auto cov3D_precomp           = torch::Tensor();


    {
        SAIGA_OPTIONAL_TIME_MEASURE("GS Start", timer_system);

        float render_scale = scene.scene->dataset_params.render_scale;

        int image_height = batch->img.h;
        int image_width  = batch->img.w;



        // auto cam    = scene.poses->Download(batch.front()->img.image_index);
        // auto gl_mat = cam.matrix().cast<float>();
        // torch::Tensor world_view_transform =
        //     torch::from_blob(gl_mat.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA);

        // PrintTensorInfo(world_view_transform);

        // PrintTensorInfo(world_view_transform);
        // PrintTensorInfo(scene.poses->GetPoseMat4(batch.front()->img.image_index));

        // torch::Tensor proj_mat            = getProjectionMatrix(0.1f, 1000.f, fov_x, fov_y).to(torch::kCUDA);
        torch::Tensor proj_mat;
        torch::Tensor full_proj_transform;
        torch::Tensor cam_center;
        torch::Tensor inv_viewprojmatrix;
        torch::Tensor world_view_transform;

        float fov_x = 0;
        float fov_y = 0;
        {
            if (batch->img.own_proj_mat)
            {
                SAIGA_OPTIONAL_TIME_MEASURE("blob", timer_system);
                if (viewer_only && batch->vrgs.proj_mat.defined())
                {
                    proj_mat = batch->vrgs.proj_mat;
                    fov_y    = batch->vrgs.fovs.y();
                    fov_x    = batch->vrgs.fovs.x();
                }
                else
                {
                    proj_mat =
                        torch::from_blob(batch->img.proj_mat.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA);
                    fov_y        = atan(1.f / batch->img.proj_mat(1, 1)) * 2.f;
                    float aspect = batch->img.proj_mat(1, 1) / batch->img.proj_mat(0, 0);
                    fov_x        = 2 * atan(tan(fov_y * 0.5) * aspect);
                }
            }
            else
            {
                auto K_crop = batch->img.crop_transform;
                auto K_cam  = scene.intrinsics->DownloadK()[batch->img.camera_index];  // TODO add camera index
                auto K      = K_cam;                                                   // TODO add crop
                fov_x       = focal2fov(K.fx * render_scale, image_width);
                fov_y       = focal2fov(K.fy * render_scale, image_height);

                proj_mat = getProjectionMatrix(0.1f, 1000.f, fov_x, fov_y).to(torch::kCUDA);
            }
            // PrintTensorInfo(proj_mat);

            // PrintTensorInfo(world_view_transform);
            {
                SAIGA_OPTIONAL_TIME_MEASURE("inv", timer_system);
                if (viewer_only && batch->vrgs.proj_mat.defined())
                {
                    full_proj_transform  = batch->vrgs.full_proj_transform;
                    inv_viewprojmatrix   = batch->vrgs.inv_viewprojmatrix;
                    cam_center           = batch->vrgs.cam_center;
                    world_view_transform = batch->vrgs.world_view_transform;
                }
                else
                {
                    world_view_transform = scene.poses->GetPoseMat4(batch->img.image_index);

                    // auto pose = scene.poses->GetPose(batch->img.image_index).squeeze();
                    // cam_center          = pose.slice(0, 4, 7).contiguous().clone();
                    full_proj_transform = world_view_transform.unsqueeze(0).bmm(proj_mat.unsqueeze(0)).squeeze(0);
                    // cam_center          = world_view_transform.inverse()[3].slice(0, 0, 3);
                    inv_viewprojmatrix = full_proj_transform.inverse();
                    cam_center =
                        world_view_transform.inverse().index({3, torch::indexing::Slice(0, 3)}).clone().contiguous();
                }
            }
        }
        // Ensure background tensor (bg_color) is on GPU!
        bg_color = bg_color.to(torch::kCUDA);

        // Set up rasterization configuration
        GaussianRasterizationSettings raster_settings;
        raster_settings.image_height       = static_cast<int>(image_height);
        raster_settings.image_width        = static_cast<int>(image_width);
        raster_settings.tanfovx            = std::tan(fov_x * 0.5f);
        raster_settings.tanfovy            = std::tan(fov_y * 0.5f);
        raster_settings.bg                 = bg_color;
        raster_settings.scale_modifier     = scaling_modifier;
        raster_settings.viewmatrix         = world_view_transform;
        raster_settings.projmatrix         = full_proj_transform;
        raster_settings.inv_viewprojmatrix = inv_viewprojmatrix;
        raster_settings.sh_degree          = gaussianModel->Get_active_sh_degree();
        raster_settings.camera_center      = cam_center;
        raster_settings.prefiltered        = false;
        raster_settings.splatting_args     = scene.stp_splatting_args;
        raster_settings.render_depth       = false;
        raster_settings.debug              = false;
        raster_settings.viewer_only        = viewer_only;

        rasterizer = GaussianRasterizer(raster_settings, timer_system);

        means3D = gaussianModel->Get_xyz();
        // PrintTensorInfo(means3D);
        means2D = torch::zeros_like(gaussianModel->Get_xyz()).requires_grad_(true);
        means2D.retain_grad();
        opacity = gaussianModel->Get_opacity();


        scales    = gaussianModel->Get_scaling();
        rotations = gaussianModel->Get_rotation();

        shs = gaussianModel->Get_features();
        // torch::cuda::synchronize();
    }



    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto [rendered_image, radii] =
        rasterizer.forward(means3D, means2D, opacity, shs, colors_precomp, scales, rotations, cov3D_precomp);

    // Apply visibility filter to remove occluded Gaussians.
    // render, viewspace_points, visibility_filter, radii
    return {rendered_image, means2D, radii > 0, radii};
}

/*
static std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t)
{
    auto lambda = [&t](size_t N)
    {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}*/
std::function<char*(size_t N)> resizeFunctional(void** ptr, size_t& S)
{
    auto lambda = [ptr, &S](size_t N)
    {
        if (N > S)
        {
            if (*ptr) CHECK_CUDA_ERROR(cudaFree(*ptr));
            CHECK_CUDA_ERROR(cudaMalloc(ptr, 2 * N));
            S = 2 * N;
        }
        return reinterpret_cast<char*>(*ptr);
    };
    return lambda;
}
static void *geomPtr = nullptr, *binningPtr = nullptr, *imgPtr = nullptr;
static size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;


template <bool half_float>
inline torch::Tensor render_inference(NeuralScene& scene, const NeuralTrainData& batch,
                                      std::shared_ptr<GaussianModel>& gaussianModel, torch::Tensor& bg_color,
                                      bool viewer_only = true, CUDA::CudaTimerSystem* timer_system = nullptr,
                                      float scaling_modifier = 1.f, torch::Tensor override_color = torch::empty({}))
{
    // SAIGA_ASSERT(batch.size() == 1);
    //  get camera data to opengl/3Dgs
    SAIGA_OPTIONAL_TIME_MEASURE("Gaussian Render", timer_system);

    float render_scale  = scene.scene->dataset_params.render_scale;
    int image_height    = batch->img.h;
    int image_width     = batch->img.w;
    bool render_depth   = false;
    int degree          = gaussianModel->Get_active_sh_degree();
    auto splatting_args = scene.stp_splatting_args;
    // static float sc_mod = scaling_modifier;
    // ImGui::SliderFloat("slider scale", &sc_mod, 0, 2);
    // scaling_modifier = sc_mod;

    torch::Tensor means3D   = gaussianModel->Get_xyz();
    torch::Tensor opacity   = gaussianModel->Get_opacity();
    torch::Tensor scales    = gaussianModel->Get_scaling();
    torch::Tensor rotations = gaussianModel->Get_rotation();
    torch::Tensor sh        = gaussianModel->Get_features();


    torch::Tensor proj_mat;
    torch::Tensor viewprojmatrix;
    torch::Tensor cam_center;
    torch::Tensor inv_viewprojmatrix;
    torch::Tensor viewmatrix;
    float fov_x = 0;
    float fov_y = 0;
    {
        if (batch->img.own_proj_mat)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("blob", timer_system);
            if (viewer_only && batch->vrgs.proj_mat.defined())
            {
                proj_mat = batch->vrgs.proj_mat;
                fov_y    = batch->vrgs.fovs.y();
                fov_x    = batch->vrgs.fovs.x();
            }
            else
            {
                proj_mat = torch::from_blob(batch->img.proj_mat.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA);
                fov_y    = atan(1.f / batch->img.proj_mat(1, 1)) * 2.f;
                float aspect = batch->img.proj_mat(1, 1) / batch->img.proj_mat(0, 0);
                fov_x        = 2.f * atan(tan(fov_y * 0.5f) * aspect);
            }
        }
        else
        {
            auto K_crop = batch->img.crop_transform;
            auto K_cam  = scene.intrinsics->DownloadK()[batch->img.camera_index];  // TODO add camera index
            auto K      = K_cam;                                                   // TODO add crop
            fov_x       = focal2fov(K.fx * render_scale, image_width);
            fov_y       = focal2fov(K.fy * render_scale, image_height);

            proj_mat = getProjectionMatrix(0.1f, 1000.f, fov_x, fov_y).to(torch::kCUDA);
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("inv", timer_system);
            if (viewer_only && batch->vrgs.proj_mat.defined())
            {
                viewprojmatrix     = batch->vrgs.full_proj_transform;
                inv_viewprojmatrix = batch->vrgs.inv_viewprojmatrix;
                cam_center         = batch->vrgs.cam_center;
                viewmatrix         = batch->vrgs.world_view_transform;
            }
            else
            {
                viewmatrix         = scene.poses->GetPoseMat4(batch->img.image_index);
                viewprojmatrix     = viewmatrix.unsqueeze(0).bmm(proj_mat.unsqueeze(0)).squeeze(0);
                inv_viewprojmatrix = viewprojmatrix.inverse();
                cam_center         = viewmatrix.inverse().index({3, torch::indexing::Slice(0, 3)}).clone().contiguous();
            }
        }
    }
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    if (half_float) float_opts = float_opts.dtype(torch::kHalf);
    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;


    const int NUM_CHANNELS  = 3;
    torch::Tensor out_color = torch::full({NUM_CHANNELS + 1, H, W}, 0.0, float_opts);

    out_color.slice(0, NUM_CHANNELS, NUM_CHANNELS + 1) = 10000;
    // torch::Tensor radii     = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    // static torch::Tensor geomBuffer          = torch::empty({0}, options.device(device));
    // static torch::Tensor binningBuffer       = torch::empty({0}, options.device(device));
    // static torch::Tensor imgBuffer           = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc    = resizeFunctional(&geomPtr, allocdGeom);
    std::function<char*(size_t)> binningFunc = resizeFunctional(&binningPtr, allocdBinning);
    std::function<char*(size_t)> imgFunc     = resizeFunctional(&imgPtr, allocdImg);

    int rendered = 0;
    if (P != 0)
    {
        int M = 0;
        if (sh.size(0) != 0)
        {
            M = sh.size(1);
            // std::cout << M << std::endl;
        }

        // CudaRasterizer::SplattingSettings settings = settings_dict.get<CudaRasterizer::SplattingSettings>();
        DebugVisualizationData debug_data{
            DebugVisualization::Disabled, 0, 0,
            [](const DebugVisualizationData& instance, float value, float min, float max, float avg, float std)
            {
                std::cout << toString(instance.type) << " for pixel (" << instance.debugPixel[0] << ", "
                          << instance.debugPixel[1] << "): value=" << value << ", min=" << min << ", max=" << max
                          << ", avg=" << avg << ", std=" << std << std::endl;
            }};
        if (render_depth)
        {
            debug_data.type = DebugVisualization::Depth;
        }
        // debug_data.timing_enabled = true;

        float* color_data_ptr =
            half_float ? (float*)out_color.contiguous().data<at::Half>() : out_color.contiguous().data<float>();

        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc, binningFunc, imgFunc, P, degree, M, bg_color.contiguous().data<float>(), W, H, splatting_args,
            debug_data, means3D.contiguous().data<float>(), sh.contiguous().data_ptr<float>(), nullptr,
            opacity.contiguous().data<float>(), scales.contiguous().data_ptr<float>(), scaling_modifier,
            rotations.contiguous().data_ptr<float>(), nullptr, viewmatrix.contiguous().data<float>(),
            viewprojmatrix.contiguous().data<float>(), inv_viewprojmatrix.contiguous().data<float>(),
            cam_center.contiguous().data<float>(), std::tan(fov_x * 0.5f), std::tan(fov_y * 0.5f), false,
            color_data_ptr, nullptr, false, half_float);

        if (debug_data.timing_enabled && !debug_data.timings_text.empty())
            std::cout << debug_data.timings_text.c_str() << std::endl;
    }
    return out_color;
}
