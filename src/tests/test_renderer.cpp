/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/VGGLoss.h"

#include "config.h"
#include "data/Dataset.h"
#include "data/NeuralScene.h"
#include "data/Settings.h"
#include "models/MyAdam.h"
#include "models/Pipeline.h"
#include "rendering/AlphaListSort.h"
#include "rendering/NeuralPointCloudCuda.h"
#include "rendering/PointRenderer.h"
#include "rendering/RenderInfo.h"
using namespace Saiga;

#include "gtest/gtest.h"

#include "numerical_testing.h"

std::shared_ptr<CombinedParams> default_params()
{
    std::shared_ptr<CombinedParams> params                = std::make_shared<CombinedParams>();
    params->train_params.batch_size                       = 1;
    params->pipeline_params.skip_neural_render_network    = true;
    params->train_params.texture_color_init               = true;
    params->render_params.super_sampling                  = false;
    params->render_params.add_depth_to_network            = false;
    params->render_params.super_sampling                  = false;
    params->pipeline_params.enable_environment_map        = false;
    params->pipeline_params.cat_masks_to_color            = false;
    params->render_params.drop_out_points_by_radius       = false;
    params->render_params.dropout                         = 0.f;
    params->render_params.dist_cutoff                     = 3000.f;
    params->render_params.check_normal                    = false;
    params->points_adding_params.sigmoid_narrowing_factor = 0.f;
    params->Check();
    params->net_params.num_input_layers = 4;

    return params;
}

void fixall(std::shared_ptr<CombinedParams> params)
{
    params->optimizer_params.fix_environment_map = true;
    params->optimizer_params.fix_exposure        = true;
    params->optimizer_params.fix_intrinsics      = true;
    params->optimizer_params.fix_motion_blur     = true;
    params->optimizer_params.fix_points          = true;
    params->optimizer_params.fix_poses           = true;
    params->optimizer_params.fix_render_network  = true;
    params->optimizer_params.fix_response        = true;
    params->optimizer_params.fix_rolling_shutter = true;
    params->optimizer_params.fix_texture         = true;
    params->optimizer_params.fix_vignette        = true;
    params->optimizer_params.fix_wb              = true;
    params->optimizer_params.fix_point_size      = true;
}


int RENDER_MODE_SELECT = 4;

struct RendererTesting
{
    RendererTesting(std::shared_ptr<CombinedParams> para, int wid = 32, int hei = 32)
        : params(para), width(wid), height(hei)
    {
        torch::manual_seed(362346);
        Random::setSeed(93467);


        render_module = PointRenderModule(params);
        render_module->train(true);

        int current_epoch = 20;

        IntrinsicsPinholef K_mat(16, 16, 15.5, 15.5, 0);

        std::shared_ptr<SceneData> scene_d = std::make_shared<SceneData>(32, 32, K_mat);
        {
            std::vector<vec3> position({vec3(-0.7, -0.7, 1), vec3(0, 0, 1), vec3(0.7, 0.7, 1)});
            std::vector<vec3> normal({vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 1, 0)});
            std::vector<vec4> color({vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0)});
            UnifiedMesh pc;
            pc.position = position;
            pc.normal   = normal;
            pc.color    = color;

            pc.ReorderMorton64();
            pc.RandomBlockShuffle(256);
            scene_d->point_cloud = pc;
            scene_d->ComputeRadius();
            scene_d->SortBlocksByRadius(256);

            int n_frames = 1;
            scene_d->frames.resize(n_frames);
            std::vector<Sophus::SE3d> poses;
            {
                Vec3 translation = vec3(0, 0, 0);
                Quat q           = Quat::Identity();
                poses.push_back(Sophus::SE3d(q, translation));
            }
            for (int i = 0; i < scene_d->frames.size(); ++i)
            {
                auto& fd        = scene_d->frames[i];
                fd.image_index  = i;
                fd.camera_index = 0;

                if (!poses.empty())
                {
                    fd.pose = poses[i];
                    // std::cout << "pose (" << i << "): " << fd.pose << std::endl;
                }
                // if (!exposures.empty()) fd.exposure_value = exposures[i];
                // if (!wbs.empty()) fd.white_balance = wbs[i];
                // if (!images.empty()) fd.target_file = images[i];
                // if (!masks.empty()) fd.mask_file = masks[i];

                auto cam      = scene_d->scene_cameras[fd.camera_index];
                fd.K          = cam.K;
                fd.distortion = cam.distortion;
                fd.w          = cam.w;
                fd.h          = cam.h;
            }
        }

        auto& f = scene_d->frames[0];
        std::cout << "CPU RENDER" << std::endl;
        cpu_img = scene_d->CPURenderFrame(0, 1.f);

        cpu_img.save("asdf_cpu.png");
        cpu_img_bilinear = scene_d->CPURenderFrameBilinear(0, 1.f);
        cpu_img_bilinear.save("asdf_cpu_bilinear.png");



        scene = std::make_shared<NeuralScene>(scene_d, params);

        scene->to(torch::kCUDA);
        PrintTensorInfo(scene->point_cloud_cuda->t_position);
        PrintTensorInfo(scene->point_cloud_cuda->t_point_size);
    }
    torch::Tensor render(PointRendererCache::RenderMode rendermode, int current_epoch = 20, int layer = 0,
                         std::string save_img = std::string())
    {
        std::vector<NeuralTrainData> batch;
        NeuralTrainData pd = std::make_shared<TorchFrameData>();
        {
            auto t     = ImageViewToTensor(cpu_img.getImageView()).clone();
            pd->target = t;
            // return;
            pd->camera_index = torch::zeros({1});
            pd->scale        = torch::ones({1});
            pd->scene_id     = 0;

            pd->img.crop_transform = IntrinsicsPinholef();
            pd->img.h              = height;
            pd->img.w              = width;
            pd->img.camera_index   = 0;
            pd->img.image_index    = 0;
            Matrix<float, 2, 2> crop_rot;
            crop_rot.setZero();
            crop_rot(0, 0)        = 1;
            crop_rot(1, 1)        = 1;
            pd->img.crop_rotation = crop_rot;
            pd->to(torch::kCUDA);
        }
        batch.push_back(pd);

        std::vector<torch::Tensor> neural_images;
        std::vector<torch::Tensor> masks;
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Sigmoid on confidence", timer_system);
            float narrowing_fac = 0;
            scene->texture->PrepareConfidence(narrowing_fac);
        }
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Prep Tex", timer_system);
            scene->texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
        }
        scene->dynamic_refinement_t = torch::zeros({1}).cuda();

        params->render_params.render_mode = rendermode;
        render_module->params             = params;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("Render", timer_system);
            std::tie(neural_images, masks) = render_module->forward(*scene, batch, current_epoch, timer_system);
        }
        if (!save_img.empty())
        {
            auto img_gpu = TensorToImage<ucvec3>(neural_images[0]);
            img_gpu.save("asdf_" + save_img + "_gpu.png");
        }
        // std::cout << neural_images.size() << std::endl;
        return neural_images[layer];
    }


    double getInitialGradient(torch::Tensor& param, at::ArrayRef<at::indexing::TensorIndex> index_in_param,
                              torch::Tensor target, PointRendererCache::RenderMode render_mode, int layer,
                              double eps = 1e-4, bool verbose = false)
    {
        // initial render
        auto neural_img = render(render_mode, 20, layer);
        if (verbose) std::cout << TensorInfo(neural_img) << TensorInfo(target) << std::endl;
        auto loss = torch::l1_loss(neural_img, target);
        if (verbose)
        {
            std::cout << "Target: " << TensorInfo(target) << " - Rendered: " << TensorInfo(neural_img) << std::endl;
            std::cout << "Loss initial: " << loss.item<double>() << std::endl;
        }
        loss.backward();

        if (verbose) std::cout << TensorInfo(param.mutable_grad()) << std::endl;
        if (verbose) std::cout << TensorInfo(param.mutable_grad().index(index_in_param)) << std::endl;
        auto gradient_init = param.mutable_grad().index(index_in_param).item<double>();
        if (verbose) std::cout << "backwards gradient: " << gradient_init << std::endl;

        if (verbose) std::cout << "before change" << TensorInfo(param) << std::endl;

        if (verbose)
        {
            auto img_gpu = TensorToImage<ucvec3>(neural_img);
            img_gpu.save("optim_evalparam.png");
            auto img_gt = TensorToImage<ucvec3>(target);
            img_gt.save("optim_evalparam_gt.png");
        }

        return gradient_init;
    }


    double getFiniteDiff(torch::Tensor& param, at::ArrayRef<at::indexing::TensorIndex> index_in_param,
                         torch::Tensor target, PointRendererCache::RenderMode render_mode, int layer, double eps = 1e-4,
                         bool verbose = false)
    {
        auto t = param.clone();

        // finite diff:
        double r = 0;
        {
            {
                torch::NoGradGuard ngg;
                param.index(index_in_param) += eps;
            }
            if (verbose) std::cout << "after plus" << TensorInfo(param) << std::endl;
            auto neural_img_plus = render(render_mode, 20, layer);
            auto loss            = torch::l1_loss(neural_img_plus, target);
            if (verbose) std::cout << "loss +plus" << loss.item<float>() << std::endl;

            r += loss.item<double>();

            {
                torch::NoGradGuard ngg;

                // reset
                param.set_(t.clone());
            }
        }
        {
            {
                torch::NoGradGuard ngg;
                param.index(index_in_param) -= eps;
            }
            if (verbose) std::cout << "after minus" << TensorInfo(param) << std::endl;

            auto neural_img_plus = render(render_mode, 20, layer);
            auto loss            = torch::l1_loss(neural_img_plus, target);
            if (verbose) std::cout << "l-" << loss.item<float>() << std::endl;

            r -= loss.item<double>();
            {
                torch::NoGradGuard ngg;

                // reset
                param.set_(t.clone());
            }
        }
        return r;
    }
    inline bool SameSign(float a, float b) { return a * b >= 0.0f; }


    int EvaluateParam(torch::Tensor& param, at::ArrayRef<at::indexing::TensorIndex> index_in_param,
                      torch::Tensor target, PointRendererCache::RenderMode render_mode, int layer, double eps = 1e-4,
                      bool verbose = false)
    {
        if (verbose) std::cout << TensorInfo(param) << std::endl;
        // SAIGA_ASSERT(param.sizes() == 1 && param.size(0) == 1);
        double eps_scale = 1.0 / (2.0 * eps);
        auto optim       = makeOptimizerForParam(param);
        if (verbose) std::cout << ">>>>>>>>>>>>>>> ININININITAL" << std::endl;
        auto gradient_init = getInitialGradient(param, index_in_param, target, render_mode, layer, eps, verbose);
        if (verbose) std::cout << "<<<<<<<<<<<<<<< ININININITAL" << std::endl;

        double r = getFiniteDiff(param, index_in_param, target, render_mode, layer, eps, verbose);
        r *= eps_scale;

        // if (verbose)
        double absolute_difference = fabs(r - gradient_init);
        // EXPECT_NEAR(r, gradient_init, absolute_difference);
        bool close = absolute_difference <= eps;

        // bool close = ExpectClose(r, gradient_init, eps);
        //  if (verbose)
        //  std::cout << "OutPut_gradients:" << r << " _____ " << gradient_init << ": " << r / gradient_init << ";; "
        //            << absolute_difference << std::endl;

        if (!SameSign(r, gradient_init) || !close)
        {
            std::cout << r << " _____ " << gradient_init << ": " << r / gradient_init << ";; " << absolute_difference
                      << std::endl;
            if (verbose)
            {
                std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
                std::cout << r << " _____ " << gradient_init << ": " << r / gradient_init << std::endl;
                std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << index_in_param << param.index(index_in_param)
                          << std::endl
                          << std::endl;
            }
            // if (!close) CHECK(false);
        }
        //  CHECK(close);
        int result = close ? 1 : 0;
        if (gradient_init == 0 && r == 0) result = 2;
        return result;
    }


    int EvaluatePose(at::ArrayRef<at::indexing::TensorIndex> index_in_param, torch::Tensor target,
                     PointRendererCache::RenderMode render_mode, int layer, double eps = 1e-4, bool verbose = false)
    {
        if (verbose) std::cout << TensorInfo(scene->poses->tangent_poses) << std::endl;
        // SAIGA_ASSERT(param.sizes() == 1 && param.size(0) == 1);
        double eps_scale = 1.0 / (2.0 * eps);
        auto optim       = makeOptimizerForParam(scene->poses->tangent_poses);

        auto gradient_init =
            getInitialGradient(scene->poses->tangent_poses, index_in_param, target, render_mode, layer, eps, verbose);


        auto t = scene->poses->tangent_poses.clone();
        auto p = scene->poses->poses_se3.clone();

        // finite diff:
        double r = 0;
        {
            {
                torch::NoGradGuard ngg;

                scene->poses->tangent_poses.index(index_in_param) += eps;
                scene->poses->ApplyTangent();
            }
            if (verbose) std::cout << "poses after plus" << TensorInfo(scene->poses->poses_se3) << std::endl;
            auto neural_img_plus = render(render_mode, 20, layer);
            auto loss            = torch::l1_loss(neural_img_plus, target);
            if (verbose) std::cout << "loss +plus" << loss.item<float>() << std::endl;

            r += loss.item<double>();

            {
                torch::NoGradGuard ngg;

                // reset
                scene->poses->tangent_poses.set_(t.clone());
                scene->poses->poses_se3.set_(p.clone());
            }
        }
        {
            {
                torch::NoGradGuard ngg;
                scene->poses->tangent_poses.index(index_in_param) -= eps;
                scene->poses->ApplyTangent();
            }
            if (verbose) std::cout << "after minus" << TensorInfo(scene->poses->poses_se3) << std::endl;

            auto neural_img_plus = render(render_mode, 20, layer);
            auto loss            = torch::l1_loss(neural_img_plus, target);
            if (verbose) std::cout << "l-" << loss.item<float>() << std::endl;

            r -= loss.item<double>();
            {
                torch::NoGradGuard ngg;

                // reset
                scene->poses->tangent_poses.set_(t.clone());
                scene->poses->poses_se3.set_(p.clone());
            }
        }
        r *= eps_scale;
        double absolute_difference = fabs(r - gradient_init);
        // EXPECT_NEAR(r, gradient_init, absolute_difference);
        bool close = absolute_difference <= eps;
        // if (verbose)
        if (!SameSign(r, gradient_init) || !close)
        {
            std::cout << r << " _____ " << gradient_init << ": " << r / gradient_init << ";; " << absolute_difference
                      << std::endl;
            if (verbose)
            {
                std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
                std::cout << r << " _____ " << gradient_init << ": " << r / gradient_init << std::endl;
                std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << index_in_param << std::endl << std::endl;
            }
            // if (!close) CHECK(false);
        }

        // bool close = ExpectClose(r, gradient_init, eps * 100);
        // CHECK(close);
        int result = close ? 1 : 0;
        if (gradient_init == 0 && r == 0) result = 2;
        return result;
    }

    std::shared_ptr<torch::optim::SGD> makeOptimizerForParam(torch::Tensor& param)
    {
        using TexOpt   = torch::optim::SGDOptions;
        using TexOptim = torch::optim::SGD;
        std::vector<torch::optim::OptimizerParamGroup> g;
        {
            auto opt_t = std::make_unique<TexOpt>(1);
            std::vector<torch::Tensor> ts;
            ts.push_back(param);
            g.emplace_back(ts, std::move(opt_t));
        }
        auto texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
        texture_optimizer->zero_grad();
        return texture_optimizer;
    }
    int width, height;
    std::shared_ptr<CombinedParams> params;
    TemplatedImage<ucvec3> cpu_img, cpu_img_bilinear;
    std::shared_ptr<NeuralScene> scene;
    PointRenderModule render_module = nullptr;
    // PointRendererCache cache;
    CUDA::CudaTimerSystem* timer_system = nullptr;
};


/*
TEST(RendererTest, BilinearRendererTest)
{
    auto def = default_params();
    RendererTesting rendertest(def, 32, 32);
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    int i = 3;
    {
        std::cout << "Render with mode " << i << std::endl;
        rendertest.render(PointRendererCache::RenderMode(i), 20, "render" + std::to_string(i));
    }
}*/


torch::Tensor get_target(RendererTesting& rendertest, PointRendererCache::RenderMode render_mode, int layer)
{
    auto target_img = rendertest.cpu_img_bilinear.getImageView();
    if (render_mode < 3) target_img = rendertest.cpu_img.getImageView();
    int factor                        = pow(2, layer);
    TemplatedImage<ucvec3> img_scaled = TemplatedImage<ucvec3>(target_img.h / factor, target_img.w / factor);
    target_img.copyScaleDownPow2(img_scaled.getImageView(), factor);
    torch::Tensor target = ImageViewToTensor(img_scaled.getImageView());
    target               = target.to(torch::kFloat32).clone().cuda().contiguous().unsqueeze(0);
    return target;
}



std::pair<int, int> testConfGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_texture = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        rendertest.scene->texture->confidence_raw *= torch::rand_like(rendertest.scene->texture->confidence_raw);
        rendertest.scene->texture->confidence_raw += 0.2 * torch::rand_like(rendertest.scene->texture->confidence_raw);
    }
    std::cout.clear();
    // std::cout << "RANDOMED: " << rendertest.scene->texture->confidence_raw << std::endl;

    torch::Tensor target = get_target(rendertest, render_mode, layer);

    int num_failed = 0;
    int tests_size = int(rendertest.scene->texture->confidence_raw.size(1));

    for (int i = 0; i < rendertest.scene->texture->confidence_raw.size(1); ++i)
    {
        auto param = rendertest.scene->texture->confidence_raw;  //.squeeze();

        int result = rendertest.EvaluateParam(param, {0, i}, target, render_mode, layer, 1e-4, false);
        if (!result) num_failed++;
        if (result == 2) tests_size--;
    }
    auto ret = std::make_pair(num_failed, tests_size);
    return ret;
}

TEST(RendererTest, ConfidenceGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    for (int layer = 0; layer < 4; ++layer)
    {
        int num_fail = 0;
        int tests    = 0;
        int all      = 0;
        std::cout << "Test conf grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testConfGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "Conf grad test (layer " << layer << "): " << num_fail << "/" << all << " failed." << std::endl;
    }
}


std::pair<int, int> testPointSizeGradientTest(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_point_size    = false;
    params->render_params.use_layer_point_size = true;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        PrintTensorInfo(rendertest.scene->point_cloud_cuda->t_point_size);

        rendertest.scene->point_cloud_cuda->t_point_size.set_(
            log(exp(torch::rand_like(rendertest.scene->point_cloud_cuda->t_point_size) * 4) - 1));
    }
    std::cout.clear();
    // std::cout << "RANDOMED: " << rendertest.scene->texture->confidence_raw << std::endl;

    torch::Tensor target = get_target(rendertest, render_mode, layer);

    int num_failed = 0;
    int tests_size = rendertest.scene->point_cloud_cuda->t_point_size.size(0);

    for (int i = 0; i < rendertest.scene->point_cloud_cuda->t_point_size.size(0); ++i)
    {
        auto param = rendertest.scene->point_cloud_cuda->t_point_size;  //.squeeze();

        int result = rendertest.EvaluateParam(param, {i, 0}, target, render_mode, layer, 1e-4, false);
        if (!result) num_failed++;
        if (result == 2) tests_size--;
    }
    auto ret = std::make_pair(num_failed, tests_size);
    return ret;
}

TEST(RendererTest, PointSizeGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    for (int layer = 0; layer < 4; ++layer)
    {
        int num_fail = 0;
        int tests    = 0;
        int all      = 0;
        std::cout << "Test PointSizeGradient with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 10000; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) =
                testPointSizeGradientTest(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "PointSizeGradientTest (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}



std::pair<int, int> testPosGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_points = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        // rendertest.scene->point_cloud_cuda->t_position =
        //     torch::rand_like(rendertest.scene->point_cloud_cuda->t_position) * 0.001;

        auto rand_ =
            0.2 * (torch::rand_like(rendertest.scene->point_cloud_cuda->t_position.slice(1, 0, 2))) * 0.05 + 0.005;

        rendertest.scene->point_cloud_cuda->t_position.slice(1, 0, 2) += rand_;
    }
    std::cout.clear();
    // std::cout << "RANDOMED: " << rendertest.scene->point_cloud_cuda->t_position << std::endl;

    torch::Tensor target = get_target(rendertest, render_mode, layer);

    int num_failed = 0;
    int tests_size = rendertest.scene->point_cloud_cuda->t_position.size(0) *
                     (rendertest.scene->point_cloud_cuda->t_position.size(1) - 1);
    for (int j = 0; j < rendertest.scene->point_cloud_cuda->t_position.size(0); ++j)
    {
        for (int i = 0; i < rendertest.scene->point_cloud_cuda->t_position.size(1) - 2; ++i)
        {
            auto param = rendertest.scene->point_cloud_cuda->t_position;  //.squeeze();
            int result = rendertest.EvaluateParam(param, {j, i}, target, render_mode, layer, 1e-4, false);
            if (!result) num_failed++;
            if (result == 2) tests_size--;
        }
    }
    auto ret = std::make_pair(num_failed, int(tests_size));

    // std::cout << num_failed << " _ "
    //           << int(rendertest.scene->point_cloud_cuda->t_position.size(0) *
    //                  (rendertest.scene->point_cloud_cuda->t_position.size(1) - 2))
    //           << std::endl;

    return ret;
}

TEST(RendererTest, PositionGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    for (int layer = 0; layer < 4; ++layer)
    {
        int num_fail = 0;
        int tests    = 0;
        int all      = 0;
        std::cout << "Test position grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testPosGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "Position Gradient Test (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}



std::pair<int, int> testIntrinsicsGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_intrinsics = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        rendertest.scene->intrinsics->intrinsics +=
            0.2 * (torch::rand_like(rendertest.scene->intrinsics->intrinsics) * 2 - 1) * 0.05;
    }
    rendertest.scene->intrinsics->train(true);
    std::cout.clear();
    //   std::cout << "RANDOMED: " << rendertest.scene->intrinsics->intrinsics << std::endl;

    torch::Tensor target = get_target(rendertest, render_mode, layer);



    int num_failed = 0;
    int tests_size = rendertest.scene->intrinsics->intrinsics.size(1);

    for (int i = 0; i < tests_size; ++i)
    {
        auto param = rendertest.scene->intrinsics->intrinsics;  //.squeeze();
        int result = rendertest.EvaluateParam(param, {0, i}, target, render_mode, layer, 1e-4, false);
        if (!result) num_failed++;
        if (result == 2) tests_size--;
    }
    auto ret = std::make_pair(num_failed, int(tests_size));
    return ret;
}

TEST(RendererTest, IntrinsicsGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    for (int layer = 0; layer < 4; ++layer)
    {
        int all      = 0;
        int num_fail = 0;
        int tests    = 0;
        std::cout << "Test intrinsics grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testIntrinsicsGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "Intrinsics Gradient Test (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}


std::pair<int, int> testCamPoseGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_poses = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        /*
        //        rendertest.scene->scene->frames.front().pose
        {
            Quat q = rendertest.scene->scene->frames.front().pose.unit_quaternion();
            q      = Sophus::SO3d::exp(Random::MatrixGauss<Vec3>(0, sdev_rot)).unit_quaternion() * q;
            rendertest.scene->scene->frames.front().pose.setQuaternion(q);
        }

        {
            rendertest.scene->scene->frames.front().pose.translation() += Random::MatrixGauss<Vec3>(0, sdev_trans);
        }
        rendertest.scene->poses->poses_se3.set_(poses2->poses_se3);
        */

        //  float noise_pose_r = 0.02f;
        //  float noise_pose_t = 0.02f;
        //  rendertest.scene->scene->AddPoseNoise(radians(noise_pose_r), noise_pose_t);
        //
        //  auto poses2 = PoseModule(rendertest.scene->scene);
        //  poses2->to(torch::kCUDA);
        //  rendertest.scene->poses->to(torch::kCUDA);
        //
        //  // PrintTensorInfo(ns->poses->poses_se3);
        //  // PrintTensorInfo(poses2->poses_se3);
        //  rendertest.scene->poses->poses_se3.set_(poses2->poses_se3);



        rendertest.scene->poses->poses_se3 +=
            0.2 * (torch::rand_like(rendertest.scene->poses->poses_se3) * 2 - 1) * 0.05;
    }
    std::cout.clear();

    //  std::cout << "RANDOM poses: " << rendertest.scene->poses->poses_se3 << std::endl;
    torch::Tensor target = get_target(rendertest, render_mode, layer);


    int num_failed = 0;
    int tests_size = rendertest.scene->poses->tangent_poses.size(0) * (rendertest.scene->poses->tangent_poses.size(1));

    for (int j = 0; j < rendertest.scene->poses->tangent_poses.size(0); ++j)
    {
        for (int i = 0; i < rendertest.scene->poses->tangent_poses.size(1); ++i)
        {
            auto param = rendertest.scene->poses->tangent_poses;  //.squeeze();
            int result = rendertest.EvaluatePose({j, i}, target, render_mode, layer, 1e-4, false);
            if (!result) num_failed++;
            if (result == 2) tests_size--;
        }
    }
    auto ret = std::make_pair(num_failed, int(tests_size));
    return ret;
}

TEST(RendererTest, PoseGradientTest)
{
    for (int layer = 0; layer < 4; ++layer)
    {
        int all      = 0;
        int num_fail = 0;
        int tests    = 0;
        std::cout << "Test pose grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testCamPoseGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "Pose Gradient Test (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}


std::pair<int, int> testTextureGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_texture = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());

        rendertest.scene->texture->texture_raw *= torch::rand_like(rendertest.scene->texture->texture_raw);
        rendertest.scene->texture->texture_raw += 0.2 * torch::rand_like(rendertest.scene->texture->texture_raw);
    }
    std::cout.clear();

    torch::Tensor target = get_target(rendertest, render_mode, layer);


    int num_failed = 0;
    int tests_size = rendertest.scene->texture->texture_raw.size(0) * rendertest.scene->texture->texture_raw.size(1);

    for (int j = 0; j < rendertest.scene->texture->texture_raw.size(0); ++j)
    {
        for (int i = 0; i < rendertest.scene->texture->texture_raw.size(1); ++i)
        {
            {
                auto param = rendertest.scene->texture->texture_raw;  //.squeeze();
                // std::cout << TensorInfo(rendertest.scene->texture->texture_raw) << std::endl;

                int result = rendertest.EvaluateParam(param, {j, i}, target, render_mode, layer);
                if (!result) num_failed++;
                if (result == 2) tests_size--;
            }
        }
    }
    auto ret = std::make_pair(num_failed, int(tests_size));
    return ret;
}

TEST(RendererTest, TextureGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)


    for (int layer = 0; layer < 4; ++layer)
    {
        int all      = 0;
        int num_fail = 0;
        int tests    = 0;
        std::cout << "Test texture grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testTextureGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);
            num_fail += failed;
            all += all_t;
        }
        std::cout << "Texture Gradient Test (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}

std::pair<int, int> testBackgroundGrad(PointRendererCache::RenderMode render_mode, int layer)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_intrinsics = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        torch::manual_seed(std::rand());
        rendertest.scene->texture->background_color_raw +=
            0.2 * (torch::rand_like(rendertest.scene->texture->background_color_raw) * 2 - 1);
    }
    std::cout.clear();
    // std::cout << "RANDOMED: " << rendertest.scene->texture->background_color_raw << std::endl;

    torch::Tensor target = get_target(rendertest, render_mode, layer);

    int num_failed = 0;
    int tests_size = rendertest.scene->texture->background_color_raw.size(0);

    for (int i = 0; i < tests_size; ++i)
    {
        auto param = rendertest.scene->texture->background_color_raw;  //.squeeze();
        int result = rendertest.EvaluateParam(param, {i}, target, render_mode, layer, 1e-4, false);
        if (!result) num_failed++;
        if (result == 2) tests_size--;
    }

    auto ret = std::make_pair(num_failed, tests_size);
    return ret;
}

TEST(RendererTest, BackgroundGradientTest)
{
    // for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)

    int tests = 0;


    for (int layer = 0; layer < 4; ++layer)
    {
        int all      = 0;
        int num_fail = 0;
        std::cout << "Test background grad with mode " << RENDER_MODE_SELECT << std::endl;
        for (tests = 0; tests < 100; ++tests)
        {
            int failed, all_t;
            std::tie(failed, all_t) = testBackgroundGrad(PointRendererCache::RenderMode(RENDER_MODE_SELECT), layer);

            num_fail += failed;
            all += all_t;
        }
        std::cout << "Background Gradient Test (layer " << layer << "): " << num_fail << "/" << all << " failed."
                  << std::endl;
    }
}

/*

void testConfidence(PointRendererCache::RenderMode render_mode)
{
    std::cout.setstate(std::ios_base::failbit);
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_texture = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        // rendertest.scene->texture->confidence_raw.set_(
        //     torch::full_like(rendertest.scene->texture->confidence_raw, -.05f));
        rendertest.scene->texture->confidence_raw *= 0.001f;
    }
    using TexOpt   = torch::optim::SGDOptions;
    using TexOptim = torch::optim::SGD;
    std::vector<torch::optim::OptimizerParamGroup> g;
    std::cout.clear();
    {
        auto opt_t = std::make_unique<TexOpt>(200);
        std::vector<torch::Tensor> ts;
        ts.push_back(rendertest.scene->texture->confidence_raw);
        g.emplace_back(ts, std::move(opt_t));
    }
    auto texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
    texture_optimizer->zero_grad();

    const int epochs = 100;
    torch::Tensor target;
    auto cpu_img = rendertest.cpu_img.getImageView();
    if (render_mode >= 3) cpu_img = rendertest.cpu_img_bilinear.getImageView();

    target = ImageViewToTensor(cpu_img).to(torch::kFloat32).clone().cuda().contiguous().unsqueeze(0);
    for (int epoch_id = 0; epoch_id < epochs; ++epoch_id)
    {
        auto neural_img = rendertest.render(render_mode, epoch_id);
        auto loss       = torch::mse_loss(neural_img, target);
        std::cout << TensorInfo(loss) << std::endl;
        loss.backward();

        std::cout << TensorInfo(rendertest.scene->texture->confidence_raw.mutable_grad()) << std::endl;

        if (epoch_id % 10 == 0)
        {
            auto img_gpu = TensorToImage<ucvec3>(neural_img);
            img_gpu.save("optim_conf_" + std::to_string(epoch_id) + ".png");
            auto img_gt = TensorToImage<ucvec3>(target);
            img_gt.save("optim_conf_" + std::to_string(epoch_id) + "_gt.png");
        }
        texture_optimizer->step();
        texture_optimizer->zero_grad();
    }
    auto neural_img = rendertest.render(render_mode, epochs, "conf" + std::to_string(render_mode));

    const float EPSILON = 0.01;
    bool all_eq         = torch::all(torch::abs(target - neural_img) < EPSILON).item<bool>();
    if (!all_eq)
    {
        std::cout << TensorInfo(neural_img) << std::endl;
        std::cout << TensorInfo(target) << std::endl;
    }
    CHECK_EQ(all_eq, true);
}



TEST(RendererTest, ConfidenceOptimTest)
{
    //    for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    int i = 3;
    {
        std::cout << "Test confidence with mode " << i << std::endl;
        testConfidence(PointRendererCache::RenderMode(i));
    }
    exit(1);
}


void testTextureOptim(PointRendererCache::RenderMode render_mode)
{
    auto params = default_params();
    fixall(params);
    params->optimizer_params.fix_texture = false;
    RendererTesting rendertest(params, 32, 32);
    {
        torch::NoGradGuard ngg;
        rendertest.scene->texture->texture.set_(torch::full_like(rendertest.scene->texture->texture, 0.4f));
    }
    using TexOpt   = torch::optim::AdamOptions;
    using TexOptim = torch::optim::MyAdam;
    std::vector<torch::optim::OptimizerParamGroup> g;

    {
        auto opt_t = std::make_unique<TexOpt>(0.01);
        std::vector<torch::Tensor> ts;
        ts.push_back(rendertest.scene->texture->texture);
        g.emplace_back(ts, std::move(opt_t));
    }
    auto texture_optimizer = std::make_shared<TexOptim>(g, TexOpt(1));
    texture_optimizer->zero_grad();

    const int epochs = 1000;
    auto target      = ImageViewToTensor(rendertest.cpu_img.getImageView())
                      .to(torch::kFloat32)
                      .clone()
                      .cuda()
                      .contiguous()
                      .unsqueeze(0);
    for (int epoch_id = 0; epoch_id < epochs; ++epoch_id)
    {
        auto neural_img = rendertest.render(render_mode, epoch_id);
        auto loss       = torch::l1_loss(neural_img, target);
        loss.backward();
        texture_optimizer->step();
        texture_optimizer->zero_grad();
    }
    auto neural_img = rendertest.render(render_mode, epochs, "texture" + std::to_string(render_mode));
    // std::cout << TensorInfo(neural_img) << std::endl;
    // std::cout << TensorInfo(target) << std::endl;
    const float EPSILON = 0.01;
    CHECK_EQ(torch::all(torch::abs(target - neural_img) < EPSILON).item<bool>(), true);
}

TEST(RendererTest, TextureOptimTestTest)
{
    for (int i = 0; i < PointRendererCache::RenderMode::SIZE; ++i)
    {
        testTextureOptim(PointRendererCache::RenderMode(i));
    }
}*/

/*

Test conf grad with mode 3
Conf grad test (layer 0): 0/0 failed.
Test conf grad with mode 3
Conf grad test (layer 1): 0/0 failed.
Test conf grad with mode 3
Conf grad test (layer 2): 0/0 failed.
Test conf grad with mode 3
Conf grad test (layer 3): 0/0 failed.
Test position grad with mode 3
0.00139291 _____ 0: inf;; 0.00139291
0.00303102 _____ 0.00479275: 0.632417;; 0.00176173
0.00737447 _____ 0.00566853: 1.30095;; 0.00170594
0.00558168 _____ 0.00678426: 0.822739;; 0.00120259
0.00447864 _____ 0.00501002: 0.893938;; 0.000531375
0.00613276 _____ 0.00586257: 1.04609;; 0.000270189
-1.45519e-07 _____ 1.16415e-10: -1250;; 1.45636e-07
0.00525222 _____ 0.00678085: 0.774567;; 0.00152863
-1.45519e-07 _____ 1.16415e-10: -1250;; 1.45636e-07
0.00894608 _____ 0.010347: 0.86461;; 0.00140087
0.00230619 _____ 0: inf;; 0.00230619
-2.91038e-07 _____ 1.16415e-10: -2500;; 2.91155e-07
1.45519e-07 _____ -1.16415e-10: -1250;; 1.45636e-07
Position Gradient Test (layer 0): 9/900 failed.
Test position grad with mode 3
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
2.32831e-06 _____ -2.32831e-10: -10000;; 2.32854e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
2.32831e-06 _____ -2.32831e-10: -10000;; 2.32854e-06
-2.32831e-06 _____ 2.32831e-10: -10000;; 2.32854e-06
Position Gradient Test (layer 1): 0/900 failed.
Test position grad with mode 3
-4.65661e-06 _____ 9.31323e-10: -5000;; 4.65754e-06
-4.65661e-06 _____ 4.65661e-10: -10000;; 4.65708e-06
-4.65661e-06 _____ 9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -4.65661e-10: -10000;; 4.65708e-06
-4.65661e-06 _____ 4.65661e-10: -10000;; 4.65708e-06
-9.31323e-06 _____ 9.31323e-10: -10000;; 9.31416e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
9.31323e-06 _____ -2.32831e-10: -40000;; 9.31346e-06
9.31323e-06 _____ -2.32831e-10: -40000;; 9.31346e-06
4.65661e-06 _____ -4.65661e-10: -10000;; 4.65708e-06
-9.31323e-06 _____ 9.31323e-10: -10000;; 9.31416e-06
1.39698e-05 _____ -9.31323e-10: -15000;; 1.39708e-05
Position Gradient Test (layer 2): 0/900 failed.
Test position grad with mode 3
Position Gradient Test (layer 3): 0/900 failed.
Test intrinsics grad with mode 3
0.00304455 _____ 0.00292758: 1.03996;; 0.000116976
-0.00313565 _____ -0.00298028: 1.05213;; 0.00015537
-0.00303204 _____ -0.00292067: 1.03813;; 0.000111365
0.0135975 _____ 0.0109705: 1.23945;; 0.00262693
0.00372791 _____ 0.00138519: 2.69125;; 0.00234271
0.000460277 _____ 0.000318217: 1.44643;; 0.000142061
-9.03674e-05 _____ -0.000228742: 0.395063;; 0.000138375
3.0559e-06 _____ -8.14907e-10: -3750;; 3.05672e-06
-3.0559e-06 _____ 1.16415e-10: -26250;; 3.05602e-06
-7.27596e-08 _____ 6.98492e-10: -104.167;; 7.34581e-08
-3.20142e-06 _____ 4.65661e-10: -6875;; 3.20189e-06
-0.0158382 _____ -0.01886: 0.839774;; 0.00302186
-6.25732e-06 _____ 3.49246e-10: -17916.7;; 6.25767e-06
3.49246e-06 _____ -1.04774e-09: -3333.33;; 3.49351e-06
-0.0095705 _____ -0.00744309: 1.28582;; 0.00212741
-0.00933185 _____ -0.00729423: 1.27935;; 0.00203762
-0.00909073 _____ -0.00714835: 1.27172;; 0.00194238
0.00934859 _____ 0.00730341: 1.28003;; 0.00204518
0.00911066 _____ 0.00715734: 1.27291;; 0.00195332
0.00885848 _____ 0.00701419: 1.26294;; 0.00184428
0.0254977 _____ 0.0332891: 0.765948;; 0.00779138
0.0154227 _____ 0.02333: 0.661067;; 0.0079073
-3.20142e-06 _____ 4.31062e-06: -0.742682;; 7.51204e-06
3.0559e-06 _____ -3.49246e-10: -8750;; 3.05625e-06
2.91038e-06 _____ -1.16415e-09: -2500;; 2.91155e-06
-6.33008e-06 _____ 2.91038e-11: -217500;; 6.33011e-06
-6.36646e-06 _____ 4.65661e-10: -13671.9;; 6.36693e-06
-1.45519e-07 _____ 4.65661e-10: -312.5;; 1.45985e-07
-7.27596e-08 _____ 4.65661e-10: -156.25;; 7.32252e-08
-7.27596e-08 _____ 2.32831e-10: -312.5;; 7.29924e-08
1.45519e-07 _____ -4.65661e-10: -312.5;; 1.45985e-07
3.20142e-06 _____ -2.32831e-10: -13750;; 3.20165e-06
-3.12866e-06 _____ 2.32831e-10: -13437.5;; 3.12889e-06
3.12866e-06 _____ -2.32831e-10: -13437.5;; 3.12889e-06
1.45519e-07 _____ -4.65661e-10: -312.5;; 1.45985e-07
2.18279e-07 _____ -2.32831e-10: -937.5;; 2.18512e-07
-0.00446031 _____ -0.00407524: 1.09449;; 0.00038507
-0.00433822 _____ -0.00399373: 1.08626;; 0.000344485
-0.00421744 _____ -0.00391386: 1.07757;; 0.000303579
0.00432832 _____ 0.00398719: 1.08556;; 0.000341127
0.00420085 _____ 0.00390745: 1.07509;; 0.000293396
0.00409462 _____ 0.0038293: 1.06929;; 0.000265315
0.0318227 _____ 0.0345241: 0.921755;; 0.00270135
0.041076 _____ 0.0440216: 0.933087;; 0.00294562
9.24047e-06 _____ -9.31323e-10: -9921.88;; 9.2414e-06
-2.98314e-06 _____ 2.32831e-10: -12812.5;; 2.98338e-06
-3.0559e-06 _____ 4.65661e-10: -6562.5;; 3.05637e-06
2.91038e-07 _____ -2.32831e-10: -1250;; 2.91271e-07
1.45519e-07 _____ -4.65661e-10: -312.5;; 1.45985e-07
0.00120897 _____ -9.31323e-10: -1.29812e+06;; 0.00120897
-6.18456e-06 _____ 2.32831e-10: -26562.5;; 6.1848e-06
1.45519e-07 _____ -4.65661e-10: -312.5;; 1.45985e-07
3.0559e-06 _____ -4.65661e-10: -6562.5;; 3.05637e-06
-3.20142e-06 _____ 4.65661e-10: -6875;; 3.20189e-06
7.27596e-08 _____ -2.32831e-10: -312.5;; 7.29924e-08
-3.0559e-06 _____ 9.31323e-10: -3281.25;; 3.05683e-06
3.12866e-06 _____ -4.65661e-10: -6718.75;; 3.12913e-06
-3.0559e-06 _____ 6.98492e-10: -4375;; 3.0566e-06
-3.12866e-06 _____ 6.98492e-10: -4479.17;; 3.12936e-06
-0.000227728 _____ 0.000377693: -0.602945;; 0.000605422
-0.000198424 _____ 0.000370138: -0.536082;; 0.000568563
-0.000179352 _____ 0.000362736: -0.494443;; 0.000542089
0.000224809 _____ -0.000377248: -0.595918;; 0.000602057
0.000199589 _____ -0.000369703: -0.539862;; 0.000569292
0.000182927 _____ -0.000362309: -0.504892;; 0.000545235
-0.0375581 _____ -0.0424004: 0.885795;; 0.00484231
9.31323e-06 _____ -4.65661e-10: -20000;; 9.31369e-06
-6.25732e-06 _____ 2.32831e-10: -26875;; 6.25756e-06
-0.000101136 _____ -0.000231504: 0.436865;; 0.000130368
-0.000456203 _____ -0.000323686: 1.4094;; 0.000132517
0.00196422 _____ 0.00137765: 1.42578;; 0.00058657
0.00493164 _____ 0.00279741: 1.76293;; 0.00213423
Intrinsics Gradient Test (layer 0): 36/1300 failed.
Test intrinsics grad with mode 3
1.16415e-06 _____ -5.82077e-11: -20000;; 1.16421e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
2.32831e-06 _____ -9.31323e-10: -2500;; 2.32924e-06
2.32831e-06 _____ -3.49246e-10: -6666.67;; 2.32866e-06
3.49246e-06 _____ -1.62981e-09: -2142.86;; 3.49409e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
1.16415e-06 _____ -1.39698e-09: -833.333;; 1.16555e-06
1.16415e-06 _____ -1.86265e-09: -625;; 1.16602e-06
-1.16415e-06 _____ 9.31323e-10: -1250;; 1.16508e-06
1.16415e-06 _____ -1.45519e-11: -80000;; 1.16417e-06
-2.32831e-06 _____ 1.45519e-11: -160000;; 2.32832e-06
2.32831e-06 _____ -2.79397e-09: -833.333;; 2.3311e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
1.16415e-06 _____ -2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 6.98492e-10: -1666.67;; 1.16485e-06
-1.16415e-06 _____ 1.45519e-11: -80000;; 1.16417e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
-2.32831e-06 _____ 9.31323e-10: -2500;; 2.32924e-06
-1.16415e-06 _____ 9.31323e-10: -1250;; 1.16508e-06
-1.16415e-06 _____ 9.31323e-10: -1250;; 1.16508e-06
1.16415e-06 _____ -1.86265e-09: -625;; 1.16602e-06
-1.16415e-06 _____ 9.31323e-10: -1250;; 1.16508e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 1.39698e-09: -833.333;; 1.16555e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
1.16415e-06 _____ -1.45519e-11: -80000;; 1.16417e-06
1.16415e-06 _____ -2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
-2.32831e-06 _____ 1.45519e-11: -160000;; 2.32832e-06
2.32831e-06 _____ -4.65661e-10: -5000;; 2.32877e-06
-1.16415e-06 _____ 7.27596e-12: -160000;; 1.16416e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
-2.32831e-06 _____ 4.65661e-10: -5000;; 2.32877e-06
1.16415e-06 _____ -2.32831e-10: -5000;; 1.16439e-06
1.16415e-06 _____ -2.32831e-10: -5000;; 1.16439e-06
2.32831e-06 _____ -4.65661e-10: -5000;; 2.32877e-06
-2.32831e-06 _____ 3.49246e-10: -6666.67;; 2.32866e-06
2.32831e-06 _____ -4.65661e-10: -5000;; 2.32877e-06
1.16415e-06 _____ -1.39698e-09: -833.333;; 1.16555e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
-1.16415e-06 _____ 2.79397e-09: -416.667;; 1.16695e-06
1.16415e-06 _____ -2.18279e-11: -53333.3;; 1.16418e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -9.31323e-10: -1250;; 1.16508e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -4.65661e-10: -2500;; 1.16462e-06
Intrinsics Gradient Test (layer 1): 0/1300 failed.
Test intrinsics grad with mode 3
4.65661e-06 _____ -5.82077e-11: -80000;; 4.65667e-06
-9.31323e-06 _____ 1.16415e-09: -8000;; 9.31439e-06
9.31323e-06 _____ -1.86265e-09: -5000;; 9.31509e-06
4.65661e-06 _____ -1.86265e-09: -2500;; 4.65848e-06
-4.65661e-06 _____ 1.86265e-09: -2500;; 4.65848e-06
4.65661e-06 _____ -3.72529e-09: -1250;; 4.66034e-06
4.65661e-06 _____ -2.79397e-09: -1666.67;; 4.65941e-06
4.65661e-06 _____ -2.32831e-10: -20000;; 4.65685e-06
1.39698e-05 _____ -2.91038e-11: -480000;; 1.39699e-05
9.31323e-06 _____ -2.79397e-09: -3333.33;; 9.31602e-06
-9.31323e-06 _____ 5.82077e-11: -160000;; 9.31328e-06
-9.31323e-06 _____ 3.25963e-09: -2857.14;; 9.31649e-06
-9.31323e-06 _____ 7.45058e-09: -1250;; 9.32068e-06
-4.65661e-06 _____ 8.00355e-11: -58181.8;; 4.65669e-06
-4.65661e-06 _____ 5.82077e-11: -80000;; 4.65667e-06
-4.65661e-06 _____ 1.86265e-09: -2500;; 4.65848e-06
-9.31323e-06 _____ 2.18279e-11: -426667;; 9.31325e-06
-9.31323e-06 _____ 4.65661e-10: -20000;; 9.31369e-06
4.65661e-06 _____ -6.98492e-10: -6666.67;; 4.65731e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -2.32831e-09: -2000;; 4.65894e-06
4.65661e-06 _____ -1.86265e-09: -2500;; 4.65848e-06
-4.65661e-06 _____ 9.31323e-10: -5000;; 4.65754e-06
-9.31323e-06 _____ 4.72937e-11: -196923;; 9.31327e-06
-4.65661e-06 _____ 2.32831e-09: -2000;; 4.65894e-06
-9.31323e-06 _____ 4.65661e-10: -20000;; 9.31369e-06
-4.65661e-06 _____ 1.86265e-09: -2500;; 4.65848e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -2.79397e-09: -1666.67;; 4.65941e-06
-4.65661e-06 _____ 6.98492e-10: -6666.67;; 4.65731e-06
-4.65661e-06 _____ 6.98492e-10: -6666.67;; 4.65731e-06
-4.65661e-06 _____ 1.62981e-09: -2857.14;; 4.65824e-06
9.31323e-06 _____ -1.39698e-09: -6666.67;; 9.31462e-06
4.65661e-06 _____ -1.16415e-09: -4000;; 4.65778e-06
-9.31323e-06 _____ 3.72529e-09: -2500;; 9.31695e-06
9.31323e-06 _____ -3.72529e-09: -2500;; 9.31695e-06
9.31323e-06 _____ -1.45519e-11: -640000;; 9.31324e-06
9.31323e-06 _____ -1.16415e-10: -80000;; 9.31334e-06
9.31323e-06 _____ -1.62981e-09: -5714.29;; 9.31486e-06
9.31323e-06 _____ -9.31323e-10: -10000;; 9.31416e-06
-9.31323e-06 _____ 1.39698e-09: -6666.67;; 9.31462e-06
9.31323e-06 _____ -3.72529e-09: -2500;; 9.31695e-06
-4.65661e-06 _____ 1.86265e-09: -2500;; 4.65848e-06
4.65661e-06 _____ -2.79397e-09: -1666.67;; 4.65941e-06
9.31323e-06 _____ -4.65661e-10: -20000;; 9.31369e-06
-4.65661e-06 _____ 9.31323e-10: -5000;; 4.65754e-06
-4.65661e-06 _____ 9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -1.09139e-11: -426667;; 4.65662e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
-4.65661e-06 _____ 1.86265e-09: -2500;; 4.65848e-06
-9.31323e-06 _____ 5.82077e-11: -160000;; 9.31328e-06
-9.31323e-06 _____ 9.31323e-10: -10000;; 9.31416e-06
-9.31323e-06 _____ 4.65661e-10: -20000;; 9.31369e-06
9.31323e-06 _____ -1.86265e-09: -5000;; 9.31509e-06
-9.31323e-06 _____ 1.86265e-09: -5000;; 9.31509e-06
-9.31323e-06 _____ 1.33878e-09: -6956.52;; 9.31456e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
-9.31323e-06 _____ 1.28057e-09: -7272.73;; 9.31451e-06
4.65661e-06 _____ -9.31323e-10: -5000;; 4.65754e-06
4.65661e-06 _____ -2.09548e-09: -2222.22;; 4.65871e-06
-4.65661e-06 _____ 1.39698e-09: -3333.33;; 4.65801e-06
4.65661e-06 _____ -6.98492e-10: -6666.67;; 4.65731e-06
4.65661e-06 _____ -6.98492e-10: -6666.67;; 4.65731e-06
4.65661e-06 _____ -6.98492e-10: -6666.67;; 4.65731e-06
-4.65661e-06 _____ 3.72529e-09: -1250;; 4.66034e-06
9.31323e-06 _____ -1.16415e-09: -8000;; 9.31439e-06
9.31323e-06 _____ -2.32831e-09: -4000;; 9.31555e-06
4.65661e-06 _____ -3.49246e-10: -13333.3;; 4.65696e-06
-4.65661e-06 _____ 2.09548e-09: -2222.22;; 4.65871e-06
-9.31323e-06 _____ 3.72529e-09: -2500;; 9.31695e-06
4.65661e-06 _____ -3.72529e-09: -1250;; 4.66034e-06
9.31323e-06 _____ -9.31323e-10: -10000;; 9.31416e-06
9.31323e-06 _____ -4.19095e-09: -2222.22;; 9.31742e-06
-9.31323e-06 _____ 4.65661e-10: -20000;; 9.31369e-06
-9.31323e-06 _____ 6.98492e-10: -13333.3;; 9.31392e-06
9.31323e-06 _____ -2.32831e-10: -40000;; 9.31346e-06
4.65661e-06 _____ -4.65661e-10: -10000;; 4.65708e-06
-9.31323e-06 _____ 5.82077e-11: -160000;; 9.31328e-06
-9.31323e-06 _____ 1.74623e-09: -5333.33;; 9.31497e-06
9.31323e-06 _____ -2.21189e-09: -4210.53;; 9.31544e-06
-4.65661e-06 _____ 1.62981e-09: -2857.14;; 4.65824e-06
4.65661e-06 _____ -2.32831e-10: -20000;; 4.65685e-06
Intrinsics Gradient Test (layer 2): 0/1300 failed.
Test intrinsics grad with mode 3
-0.0510737 _____ -0.0509659: 1.00211;; 0.000107788
Intrinsics Gradient Test (layer 3): 1/1300 failed.
Test pose grad with mode 3
-0.0232644 _____ -0.0224131: 1.03798;; 0.000851341
0.0156576 _____ 0.0131241: 1.19304;; 0.00253346
0.00481115 _____ 0.00741827: 0.648555;; 0.00260711
0.0272325 _____ 0.0254041: 1.07197;; 0.00182835
-0.0371979 _____ -0.0353792: 1.0514;; 0.00181866
0.015056 _____ 0.0176584: 0.852624;; 0.00260243
-0.00971733 _____ -0.0102479: 0.94823;; 0.00053053
-0.0109618 _____ -0.0103462: 1.0595;; 0.000615596
0.0159336 _____ 0.0152463: 1.04508;; 0.000687365
-0.014705 _____ -0.0152062: 0.967038;; 0.000501228
0.00563508 _____ 0.00712696: 0.790671;; 0.00149188
-0.0251306 _____ -0.0277856: 0.904446;; 0.00265501
0.0172938 _____ 0.0170509: 1.01424;; 0.00024286
-0.0123237 _____ -0.0120638: 1.02154;; 0.000259904
-0.030458 _____ -0.0331404: 0.919061;; 0.00268236
0.00753964 _____ 0.00781936: 0.964227;; 0.000279726
-0.0102403 _____ -0.0096755: 1.05838;; 0.000564828
0.00608386 _____ 0.0079507: 0.765198;; 0.00186684
0.00942266 _____ 0.00528331: 1.78348;; 0.00413935
0.00841086 _____ 0.0122769: 0.685097;; 0.00386603
-0.0183599 _____ -0.0168191: 1.09161;; 0.00154081
0.00388332 _____ 0.0071648: 0.542001;; 0.00328147
0.00101296 _____ -0.00229028: -0.442286;; 0.00330324
-0.0233855 _____ -0.0218331: 1.0711;; 0.00155241
0.0156522 _____ 0.017833: 0.877707;; 0.00218086
-0.00659988 _____ -0.011129: 0.593032;; 0.00452917
0.00957225 _____ 0.01062: 0.901346;; 0.0010477
0.00216023 _____ 0.00768426: 0.281124;; 0.00552403
0.0200287 _____ 0.0210912: 0.949623;; 0.00106251
-0.00555359 _____ -0.0104786: 0.529992;; 0.00492504
Pose Gradient Test (layer 0): 30/600 failed.
Test pose grad with mode 3
-2.32831e-06 _____ 5.23869e-10: -4444.44;; 2.32883e-06
-1.16415e-06 _____ 3.00133e-11: -38787.9;; 1.16418e-06
-1.16415e-06 _____ 5.78439e-10: -2012.58;; 1.16473e-06
-1.16415e-06 _____ 4.65661e-10: -2500;; 1.16462e-06
1.16415e-06 _____ -7.27596e-12: -160000;; 1.16416e-06
1.16415e-06 _____ -1.15051e-10: -10118.6;; 1.16427e-06
2.32831e-06 _____ -2.38288e-10: -9770.99;; 2.32854e-06
2.32831e-06 _____ -5.82077e-10: -4000;; 2.32889e-06
1.16415e-06 _____ -3.76531e-10: -3091.79;; 1.16453e-06
-1.16415e-06 _____ 4.03816e-10: -2882.88;; 1.16456e-06
-1.16415e-06 _____ 5.85715e-10: -1987.58;; 1.16474e-06
-1.16415e-06 _____ 1.45519e-09: -800;; 1.16561e-06
-2.32831e-06 _____ 5.82077e-11: -40000;; 2.32836e-06
2.32831e-06 _____ -2.61934e-10: -8888.89;; 2.32857e-06
-1.16415e-06 _____ 3.49246e-10: -3333.33;; 1.1645e-06
-2.32831e-06 _____ 9.31323e-10: -2500;; 2.32924e-06
1.16415e-06 _____ -5.45697e-11: -21333.3;; 1.16421e-06
1.16415e-06 _____ -4.1473e-10: -2807.02;; 1.16457e-06
-1.16415e-06 _____ 6.40284e-10: -1818.18;; 1.16479e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 3.45608e-11: -33684.2;; 1.16419e-06
-1.16415e-06 _____ 1.16415e-09: -1000;; 1.16532e-06
1.16415e-06 _____ -3.49246e-10: -3333.33;; 1.1645e-06
-1.16415e-06 _____ 1.15506e-10: -10078.7;; 1.16427e-06
-1.16415e-06 _____ 1.29148e-10: -9014.08;; 1.16428e-06
1.16415e-06 _____ -9.02219e-10: -1290.32;; 1.16506e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 1.74623e-10: -6666.67;; 1.16433e-06
-1.16415e-06 _____ 2.03727e-10: -5714.29;; 1.16436e-06
0.0146055 _____ 0.0111125: 1.31432;; 0.00349293
0.0219455 _____ 0.0184943: 1.1866;; 0.00345112
2.32831e-06 _____ -1.23691e-10: -18823.5;; 2.32843e-06
1.16415e-06 _____ -2.32831e-10: -5000;; 1.16439e-06
-1.16415e-06 _____ 2.32831e-10: -5000;; 1.16439e-06
2.32831e-06 _____ -2.72848e-10: -8533.33;; 2.32858e-06
-1.16415e-06 _____ 3.87445e-10: -3004.69;; 1.16454e-06
1.16415e-06 _____ -3.4197e-10: -3404.26;; 1.1645e-06
1.16415e-06 _____ -8.11269e-10: -1434.98;; 1.16496e-06
-2.32831e-06 _____ 1.16415e-10: -20000;; 2.32842e-06
-1.16415e-06 _____ 9.09495e-13: -1.28e+06;; 1.16415e-06
Pose Gradient Test (layer 1): 2/600 failed.
Test pose grad with mode 3
9.31323e-06 _____ -3.27418e-10: -28444.4;; 9.31355e-06
-1.39698e-05 _____ 1.27329e-09: -10971.4;; 1.39711e-05
-9.31323e-06 _____ 1.16415e-09: -8000;; 9.31439e-06
0.00674278 _____ 0.0072015: 0.936302;; 0.000458722
-0.00675209 _____ -0.00720086: 0.937678;; 0.000448775
-9.31323e-06 _____ 2.73576e-09: -3404.26;; 9.31596e-06
-9.31323e-06 _____ 1.72804e-09: -5389.47;; 9.31495e-06
1.39698e-05 _____ -4.25644e-10: -32820.5;; 1.39703e-05
9.31323e-06 _____ -6.98492e-10: -13333.3;; 9.31392e-06
-9.31323e-06 _____ 1.62981e-09: -5714.29;; 9.31486e-06
-9.31323e-06 _____ 9.27685e-10: -10039.2;; 9.31415e-06
9.31323e-06 _____ -1.16415e-09: -8000;; 9.31439e-06
9.31323e-06 _____ -4.62023e-10: -20157.5;; 9.31369e-06
9.31323e-06 _____ -1.39698e-09: -6666.67;; 9.31462e-06
9.31323e-06 _____ -5.09317e-10: -18285.7;; 9.31374e-06
-9.31323e-06 _____ 6.8394e-10: -13617;; 9.31391e-06
1.39698e-05 _____ -1.16415e-10: -120000;; 1.397e-05
9.31323e-06 _____ -1.86265e-09: -5000;; 9.31509e-06
-9.31323e-06 _____ 3.63798e-12: -2.56e+06;; 9.31323e-06
-9.31323e-06 _____ 3.55067e-09: -2622.95;; 9.31678e-06
9.31323e-06 _____ -1.82808e-10: -50945.3;; 9.31341e-06
9.31323e-06 _____ -2.32831e-10: -40000;; 9.31346e-06
9.31323e-06 _____ -1.63709e-09: -5688.89;; 9.31486e-06
-9.31323e-06 _____ 1.47702e-09: -6305.42;; 9.3147e-06
9.31323e-06 _____ -6.98492e-10: -13333.3;; 9.31392e-06
9.31323e-06 _____ -5.38421e-10: -17297.3;; 9.31376e-06
9.31323e-06 _____ -3.08319e-10: -30206.5;; 9.31353e-06
9.31323e-06 _____ -5.91172e-12: -1.57538e+06;; 9.31323e-06
9.31323e-06 _____ -9.24047e-10: -10078.7;; 9.31415e-06
-9.31323e-06 _____ 2.32831e-10: -40000;; 9.31346e-06
-9.31323e-06 _____ 2.25555e-10: -41290.3;; 9.31345e-06
-9.31323e-06 _____ 6.98492e-10: -13333.3;; 9.31392e-06
9.31323e-06 _____ -6.71889e-10: -13861.3;; 9.3139e-06
-9.31323e-06 _____ 5.85715e-10: -15900.6;; 9.31381e-06
9.31323e-06 _____ -8.14907e-10: -11428.6;; 9.31404e-06
-9.31323e-06 _____ 9.60426e-10: -9696.97;; 9.31419e-06
1.39698e-05 _____ -1.16415e-09: -12000;; 1.3971e-05
-9.31323e-06 _____ 2.6921e-10: -34594.6;; 9.31349e-06
9.31323e-06 _____ -3.14321e-09: -2962.96;; 9.31637e-06
-9.31323e-06 _____ 4.76575e-10: -19542;; 9.3137e-06
-9.31323e-06 _____ 2.80124e-10: -33246.8;; 9.31351e-06
9.31323e-06 _____ -9.31323e-10: -10000;; 9.31416e-06
Pose Gradient Test (layer 2): 2/600 failed.
Test pose grad with mode 3
-0.0236928 _____ -0.023585: 1.00457;; 0.000107807
-0.0231713 _____ -0.0215114: 1.07717;; 0.00165995
0.0506267 _____ 0.0488513: 1.03634;; 0.00177543
-0.0503287 _____ -0.0501164: 1.00424;; 0.000212299
Pose Gradient Test (layer 3): 4/600 failed.
Test texture grad with mode 3
0.000100699 _____ 0.000323342: 0.311432;; 0.000222643
Texture Gradient Test (layer 0): 1/900 failed.
Test texture grad with mode 3
Texture Gradient Test (layer 1): 0/900 failed.
Test texture grad with mode 3
0.0044005 _____ 0.00408058: 1.0784;; 0.00031992
0.000516884 _____ 0.00517347: 0.0999104;; 0.00465659
Texture Gradient Test (layer 2): 2/900 failed.
Test texture grad with mode 3
Texture Gradient Test (layer 3): 0/900 failed.
Test background grad with mode 3
-0.332175 _____ -0.332364: 0.999431;; 0.00018926
-0.332221 _____ -0.332364: 0.999571;; 0.000142694
-0.332221 _____ -0.332364: 0.999571;; 0.000142694
Background Gradient Test (layer 0): 3/300 failed.
Test background grad with mode 3
-0.323802 _____ -0.323911: 0.999665;; 0.000108361
-0.328831 _____ -0.328718: 1.00034;; 0.000113308
-0.328422 _____ -0.328304: 1.00036;; 0.000117421
0.329614 _____ 0.329453: 1.00049;; 0.000160396
-0.323951 _____ -0.324066: 0.999646;; 0.000114679
-0.323951 _____ -0.324066: 0.999646;; 0.000114679
Background Gradient Test (layer 1): 6/300 failed.
Test background grad with mode 3
0.317916 _____ 0.317813: 1.00033;; 0.000103414
0.317916 _____ 0.317813: 1.00033;; 0.000103414
Background Gradient Test (layer 2): 2/300 failed.
Test background grad with mode 3
0.278801 _____ 0.278681: 1.00043;; 0.000119954
-0.273511 _____ -0.27339: 1.00044;; 0.000120729
-0.161827 _____ -0.161928: 0.999375;; 0.000101134
0.278801 _____ 0.278681: 1.00043;; 0.000119954
0.278801 _____ 0.278681: 1.00043;; 0.000119954
0.278801 _____ 0.278681: 1.00043;; 0.000119954
-0.273511 _____ -0.27339: 1.00044;; 0.000120729
0.278503 _____ 0.278681: 0.999361;; 0.000178069
0.278801 _____ 0.278681: 1.00043;; 0.000119954
-0.16205 _____ -0.161928: 1.00076;; 0.000122383
-0.273511 _____ -0.27339: 1.00044;; 0.000120729
-0.273287 _____ -0.27339: 0.999624;; 0.000102788
0.0747293 _____ 0.278681: 0.268154;; 0.203951
0.278801 _____ 0.278681: 1.00043;; 0.000119954
0.278801 _____ 0.278681: 1.00043;; 0.000119954
Background Gradient Test (layer 3): 15/300 failed.
Process finished with exit code 0


 */