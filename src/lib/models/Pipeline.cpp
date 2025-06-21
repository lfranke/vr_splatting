/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Pipeline.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include "gaussian/render_utils.cuh"
CUDA::CudaTimerSystem* timer_for_nets;
#include "../opengl/EyeTracking.h"


NeuralPipeline::NeuralPipeline(std::shared_ptr<CombinedParams> _params) : params(_params)
{
    params->Check();
    if (params->net_params.network_version == "MultiScaleUnet2d")
    {
        // render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2d>>(params->net_params);
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2d>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSlim")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dSlim>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dUltraSlim")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dUltraSlim>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnly")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnly>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnlySmall")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnlySmall>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dDecOnlySmallFixed")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dDecOnlySmallFixed>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonics")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonics>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsFixed")
    {
        render_network =
            std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonicsFixed>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiSphericalHarmonics")
    {
        render_network = std::make_shared<DerivedRenderNet<MultiSphericalHarmonics>>(params->net_params);
    }
    else if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsInL2Fixed")
    {
        render_network =
            std::make_shared<DerivedRenderNet<MultiScaleUnet2dSphericalHarmonicsInL2Fixed>>(params->net_params);
    }
    else
    {
        SAIGA_ASSERT(false);
    }
    render_module = PointRenderModule(params);

    if (!params->optimizer_params.fix_dynamic_refinement)
    {
        // refinement_module = RefinementNet(params->net_params);
        dynamic_refinement_module = DynamicRefinementMLP();
    }
    if (params->train_params.checkpoint_directory != "") LoadCheckpoint(params->train_params.checkpoint_directory);

    if (params->optimizer_params.network_checkpoint_directory != "")
    {
        std::cout << "Resuming network from " << params->optimizer_params.network_checkpoint_directory << std::endl;
        LoadCheckpoint(params->optimizer_params.network_checkpoint_directory);
    }
    // PrintModelParamsCompact(render_network.get());

    if (params->net_params.half_float)
    {
        render_network->to(torch::kFloat16);
    }
    else
    {
        render_network->to(torch::kFloat32);
    }
    std::cout << "RENDER NETWORK: NUM PARAMETERS: " << render_network->parameters().size() << std::endl;
    // render_network->PrintModelParamsCompact();
    {
        size_t sum = 0;

        for (auto& t : render_network->parameters())
        {
            size_t local_sum = 1;
            for (auto i : t.sizes())
            {
                local_sum *= i;
            }
            sum += local_sum;
        }
        std::cout << "Total Model Params: " << sum << std::endl;
    }
    render_network->train();
    render_network->to(device);
    if (!params->optimizer_params.fix_dynamic_refinement)
    {
        dynamic_refinement_module->train();
        dynamic_refinement_module->to(device);
    }


    if (params->pipeline_params.train)
    {
        // loss_vgg =
        // std::make_shared<Saiga::PretrainedVGG19Loss>(Saiga::PretrainedVGG19Loss("loss/traced_caffe_vgg.pt"));

        // loss_vgg =
        //     std::make_shared<Saiga::PretrainedVGG19Loss>(Saiga::PretrainedVGG19Loss("loss/traced_caffe_vgg_optim.pt"));

        std::cout << "Loading VGG from: " << params->train_params.vgg_path << std::endl;
        loss_vgg =
            std::make_shared<Saiga::PretrainedVGG19Loss>(Saiga::PretrainedVGG19Loss(params->train_params.vgg_path));

        // loss_vgg = std::make_shared<Saiga::PretrainedVGGLpipsLoss>(
        //     Saiga::PretrainedVGGLpipsLoss("loss/frozen_optim_vgg_lpips.pt"));

        //
        //   loss_vgg =
        //   std::make_shared<Saiga::PretrainedVGG19Loss>(Saiga::PretrainedVGG19Loss("loss/traced_vgg16.pt"));
        loss_vgg->eval();
        loss_vgg->to(device);


        loss_lpips.module.eval();
        loss_lpips.module.to(device);

        loss_ssim->eval();
        loss_ssim->to(device);

        if (!params->optimizer_params.fix_render_network)
        {
            // In half precision the default eps of 1e-8 is rounded to 0
            double adam_eps = params->net_params.half_float ? 1e-4 : 1e-8;
            // if (params->optimizer_params.use_myadam_everywhere)
            //{
            //     using TexOpt   = torch::optim::MyAdamOptions;
            //     using TexOptim = torch::optim::MyAdam;
            //     std::vector<torch::optim::OptimizerParamGroup> g;
            //     auto opt_t = std::make_unique<TexOpt>(params->optimizer_params.lr_render_network);
            //     opt_t->eps(adam_eps);
            //     std::vector<torch::Tensor> ts = render_network->parameters();
            //     g.emplace_back(ts, std::move(opt_t));
            //     render_optimizer = std::make_shared<TexOptim>(g,
            //     TexOpt(params->optimizer_params.lr_render_network)); if
            //     (!params->optimizer_params.fix_dynamic_refinement)
            //     {
            //         std::vector<torch::optim::OptimizerParamGroup> g2 =
            //         {dynamic_refinement_module->parameters()}; refinement_optimizer =
            //         std::make_shared<torch::optim::MyAdam>(
            //             g2, torch::optim::MyAdamOptions().lr(params->optimizer_params.lr_dynamic_refinement));
            //     }
            // }
            // else
            {
                std::vector<torch::optim::OptimizerParamGroup> g = {render_network->parameters()};
                render_optimizer                                 = std::make_shared<torch::optim::Adam>(
                    g, torch::optim::AdamOptions().lr(params->optimizer_params.lr_render_network).eps(adam_eps));

                if (!params->optimizer_params.fix_dynamic_refinement)
                {
                    std::vector<torch::optim::OptimizerParamGroup> g2 = {dynamic_refinement_module->parameters()};
                    refinement_optimizer                              = std::make_shared<torch::optim::SGD>(
                        g2, torch::optim::SGDOptions(params->optimizer_params.lr_dynamic_refinement));
                }
            }
        }
        else
        {
            render_network->eval();
        }
    }
}


void evaluateSH(torch::Tensor& positions_in, torch::Tensor pose_in, torch::Tensor& texture_in,
                torch::Tensor& texture_out, int sh_bands = 3)
{
#ifdef USE_TINYCUDANN

    static std::shared_ptr<SphericalHarmonicsEncoding> enc;
    if (!enc)
    {
        enc = std::make_shared<SphericalHarmonicsEncoding>(sh_bands);
        enc->module->to(torch::kCUDA);
    }

    torch::Tensor norm_dir = torch::nn::functional::normalize(positions_in.slice(1, 0, 3) - pose_in,
                                                              torch::nn::functional::NormalizeFuncOptions().dim(1));

    auto sh_enc = enc->forward(norm_dir);

    int feature_out = texture_in.size(0) / (sh_bands * sh_bands);
    auto t_per      = texture_in.permute({1, 0});

    auto enc_repeated = sh_enc.repeat_interleave(feature_out, 1);

    auto tex = (t_per * enc_repeated);

    texture_out = tex.reshape({tex.size(0), feature_out, sh_bands * sh_bands}).sum(-1).permute({1, 0});
#else
    SAIGA_ASSERT("Not compiled with TINYCUDANN");
#endif
}


ForwardResult NeuralPipeline::Forward(NeuralScene& scene, std::vector<NeuralTrainData>& batch,
                                      torch::Tensor global_mask, bool loss_statistics, int current_epoch,
                                      bool keep_image, float fixed_exposure, vec3 fixed_white_balance)
{
    render_module->params = params;

    bool render_gaussian = params->render_params.render_gaussian;
    bool render_trips    = params->render_params.render_trips;
    bool foveation       = params->render_params.render_foveated;

    // render_gaussian = true;
    // render_trips    = true;
    // foveation       = true;

    torch::Tensor x, viewspace_point_tensor, visibility_filter, radii;
    //[batch,3,h,w]
    torch::Tensor neural_images_gs;
    std::vector<torch::Tensor> neural_images;


    // std::cout << fovea_pos.x() << ", " <<  fovea_pos.y() << std::endl;

    if (!params->render_params.viewer_only)
    {
        // TODO IMPLEMENT
        SAIGA_ASSERT(batch.size() == 1);
    }
    else
    {
        //   ImGui::SliderInt("x_fov", &fovea_pos.x(), fovea_rad / 2, batch.front()->img.w - fovea_rad / 2);
        //   ImGui::SliderInt("y_fov", &fovea_pos.y(), fovea_rad / 2, batch.front()->img.h - fovea_rad / 2);
        //    ImGui::SliderInt("fovea_rad", &fovea_rad, 0, batch.front()->img.h);
    }

    if (render_gaussian)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Gaussian", timer_system);
        std::vector<torch::Tensor> gs_img_to_batch;
        for (int i = 0; i < batch.size(); ++i)
        {
            SAIGA_OPTIONAL_TIME_MEASURE(i == 0 ? "Left" : "Right", timer_system);

            auto b_one                = batch[i];
            static bool train_version = false;
            // ImGui::Checkbox("GS Train Version", &train_version);
            if (!params->render_params.viewer_only || train_version ||
                params->render_params.stp_config == "stp_config_org.json")
            {
                auto [image, viewspace_point_tensor_res, visibility_filter_res, radii_res] =
                    render(scene, b_one, scene.gaussian_model, scene.gaussian_background,
                           params->render_params.viewer_only, timer_system);

                viewspace_point_tensor = viewspace_point_tensor_res;
                visibility_filter      = visibility_filter_res;
                radii                  = radii_res;
                if (params->net_params.half_float)
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("half_float", timer_system);
                    image = image.to(torch::kFloat16);
                    // for (auto& b : batch) b->uv = b->uv.to(torch::kFloat16);
                }
                gs_img_to_batch.push_back(image.unsqueeze(0));
            }
            else
            {
                torch::Tensor image;
                if (params->net_params.half_float)
                    image = render_inference<true>(scene, b_one, scene.gaussian_model, scene.gaussian_background,
                                                   params->render_params.viewer_only, timer_system);
                else
                    image = render_inference<false>(scene, b_one, scene.gaussian_model, scene.gaussian_background,
                                                    params->render_params.viewer_only, timer_system);
                gs_img_to_batch.push_back(image.unsqueeze(0));
            }
            // gs_img_to_batch.back().slice(1, 0, 3) *= 0.f;
        }
        neural_images_gs = torch::cat(gs_img_to_batch, 0);
    }
    int fovea_rad = 512;

    std::vector<ivec2> fovea_poses;
    //   else
    if (render_trips)  // render trips
    {
        std::vector<torch::Tensor> masks;
        // if (!scene.baked_for_inference)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Sigmoid on confidence", timer_system);
            float narrowing_fac =
                current_epoch > 0 ? params->points_adding_params.sigmoid_narrowing_factor * current_epoch : 0;
            scene.texture->PrepareConfidence(narrowing_fac, params->net_params.num_input_layers);
        }
        // if (!scene.baked_for_inference)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Prep Tex", timer_system);
            scene.texture->PrepareTexture(params->pipeline_params.non_subzero_texture);
        }
        if (params->pipeline_params.num_spherical_harmonics_bands_per_point > 0)
        {
            // not implemented
            SAIGA_ASSERT(params->train_params.batch_size == 1);
            vec3 pose_cpu = scene.poses->Download(batch.front()->img.image_index).translation().cast<float>();
            torch::Tensor pose =
                torch::from_blob(&pose_cpu, {3}, torch::TensorOptions().dtype(torch::kFloat)).clone().cuda();
            evaluateSH(scene.point_cloud_cuda->t_position, pose, scene.texture->texture_raw, scene.texture->texture,
                       params->pipeline_params.num_spherical_harmonics_bands_per_point);
            // PrintTensorInfo(scene.texture->texture);
        }

        for (int i = 0; i < batch.size(); ++i)
        {
            ivec2 fovea_pos;
            if (params->render_params.viewer_only && batch.size() > 1)
            {
#ifndef HEADLESS
                fovea_pos = EyeTracking::getEyeLookPositionInPixelsForEye(i, batch[i]->img.w, batch[i]->img.h);
#endif
            }
            else
            {
                fovea_pos = batch[i]->pixel_view_pos;  // ivec2(900, 800);
                //    std::cout << ((i == 0) ? "left: " : "right ") << fovea_pos.x() << ", " << fovea_pos.y() <<
                //    std::endl;
                //  ivec2 fovea_pos = ivec2(900, 800);
            }
            int half_fovea_rad = ceil(fovea_rad / 2);
            fovea_pos.x()      = min(max(fovea_pos.x(), half_fovea_rad), batch[i]->img.w - half_fovea_rad);
            fovea_pos.y()      = min(max(fovea_pos.y(), half_fovea_rad), batch[i]->img.h - half_fovea_rad);
            fovea_poses.push_back(fovea_pos);
        }
        if (foveation)
        {
            for (int i = 0; i < batch.size(); ++i)
            {
                batch[i]->img.crop_transform = IntrinsicsPinholef(1, 1, -(fovea_poses[i].x() - fovea_rad / 2),
                                                                  -(fovea_poses[i].y() - fovea_rad / 2), 0);

                batch[i]->img.w = fovea_rad;
                batch[i]->img.h = fovea_rad;
                batch[i]->img.crop_rotation.setZero();
                batch[i]->img.crop_rotation(0, 0) = 1;
                batch[i]->img.crop_rotation(1, 1) = 1;
            }
        }
        {
            torch::Tensor gs_depth_maps;
            //  if (render_gaussian)
            //      gs_depth_maps = neural_images_gs.slice(1, 3, 4).contiguous().to(torch::kFloat);
            //  else
            gs_depth_maps =
                torch::full({(long)batch.size(), 1, batch.front()->img.h, batch.front()->img.w}, 10000.f,at::TensorOptions().device(torch::kCUDA));
            render_module->cache->gs_depth_maps = gs_depth_maps;

            SAIGA_OPTIONAL_TIME_MEASURE("Render", timer_system);
            std::tie(neural_images, masks) = render_module->forward(scene, batch, current_epoch, timer_system);
            // PrintTensorInfo(neural_images[0]);
            SAIGA_ASSERT(neural_images.size() == params->net_params.num_input_layers);
        }

        if (params->net_params.channels_last)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("channels_last", timer_system);
            for (auto& t : neural_images) t = t.to(t.options(), false, false, torch::MemoryFormat::ChannelsLast);
        }

        if (params->net_params.half_float)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("half_float", timer_system);
            for (auto& t : neural_images) t = t.to(torch::kFloat16);
            for (auto& t : masks) t = t.to(torch::kFloat16);
            for (auto& b : batch) b->uv = b->uv.to(torch::kFloat16);
        }

        if (params->pipeline_params.log_texture)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("log_texture", timer_system);

            for (auto& n : neural_images)
            {
                n = torch::exp(n);
            }
        }
    }
    torch::Tensor local_mask;
    torch::Tensor frame_index;
    torch::Tensor full_target;
    torch::Tensor uv;
    torch::Tensor scale;
    torch::Tensor direction;

    if (render_trips && batch.size() > 2)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Stack", timer_system);

        // 2. Stack into batch
        std::vector<torch::Tensor> mask_list;
        std::vector<torch::Tensor> target_list;
        std::vector<torch::Tensor> uv_list;
        std::vector<torch::Tensor> direction_list;
        std::vector<torch::Tensor> scale_list;
        std::vector<torch::Tensor> frame_index_list;
        // multple renderings of the same image crop
        for (int i = 0; i < batch.size(); ++i)
        {
            auto& pd = batch[i];
            SAIGA_ASSERT(pd->uv.dim() == 3);
            if (pd->target.defined()) target_list.push_back((pd->target));
            if (pd->target_mask.defined()) mask_list.push_back((pd->target_mask));
            if (pd->scale.defined()) scale_list.push_back((pd->scale));

            uv_list.push_back((pd->uv));
            if (pd->direction.defined() && !params->pipeline_params.skip_neural_render_network)
            {
                //    auto pose = scene.poses->poses_se3.slice(0, pd->img.image_index, pd->img.image_index + 1);
                //    auto t    = ProjectDirectionsToWS(pd->direction, pose);
                //    direction_list.push_back(t);
            }
            if (pd->camera_index.defined()) frame_index_list.push_back((pd->camera_index));
        }
        if (!target_list.empty()) full_target = torch::stack(target_list);
        if (!mask_list.empty()) local_mask = torch::stack(mask_list);
        if (!scale_list.empty()) scale = torch::stack(scale_list);
        uv = torch::stack(uv_list);
        if (!frame_index_list.empty()) frame_index = torch::stack(frame_index_list);
        if (!direction_list.empty()) direction = torch::stack(direction_list);
    }
    else
    {
        uv = batch.front()->uv.unsqueeze(0);
        if (batch.front()->target_mask.defined()) local_mask = batch.front()->target_mask.unsqueeze(0);
        if (batch.front()->camera_index.defined()) frame_index = batch.front()->camera_index.unsqueeze(0);
        if (batch.front()->target.defined()) full_target = batch.front()->target.unsqueeze(0);
        if (batch.front()->scale.defined()) scale = batch.front()->scale.unsqueeze(0);
    }
    torch::Tensor network_output;
    torch::Tensor gs_centers_t;
    if (render_trips)  // render trips
    {
        if (render_gaussian && params->pipeline_params.add_gs_to_network)
        {
            std::vector<torch::Tensor> combined_gs_np;
            std::vector<torch::Tensor> gs_centers;
            //   PrintTensorInfo(neural_images[0]);

            for (int i = 0; i < batch.size(); ++i)
            {
                torch::Tensor gs_center =
                    neural_images_gs.slice(0, i, i + 1)
                        .slice(1, 0, 3)
                        .slice(2, fovea_poses[i].y() - fovea_rad / 2, fovea_poses[i].y() + fovea_rad / 2)
                        .slice(3, fovea_poses[i].x() - fovea_rad / 2, fovea_poses[i].x() + fovea_rad / 2);

                gs_centers.push_back(gs_center);
                //  gs_center                      = torch::zeros_like(gs_center);
                torch::Tensor neural_rendering = neural_images[0].slice(0, i, i + 1);
                //  PrintTensorInfo(gs_center);
                //  PrintTensorInfo(neural_rendering);
                float epoch_factor = (current_epoch < 10) ? 0.0f : 1.f;
                auto np_and_gs     = torch::cat({neural_rendering, gs_center.clone().detach() * epoch_factor}, 1);
                // PrintTensorInfo(np_and_gs);
                combined_gs_np.push_back(np_and_gs);
            }



            neural_images[0] = torch::cat(combined_gs_np, 0).clone();
            gs_centers_t     = torch::cat(gs_centers, 0).clone();
        }

        /////////////////////////NETWORK
        if (params->pipeline_params.skip_neural_render_network)
        {
            x = neural_images.front();
            SAIGA_ASSERT(x.size(1) == 3);
        }
        else if (params->pipeline_params.skip_neural_render_network_but_add_layers)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Add Layers", timer_system);
            std::vector<double> scale     = {2.0, 2.0};
            const int num_layers_upsample = params->net_params.num_input_layers;
            const int last_used_layer_idx = 0;

            auto up_func = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(scale).mode(torch::kNearest));
            for (int i = num_layers_upsample - 1; i >= last_used_layer_idx; --i)
            {
                if (i != num_layers_upsample - 1)
                {
                    x = up_func->forward(x);
                    x = torch::nn::functional::pad(
                        x, torch::nn::functional::PadFuncOptions(
                               {0, 0, neural_images[i].size(2) - x.size(2), neural_images[i].size(3) - x.size(3)})
                               .mode(torch::kConstant));
                    x = x + neural_images[i];
                }
                else
                    x = neural_images[i];
            }
        }
        else
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Unet", timer_system);
            SAIGA_ASSERT(!neural_images.empty());
            SAIGA_ASSERT(neural_images.size() == params->net_params.num_input_layers);
            SAIGA_ASSERT(neural_images.front().size(1) ==
                         (params->net_params.num_input_channels + (params->pipeline_params.add_gs_to_network ? 3 : 0)));

            if (params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonics" ||
                params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsFixed" ||
                params->net_params.network_version == "MultiScaleUnet2dSphericalHarmonicsInL2Fixed" ||
                params->net_params.network_version == "MultiSphericalHarmonics")
            {
                neural_images.push_back(direction);
            }

            if (timer_system) timer_for_nets = timer_system;

            if (params->render_params.output_background_mask)
                SAIGA_ASSERT(false, "NOT IMPLEMENTED");
            else
                x = render_network->forward(neural_images);

            SAIGA_ASSERT(x.size(1) == params->net_params.num_output_channels);
            SAIGA_ASSERT(x.size(1) == 3);
        }
        network_output = x.clone();
    }
    torch::Tensor target;

    if (!params->render_params.viewer_only)
    {
        // CREATE TARGET
        if (render_trips && !render_gaussian)
        {
            if (neural_images.size() > 0 &&
                (x.size(2) != neural_images.front().size(2) || x.size(3) != neural_images.front().size(3)))
            {
                SAIGA_OPTIONAL_TIME_MEASURE("crop", timer_system);

                // The unet has cropped a few pixels because the input wasn't divisible by 16
                if (uv.defined()) uv = CenterCrop2D(uv, x.sizes());
                if (full_target.defined()) target = CenterCrop2D(full_target, x.sizes());
                if (global_mask.defined()) global_mask = CenterCrop2D(global_mask, x.sizes());
                if (local_mask.defined()) local_mask = CenterCrop2D(local_mask, x.sizes());
            }
            else
            {
                target = full_target;
                if (full_target.defined() && (x.size(2) != target.size(2) || x.size(3) != target.size(3)))
                    target = CenterCrop2D(full_target, x.sizes());
            }
        }
        else  // rendered gaussians
        {
            // crop gaussian target if not exactly correct
            target = full_target;
            if (full_target.defined())
                if (neural_images_gs.size(2) != target.size(2) || neural_images_gs.size(3) != target.size(3))
                {
                    target = CenterCrop2D(full_target, neural_images_gs.sizes());
                }
        }
    }


    if (render_gaussian)
    {
        if (foveation)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("fovea_combine", timer_system);
            // neural_images_gs    = torch::zeros_like(neural_images_gs);
            static bool use_new = true;
            // ImGui::Checkbox("use new", &use_new);
            if (use_new)
            {
                std::vector<torch::Tensor> batch_combined;
                for (int i = 0; i < batch.size(); ++i)
                {
                    //   PrintTensorInfo(x);
                    auto np_copy = x.slice(0, i, i + 1).clone().squeeze(0);

                    auto gs_copy = neural_images_gs.slice(0, i, i + 1).slice(1, 0, 3).clone().squeeze(0);

                    combineparams[i].fovea_extend        = fovea_rad;
                    combineparams[i].fovea_pos_in_px     = fovea_poses[i];
                    combineparams[i].complex_fovea_merge = params->render_params.complex_fovea_merge;
                    auto res                             = combine_fovea_gs_np(gs_copy, np_copy, combineparams[i]);
                    // x        = res[0].unsqueeze(0).to(torch::kFloat16);
                    batch_combined.push_back(res[0].unsqueeze(0));
                    // batch_combined.push_back(gs_copy.unsqueeze(0).to(torch::kFloat16));
                }
                x = torch::cat(batch_combined, 0);
            }
            else
            {
                auto result = neural_images_gs.clone();
                if (foveation)
                {
                    // SAIGA_ASSERT(false);
#if 1
                    SAIGA_ASSERT(render_gaussian && render_trips);
                    // RECOMBINE foveated images
                    int border_px = 16;
                    for (int i = 0; i < batch.size(); ++i)
                    {
                        result.slice(0, i, i + 1)
                            .slice(2, fovea_poses[i].y() - fovea_rad / 2 + border_px,
                                   fovea_poses[i].y() + fovea_rad / 2 - border_px)
                            .slice(3, fovea_poses[i].x() - fovea_rad / 2 + border_px,
                                   fovea_poses[i].x() + fovea_rad / 2 - border_px) =
                            x.clone()
                                .slice(0, i, i + 1)
                                .slice(2, border_px, fovea_rad - border_px)
                                .slice(3, border_px, fovea_rad - border_px);
                    }
#endif
                }
                x = result;
            }
        }
        else
            x = neural_images_gs.slice(1, 0, 3).clone();
    }
    // PrintTensorInfo(x);

    if (params->pipeline_params.log_render)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("log_render", timer_system);
        x = torch::exp2(x);
    }

    // if (params->net_params.half_float) SAIGA_ASSERT(x.dtype() == torch::kFloat16);

    if (!params->pipeline_params.skip_sensor_model)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sensor Model", timer_system);
        x = scene.camera->forward(x, frame_index, uv, scale, fixed_exposure, fixed_white_balance, timer_system);
    }
    if (params->net_params.half_float)
    {
        SAIGA_ASSERT(x.dtype() == torch::kFloat16);
        x = x.to(torch::kFloat32);
    }
#if 0
    static int channel = 0;
    ImGui::SliderInt("channel lab: ", &channel, 0, 2);
    // RESULTS
    // auto lab = rgb_to_lab_normalized(x).slice(1, channel, channel + 1).expand({-1, 3, -1, -1}).clone();
    auto lab = rgb_to_lab_normalized(x).clone();
    using namespace torch::indexing;
    x = x.clone().index_put_({Slice(), Slice(), Slice(), Slice(1200)},
                             lab.index({Slice(), Slice(), Slice(), Slice(1200)}));
#endif

    ForwardResult fr;
    fr.x = x;
    // fr.x = neural_images_gs.slice(1, 3, 4).repeat({1, 3, 1, 1}).contiguous().to(torch::kFloat32) / 10.f;

    if (!params->render_params.viewer_only) fr.target = target;
    fr.gaussians = render_gaussian;

    if (render_gaussian)
    {
        fr.viewspace_point_tensor = viewspace_point_tensor;
        fr.visibility_filter      = visibility_filter;
        fr.radii                  = radii;
        fr.gaussians              = true;
    }

    // Eval loss only if required
    torch::Tensor lt_vgg, lt_l1, lt_mse, lt_ssim, lt_lpips;
    if (!params->render_params.viewer_only || loss_statistics)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Compute Loss", timer_system);

        if (loss_statistics || params->train_params.loss_vgg > 0 || params->train_params.loss_l1 > 0 ||
            params->train_params.loss_mse > 0 || params->train_params.loss_ssim > 0 ||
            params->train_params.loss_lpips > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("Masking", timer_system);

            SAIGA_ASSERT(target.defined());
            fr.loss = torch::zeros({1}, torch::TensorOptions().device(device));
            if (global_mask.defined())
            {
                // PrintTensorInfo(x);
                // PrintTensorInfo(global_mask);
                // PrintTensorInfo(target);

                x      = x * global_mask;
                target = target * global_mask;
            }

            if (local_mask.defined())
            {
                // std::cout << TensorInfo(x) << TensorInfo(local_mask) << TensorInfo(target) << std::endl;
                x      = x * local_mask;
                target = target * local_mask;
            }
        }

        fr.float_loss.count = 1;

        // auto x_crop = x;
        // auto t_crop = target;
        // if (params->train_params.crop_for_loss)
        //{
        //     x_crop = CenterCrop2D(x, 16);
        //     t_crop = CenterCrop2D(target, 16);
        // }
        //  PrintTensorInfo(x_crop);
        //  PrintTensorInfo(t_crop);

        if (render_trips && (loss_statistics || (params->train_params.loss_vgg > 0 &&
                                                 current_epoch > params->train_params.only_start_vgg_after_epochs)))
        {
            const int border_px = 16;
            SAIGA_ASSERT(batch.size() == 1);
            auto vgg_x =
                network_output.slice(2, border_px, fovea_rad - border_px).slice(3, border_px, fovea_rad - border_px);
            ;
            // x.slice(2, fovea_poses[0].y() - fovea_rad / 2 + border_px,
            //                  fovea_poses[0].y() + fovea_rad / 2 - border_px)
            //              .slice(3, fovea_poses[0].x() - fovea_rad / 2 + border_px,
            //                     fovea_poses[0].x() + fovea_rad / 2 - border_px);
            auto vgg_target = target
                                  .slice(2, fovea_poses[0].y() - fovea_rad / 2 + border_px,
                                         fovea_poses[0].y() + fovea_rad / 2 - border_px)
                                  .slice(3, fovea_poses[0].x() - fovea_rad / 2 + border_px,
                                         fovea_poses[0].x() + fovea_rad / 2 - border_px);

            SAIGA_OPTIONAL_TIME_MEASURE("VGG", timer_system);
            SAIGA_ASSERT(fr.loss.defined());
            // lt_vgg = loss_vgg->forward(x_crop, t_crop);
            lt_vgg = loss_vgg->forward(vgg_x, vgg_target);
            // lt_vgg = loss_vgg->forward(x, target);
            fr.loss += lt_vgg * params->train_params.loss_vgg;
            fr.float_loss.loss_vgg = lt_vgg.item().toFloat();
        }
#if 0
        if (loss_statistics || (params->train_params.loss_similarity > 0 && render_gaussian && render_trips))
        {
            const int border_px = 16;
            SAIGA_ASSERT(batch.size() == 1);
            auto similarity_np =
                network_output.slice(2, border_px, fovea_rad - border_px).slice(3, border_px, fovea_rad - border_px);

            auto similarity_gs =
                gs_centers_t.slice(2, border_px, fovea_rad - border_px).slice(3, border_px, fovea_rad - border_px);

            auto simi_loss = torch::l1_loss(network_output.mean(), similarity_gs.mean());
            fr.loss += simi_loss;
            fr.float_loss.loss_similarity = simi_loss.item().toFloat();
        }
#endif
        if (loss_statistics || params->train_params.loss_l1 > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("L1", timer_system);

            lt_l1 = torch::l1_loss(x, target);
            fr.loss += lt_l1 * params->train_params.loss_l1;
            fr.float_loss.loss_l1 = lt_l1.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_mse > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("MSE", timer_system);

            lt_mse = torch::mse_loss(x, target);
            fr.loss += lt_mse * params->train_params.loss_mse;
            fr.float_loss.loss_mse = lt_mse.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_ssim > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("SSIM", timer_system);

            lt_ssim = (1.f - loss_ssim->forward(x, target)) / 2.f;
            fr.loss += lt_ssim * params->train_params.loss_ssim;
            fr.float_loss.loss_ssim = lt_ssim.item().toFloat();
        }
        if (loss_statistics || params->train_params.loss_lpips > 0)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("LPIPS", timer_system);

            lt_lpips = loss_lpips.forward(x, target).sum();
            fr.loss += lt_lpips * params->train_params.loss_lpips;
            fr.float_loss.loss_lpips = lt_lpips.item().toFloat();
        }
        if (loss_statistics)
        {
            auto i                  = x.clamp(0, 1);
            auto t                  = target.clamp(0, 1);
            fr.float_loss.loss_psnr = loss_psnr->forward(i, t).item().toFloat();
        }
    }
    if (!params->render_params.viewer_only)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Stats and bookkeeping", timer_system);

        if (loss_statistics)
        {
            auto i                  = x.clamp(0, 1);
            auto t                  = target.clamp(0, 1);
            fr.float_loss.loss_psnr = loss_psnr->forward(i, t).item().toFloat();
            // fr.float_loss.loss_lpips = loss_lpips.forward(i, t).item().toFloat();
        }

        if (fr.loss.defined())
        {
            fr.float_loss.loss_float = fr.loss.item().toFloat();

            auto param_loss = params->optimizer_params.response_smoothness * scene.camera->ParamLoss(frame_index);
            fr.float_loss.loss_float_param = param_loss.item().toFloat();
            fr.loss += param_loss;

            if (params->pipeline_params.verbose_eval)
            {
                int index = frame_index.item().toLong();
                std::cout << "frame index " << index << std::endl;
                std::cout << "loss " << fr.float_loss.loss_float << std::endl;
                std::cout << "====" << std::endl;
            }

            if (!std::isfinite(fr.float_loss.loss_float))
            {
                for (auto i : neural_images)
                {
                    PrintTensorInfo(i);
                }
                std::cout << std::endl;
                PrintTensorInfo(uv);
                PrintTensorInfo(fr.loss);
                PrintTensorInfo(x);
                PrintTensorInfo(global_mask);
                PrintTensorInfo(target);
                std::cout << std::endl;
                std::cout << "Scene:" << std::endl;
                scene.Log("debug/");
                Log("debug/");
                throw std::runtime_error("Loss not finite :(");
            }
        }
    }
    // 5. Convert to image for visualization and debugging
    if (keep_image)
    {
        // torch::Tensor x_full;
        // if (!params->render_params.viewer_only)
        // x_full = CenterEmplace(x, torch::ones_like(full_target));
        //  else
        //     x_full = x;

        for (int i = 0; i < batch.size(); ++i)
        {
            auto x_full = CenterEmplace(x, torch::ones_like(full_target));
            fr.outputs.push_back(Saiga::TensorToImage<ucvec3>(x_full[i]));
            fr.targets.push_back(Saiga::TensorToImage<ucvec3>(full_target[i]));

            //            fr.outputs.push_back(Saiga::TensorToImage<ucvec3>(x[i]));
            //            fr.targets.push_back(Saiga::TensorToImage<ucvec3>(target[i]));
            fr.image_ids.push_back(frame_index[i].item().toLong());
        }
    }

    return fr;
}
void NeuralPipeline::OptimizerStep(int epoch_id)
{
    if (render_optimizer)
    {
        render_optimizer->step();
        render_optimizer->zero_grad();
    }
    if (refinement_optimizer && epoch_id > params->train_params.lock_dynamic_refinement_epochs)
    {
        refinement_optimizer->step();
        refinement_optimizer->zero_grad();
    }
}

void NeuralPipeline::OptimizerClear(int epoch_id)
{
    if (render_optimizer)
    {
        render_optimizer->zero_grad();
    }
    if (refinement_optimizer)
    {
        refinement_optimizer->zero_grad();
    }
}
void NeuralPipeline::UpdateLearningRate(double factor)
{
    if (render_optimizer)
    {
        UpdateOptimLR(render_optimizer.get(), factor);
    }
    if (refinement_optimizer)
    {
        UpdateOptimLR(refinement_optimizer.get(), factor);
    }
}

void NeuralPipeline::Train(bool train)
{
    // render_module = nullptr;
    // c10::cuda::CUDACachingAllocator::emptyCache();
    // std::cout << "EMPTY!!!!!!" << std::endl;
    // render_module = PointRenderModule(params);

    if (render_optimizer)
    {
        // render_optimizer->zero_grad();
    }

    render_module->train(train);
    if (params->optimizer_params.fix_render_network)
    {
        render_network->train(train);
    }
    else
    {
        render_network->train(train);
    }
}
void NeuralPipeline::Log(const std::string& dir) {}
