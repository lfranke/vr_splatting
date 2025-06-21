/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/time/time.h"
#include "saiga/core/util/MemoryUsage.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/file.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/LRScheduler.h"

#include "TrainScene.h"
#include "data/Dataset.h"
#include "data/OptionalTensorBoardLogger.h"
#include "models/Pipeline.h"
#include "opengl/TrainLiveViewer.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <csignal>
#include <torch/script.h>

#include "git_sha1.h"
#include "neat-utils/NeAT_interop.h"
#include "neat-utils/image_utils.h"

// #undef LIVE_VIEWER

#ifdef LIVE_VIEWER
std::shared_ptr<TrainViewer> viewer;
#endif

std::string full_experiment_dir;
std::shared_ptr<CombinedParams> params;
torch::Device device = torch::kCUDA;
ImGui::IMConsole console_error;

class NeuralTrainer;
NeuralTrainer* instance;

class NeuralTrainer
{
   public:
    torch::DeviceType device = torch::kCUDA;
    std::shared_ptr<NeuralPipeline> pipeline;

    torch::Tensor train_crop_mask;
    std::shared_ptr<TrainScene> train_scenes;
    std::string ep_dir;
    LRSchedulerPlateau lr_scheduler;

    std::shared_ptr<OptionalTensorBoardLogger> tblogger;

    std::string experiment_name;
    CUDA::CudaTimerSystem timer_system;
    CUDA::CudaTimerSystem timer_system_non_pipeline;

    bool restart_training = false;

    void signal_handler(int signal)
    {
        ep_dir.pop_back();
        ep_dir += "_interupted/";
        std::filesystem::create_directory(ep_dir);

        bool reduced_cp = params->train_params.reduced_check_point;
        // Save checkpoint
        console << "Saving checkpoint..." << std::endl;
        if (!reduced_cp) pipeline->SaveCheckpoint(ep_dir);
        for (auto& s : train_scenes->data) s.scene->SaveCheckpoint(ep_dir, reduced_cp);
        exit(0);
    }

    ~NeuralTrainer() {}

    NeuralTrainer()
    {
        instance = this;
        lr_scheduler =
            LRSchedulerPlateau(params->train_params.lr_decay_factor, params->train_params.lr_decay_patience, true);
        torch::set_num_threads(8);

        experiment_name     = Saiga::CurrentTimeString("%F_%H-%M-%S") + "_" + params->train_params.name;
        full_experiment_dir = params->train_params.experiment_dir + "/" + experiment_name + "/";
        std::filesystem::create_directories(full_experiment_dir);

        console.setOutputFile(full_experiment_dir + "log.txt");
        SAIGA_ASSERT(console.rdbuf());
        std::cout.rdbuf(console.rdbuf());

        console_error.setOutputFile(full_experiment_dir + "error.txt");
        SAIGA_ASSERT(console_error.rdbuf());
        std::cerr.rdbuf(console_error.rdbuf());

        tblogger = std::make_shared<OptionalTensorBoardLogger>((full_experiment_dir + "/tfevents.pb").c_str());

        train_scenes = std::make_shared<TrainScene>(params->train_params.scene_names);

        // Save all parameters into experiment output dir
        params->Save(full_experiment_dir + "/params.ini");

        {
            std::ofstream strm(full_experiment_dir + "/git.txt");
            strm << GIT_SHA1 << std::endl;
        }

        pipeline = std::make_shared<NeuralPipeline>(params);
    }
    bool run()
    {
        int train_loader_size = 0;
        dataloader_type_t train_loader =
            train_scenes->DataLoader(train_scenes->train_cropped_samplers, true, train_loader_size);

        int ev_refine_loader_size = 0;
        dataloader_type_t ev_refine_loader;
        if (params->train_params.optimize_eval_camera)
        {
            ev_refine_loader =
                train_scenes->DataLoader(train_scenes->test_cropped_samplers, true, ev_refine_loader_size);
        }
        pipeline->Train(true);


        for (int epoch_id = 0; epoch_id <= params->train_params.num_epochs; ++epoch_id)
        {
            timer_system_non_pipeline.BeginFrame();
            auto non_pipeline_timer = &timer_system_non_pipeline;
            {
                SAIGA_OPTIONAL_TIME_MEASURE("Epoch", non_pipeline_timer);

                std::cout << std::endl;
                std::cout << "=== Epoch " << epoch_id << " ===" << std::endl;
                std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);

                bool last_ep         = epoch_id == params->train_params.num_epochs;
                bool save_checkpoint = epoch_id % params->train_params.save_checkpoints_its == 0 || last_ep;

                // pipeline->Train(epoch_id);

                ep_dir = full_experiment_dir + "ep" + ep_str + "/";
                if (save_checkpoint)
                {
                    std::filesystem::create_directory(ep_dir);
                    std::filesystem::create_directory(ep_dir + "/test/");
                }
                {
                    static int descriptor_resets_in_training = 0;

                    if (descriptor_resets_in_training < params->optimizer_params.number_of_descriptor_resets &&
                        params->optimizer_params.reset_descriptors_interval > 0 &&
                        (epoch_id % params->optimizer_params.reset_descriptors_interval) == 0 && epoch_id > 0)
                    {
                        for (auto& sd : train_scenes->data)
                        {
                            std::cout << "Reset Texture" << std::endl;
                            sd.scene->CreateTexture();
                            sd.scene->CreateTextureOptimizer();
                            sd.scene->texture->to(device);
                        }
                    }
                    if (params->train_params.do_train && epoch_id > 0)
                    {
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Add-Remove", non_pipeline_timer);
                            // AddAndRemovePoints(epoch_id);
                            if (params->points_adding_params.push_point_confidences_down != 0.f &&
                                epoch_id > params->points_adding_params.start_removing_points_epoch &&
                                (epoch_id % params->points_adding_params.point_removal_epoch_interval) == 0)
                            {
                                torch::NoGradGuard no_grad;
                                for (auto& sd : train_scenes->data)

                                    sd.scene->RemovePointsWithConfUnder(
                                        params->points_adding_params.removal_confidence_cutoff, true);
                                // 0.0001 * (scene_data.scene->texture->confidence_raw.mutable_grad().max() -
                                //           scene_data.scene->texture->confidence_raw.mutable_grad().min());
                            }
                        }
                        double epoch_loss;
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Train", non_pipeline_timer);

                            epoch_loss = TrainEpoch(epoch_id, train_scenes->train_cropped_samplers, false, "Train",
                                                    train_loader, train_loader_size);
                        }
                        if (restart_training) return false;

                        for (auto& sd : train_scenes->data)
                        {
                            tblogger->add_scalar("LossTrain/" + sd.scene->scene->scene_name, epoch_id,
                                                 sd.epoch_loss.Average().loss_float);
                            sd.epoch_loss.Average().AppendToFile(
                                full_experiment_dir + "loss_train_" + sd.scene->scene->scene_name + ".txt", epoch_id);
                        }

                        if (params->train_params.optimize_eval_camera)
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Train2", non_pipeline_timer);
                            TrainEpoch(epoch_id, train_scenes->test_cropped_samplers, true, "EvalRefine",
                                       ev_refine_loader, ev_refine_loader_size);
                            for (auto& sd : train_scenes->data)
                            {
                                tblogger->add_scalar("LossEvalRefine/" + sd.scene->scene->scene_name, epoch_id,
                                                     sd.epoch_loss.Average().loss_float);
                                sd.epoch_loss.Average().AppendToFile(
                                    full_experiment_dir + "loss_eval_refine_" + sd.scene->scene->scene_name + ".txt",
                                    epoch_id);
                            }
                        }

                        auto reduce_factor           = lr_scheduler.step(epoch_loss);
                        static double current_factor = 1;
                        tblogger->add_scalar("LR/factor", epoch_id, current_factor);

                        current_factor *= reduce_factor;

                        if (reduce_factor < 1)
                        {
                            std::cout << "Reducing LR by " << reduce_factor << ". Current Factor: " << current_factor
                                      << std::endl;
                        }

                        pipeline->UpdateLearningRate(reduce_factor);
                        for (auto& s : train_scenes->data)
                        {
                            s.scene->UpdateLearningRate(epoch_id, reduce_factor);
                        }
                    }

                    if (params->train_params.debug)
                    {
                        std::cout << GetMemoryInfo() << std::endl;
                    }
                    pipeline->Log(full_experiment_dir);
                    for (auto& s : train_scenes->data)
                    {
                        s.scene->Log(full_experiment_dir);
                    }
                    bool want_eval = params->train_params.do_eval &&
                                     (!params->train_params.eval_only_on_checkpoint || save_checkpoint);

                    if (want_eval)
                    {
                        {
                            // SAIGA_OPTIONAL_TIME_MEASURE("Eval", non_pipeline_timer);
                            // EvalEpoch(epoch_id, save_checkpoint);
                        }
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Test", non_pipeline_timer);
                            TestEpoch(epoch_id);
                        }
                        for (auto& sd : train_scenes->data)
                        {
                            auto avg = sd.epoch_loss.Average();

                            tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/vgg", epoch_id,
                                                 avg.loss_vgg);
                            tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/lpips", epoch_id,
                                                 avg.loss_lpips);
                            tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/psnr", epoch_id,
                                                 avg.loss_psnr);
                            tblogger->add_scalar("LossEval/" + sd.scene->scene->scene_name + "/ssim", epoch_id,
                                                 avg.loss_ssim);

                            avg.AppendToFile(full_experiment_dir + "loss_eval_" + sd.scene->scene->scene_name + ".txt",
                                             epoch_id);
                        }
                    }
                }

                if (save_checkpoint)
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("SaveCheckpoint", non_pipeline_timer);
                    bool reduced_cp = params->train_params.reduced_check_point && !last_ep;
                    // Save checkpoint
                    console << "Saving checkpoint..." << std::endl;

                    if (!reduced_cp)
                    {
                        pipeline->SaveCheckpoint(ep_dir);
                    }

                    for (auto& s : train_scenes->data)
                    {
                        s.scene->SaveCheckpoint(ep_dir, reduced_cp);
                    }
                }
            }
            timer_system_non_pipeline.EndFrame();
            timer_system_non_pipeline.PrintTable(std::cout);
        }

        std::string finished_ep_dir = params->train_params.experiment_dir + "/" + "_f_" + experiment_name + "/";
        std::cout << "rename " << full_experiment_dir << " to " << finished_ep_dir << std::endl;
        std::filesystem::rename(full_experiment_dir, finished_ep_dir);
        return true;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double TrainEpoch(int epoch_id, std::vector<SceneDataTrainSampler>& data, bool structure_only, std::string name,
                      dataloader_type_t& loader, int loader_size)
    {
        for (auto& d : data)
        {
            d.Start(epoch_id);
        }

        auto non_pipeline_timer  = &timer_system_non_pipeline;
        auto create_loader_timer = non_pipeline_timer->CreateNewTimer("CreateDataLoader");
        create_loader_timer->Start();
        train_scenes->StartEpoch();
        // Train
        float epoch_loss = 0;
        int num_images   = 0;

        pipeline->render_module->train(true);
        create_loader_timer->Stop();

        // auto data_timer       = non_pipeline_timer->CreateNewTimer("Dataloading Each Epoch");
        // auto processing_timer = non_pipeline_timer->CreateNewTimer("Processing Each Epoch");
        // data_timer->Start();

        int longer_iter        = 1;
        int gs_iteration_start = (epoch_id - 1) * loader_size * longer_iter + 1;


        Saiga::ProgressBar bar(std::cout,
                               name + " " + std::to_string(epoch_id) + "::" + std::to_string(gs_iteration_start) + " |",
                               loader_size * params->train_params.inner_batch_size * longer_iter, 30, false, 5000);

        for (int j = 0; j < longer_iter; ++j)
        {
            for (std::vector<NeuralTrainData>& batch : *loader)
            {
                // data_timer->Stop();
                // processing_timer->Start();
                SAIGA_ASSERT(batch.size() <= params->train_params.batch_size * params->train_params.inner_batch_size);
                pipeline->timer_system = &timer_system;

                timer_system.BeginFrame();

                int scene_id_of_batch = batch.front()->scene_id;
                auto& scene_data      = train_scenes->data[scene_id_of_batch];
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("Load Scene", pipeline->timer_system);

                    train_scenes->Load(device, scene_id_of_batch);
                }
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("Setup Train", pipeline->timer_system);

                    train_scenes->Train(epoch_id, true);
                }
                int gs_iteration = (epoch_id - 1) * loader_size * longer_iter + num_images + 1;
                if (gs_iteration % 1000 == 0)
                {
                    scene_data.scene->gaussian_model->One_up_sh_degree();
                }
                {
                    if (!train_crop_mask.defined() && params->train_params.train_mask_border > 0)
                    {
                        SAIGA_OPTIONAL_TIME_MEASURE("Create Mask", pipeline->timer_system);

                        int h           = batch.front()->img.h;
                        int w           = batch.front()->img.w;
                        train_crop_mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                        TensorToImage<unsigned char>(train_crop_mask).save(full_experiment_dir + "/train_mask.png");
                    }

                    ForwardResult result;
                    {
                        SAIGA_OPTIONAL_TIME_MEASURE("Forward", pipeline->timer_system);

                        result = pipeline->Forward(*scene_data.scene, batch, train_crop_mask, false, epoch_id, false);
                        scene_data.epoch_loss += result.float_loss;
                        epoch_loss += result.float_loss.loss_float * batch.size();
                        num_images += batch.size();
                    }

                    {
                        auto timer = timer_system.Measure("Backwards");
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Backward()", pipeline->timer_system);
                            result.loss.backward();
                        }
                        if (params->points_adding_params.push_point_confidences_down != 0.f)
                        {
                            scene_data.scene->texture->confidence_raw.mutable_grad() +=
                                params->points_adding_params.push_point_confidences_down;
                            // 0.0001 * (scene_data.scene->texture->confidence_raw.mutable_grad().max() -
                            //           scene_data.scene->texture->confidence_raw.mutable_grad().min());
                        }
                        // gaussian densify and prune
                        if (result.gaussians && gs_iteration < 30000)
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("TrackGS", pipeline->timer_system);

                            torch::NoGradGuard no_grad;
                            auto visible_max_radii =
                                scene_data.scene->gaussian_model->_max_radii2D.masked_select(result.visibility_filter);
                            auto visible_radii = result.radii.masked_select(result.visibility_filter);
                            auto max_radii     = torch::max(visible_max_radii, visible_radii);
                            scene_data.scene->gaussian_model->_max_radii2D =
                                scene_data.scene->gaussian_model->_max_radii2D.masked_scatter_(result.visibility_filter,
                                                                                               max_radii);
                        }

                        if (result.gaussians && gs_iteration < 30000)
                        {
                            torch::NoGradGuard no_grad;

                            SAIGA_OPTIONAL_TIME_MEASURE("DensifyAndPruneGS", pipeline->timer_system);

                            if (scene_data.scene->optimParams.use_densify &&
                                gs_iteration < scene_data.scene->optimParams.densify_until_iter)
                            {
                                scene_data.scene->gaussian_model->Add_densification_stats(
                                    result.viewspace_point_tensor, result.visibility_filter,
                                    params->pipeline_params.gaussian_densification_abs);
                                if (gs_iteration > scene_data.scene->optimParams.densify_from_iter &&
                                    ((gs_iteration % scene_data.scene->optimParams.densification_interval) == 0))
                                {
                                    std::cout << std::endl << "GS densify" << std::endl;

                                    float size_threshold =
                                        gs_iteration > scene_data.scene->optimParams.opacity_reset_interval ? 20.f
                                                                                                            : -1.f;

                                    float grad_threshold = scene_data.scene->optimParams.densify_grad_threshold;
                                    if (params->pipeline_params.gaussian_densification_abs) grad_threshold *= 4.f;

                                    scene_data.scene->gaussian_model->Densify_and_prune(
                                        grad_threshold, scene_data.scene->optimParams.min_opacity,
                                        scene_data.scene->cameras_extent, size_threshold);
                                }

                                if (!params->render_params.opacity_decay &&
                                    gs_iteration < scene_data.scene->optimParams.opacity_reset_until_iter &&
                                    (gs_iteration % scene_data.scene->optimParams.opacity_reset_interval == 0 ||
                                     (scene_data.scene->modelParams.white_background &&
                                      gs_iteration == scene_data.scene->optimParams.densify_from_iter)))
                                {
                                    std::cout << std::endl << "GS reset opacity" << std::endl;
                                    scene_data.scene->gaussian_model->Reset_opacity();
                                }
                                if (params->render_params.opacity_decay && gs_iteration % 50 == 0 &&
                                    gs_iteration > scene_data.scene->optimParams.densify_from_iter)
                                {
                                    scene_data.scene->gaussian_model->Decay_opacity();
                                }
                            }
                        }
                        if (!structure_only)
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Optimizer Step Pipeline", pipeline->timer_system);
                            pipeline->OptimizerStep(epoch_id);
                        }
                        if (result.gaussians && gs_iteration < 30000)
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Gaussian Optimizer Step Pipeline", pipeline->timer_system);

                            //  Optimizer step
                            if (gs_iteration < scene_data.scene->optimParams.iterations)
                            {
                                scene_data.scene->gaussian_model->_optimizer->step();
                                scene_data.scene->gaussian_model->_optimizer->zero_grad(true);
                                scene_data.scene->gaussian_model->Update_learning_rate(gs_iteration);
                            }
                        }
                        {
                            SAIGA_OPTIONAL_TIME_MEASURE("Optimizer Step Scene", pipeline->timer_system);
                            scene_data.scene->OptimizerStep(epoch_id, structure_only);
                        }
                    }
                    bar.addProgress(batch.size());
                    bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) + " " +
                                   std::to_string(num_images) + " Avg=" + std::to_string(epoch_loss / num_images));

#ifdef LIVE_VIEWER
                    if (name == "Train") viewer->addLossSample(epoch_loss / float(num_images));
#endif
                }
#ifdef LIVE_VIEWER
                if (name == "Train")
                {
                    SAIGA_OPTIONAL_TIME_MEASURE("LiveViewer", pipeline->timer_system);

                    torch::NoGradGuard ngg;
                    viewer->checkAndUpdateTexture(pipeline, scene_data.scene, epoch_id);
                    restart_training = viewer->checkForRestart();
                    if (restart_training)
                    {
                        timer_system.EndFrame();
                        timer_system.PrintTable(std::cout);
                        non_pipeline_timer->EndFrame();
                        non_pipeline_timer->PrintTable(std::cout);
                        train_scenes->Unload(true);

                        // data_timer->Stop();
                        // processing_timer->Stop();
                        return 0;
                    }
                }
#endif
                timer_system.EndFrame();
                // processing_timer->Stop();
                // data_timer->Start();
            }
        }
        // data_timer->Stop();

        timer_system.PrintTable(std::cout);

        train_scenes->Unload();
        for (auto& d : data)
        {
            d.Finish();
        }
#ifdef LIVE_VIEWER
        if (name == "Train") viewer->addEpochLoss(epoch_loss / float(num_images));

#endif
        return epoch_loss / float(num_images);
    }


    void EvalEpoch(int epoch_id, bool save_checkpoint)
    {
        train_scenes->StartEpoch();

        // Eval
        torch::NoGradGuard ngg;
        float epoch_loss = 0;
        int num_images   = 0;
        int loader_size  = 0;
        auto loader      = train_scenes->DataLoader(train_scenes->eval_samplers, false, loader_size);

        pipeline->render_module->train(false);

        float best_loss  = 1000000;
        float worst_loss = 0;
        ForwardResult best_batch, worst_batch;

        bool write_test_images = save_checkpoint && params->train_params.write_test_images;

        pipeline->Train(false);

        Saiga::ProgressBar bar(std::cout, "Eval  " + std::to_string(epoch_id) + " |", loader_size, 30, false, 5000);
        for (std::vector<NeuralTrainData>& batch : *loader)
        {
            SAIGA_ASSERT(batch.size() == 1);
            int scene_id_of_batch = batch.front()->scene_id;
            auto& scene_data      = train_scenes->data[scene_id_of_batch];
            train_scenes->Load(device, scene_id_of_batch);
            train_scenes->Train(epoch_id, false);
            int camera_id = batch.front()->img.camera_index;

            SAIGA_ASSERT(!torch::GradMode::is_enabled());

            auto result =
                pipeline->Forward(*scene_data.scene, batch, scene_data.eval_crop_mask[camera_id], true, epoch_id,
                                  write_test_images | params->train_params.write_images_at_checkpoint);

            if (params->train_params.write_images_at_checkpoint)
            {
                for (int i = 0; i < result.image_ids.size(); ++i)
                {
                    // In average only write 10 images
                    if (Random::sampleBool(std::min(1.0, 10.0 / loader_size)))
                    {
                        auto err = ImageTransformation::ErrorImage(result.outputs[i], result.targets[i]);
                        TemplatedImage<ucvec3> combined(err.h, err.w + result.outputs[i].w);
                        combined.getImageView().setSubImage(0, 0, result.outputs[i].getImageView());
                        combined.getImageView().setSubImage(0, result.outputs[i].w, err.getImageView());

                        // LogImage(
                        //     tblogger.get(), combined,
                        //     "Checkpoint" + leadingZeroString(epoch_id, 4) + "/" +
                        //     scene_data.scene->scene->scene_name, result.image_ids[i]);


                        result.outputs[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) +
                                               params->train_params.output_file_type);

                        result.targets[i].save(ep_dir + "/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_gt" +
                                               params->train_params.output_file_type);
                    }
                }
            }
            if (write_test_images)
            {
                if (result.float_loss.loss_float < best_loss)
                {
                    best_loss  = result.float_loss.loss_float;
                    best_batch = result;
                }

                if (result.float_loss.loss_float > worst_loss)
                {
                    worst_loss  = result.float_loss.loss_float;
                    worst_batch = result;
                }
            }

            epoch_loss += result.float_loss.loss_float;
            scene_data.epoch_loss += result.float_loss;
            num_images += batch.size();

            std::vector<ReducedImageInfo> images_rii;
            for (int b_id = 0; b_id < batch.size(); ++b_id)
            {
                images_rii.push_back(batch[b_id]->img);
            }

            // before point adding: save error maps
            bar.addProgress(batch.size());
            bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) +
                           " Avg=" + std::to_string(epoch_loss / num_images));
        }
        bar.Quit();

        train_scenes->Unload();

        if (write_test_images)
        {
            console << "Best - Worst (Eval) [" << best_loss << ", " << worst_loss << "]" << std::endl;

            for (int i = 0; i < best_batch.targets.size(); ++i)
            {
                best_batch.targets[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_target" +
                                           params->train_params.output_file_type);
                best_batch.outputs[i].save(ep_dir + "/img_best_" + to_string(best_batch.image_ids[i]) + "_output" +
                                           params->train_params.output_file_type);
            }
            for (int i = 0; i < worst_batch.targets.size(); ++i)
            {
                worst_batch.targets[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_target" +
                                            params->train_params.output_file_type);
                worst_batch.outputs[i].save(ep_dir + "/img_worst_" + to_string(worst_batch.image_ids[i]) + "_output" +
                                            params->train_params.output_file_type);
            }
        }

        for (auto& sd : train_scenes->data)
        {
            console << "Loss: " << std::setw(20) << sd.scene->scene->scene_name << " ";
            sd.epoch_loss.Average().Print();
        }
        pipeline->Train(true);
    }


    void TestEpoch(int epoch_id)
    {
        train_scenes->StartEpoch();

        if (params->train_params.interpolate_eval_settings)
        {
            SAIGA_ASSERT(params->train_params.optimize_eval_camera == false);
            for (int i = 0; i < train_scenes->data.size(); ++i)
            {
                auto indices = train_scenes->data[i].not_training_indices;
                train_scenes->data[i].scene->camera->InterpolateFromNeighbors(indices);
            }
        }

        // Eval
        torch::NoGradGuard ngg;
        float epoch_loss = 0;
        int num_images   = 0;
        int loader_size  = 0;
        auto loader      = train_scenes->DataLoader(train_scenes->test_samplers, false, loader_size);

        pipeline->Train(false);

        float best_loss  = 1000000;
        float worst_loss = 0;
        ForwardResult best_batch, worst_batch;

        bool write_test_images = params->train_params.write_test_images;

        pipeline->render_module->train(false);

        Saiga::ProgressBar bar(std::cout, "Test  " + std::to_string(epoch_id) + " |", loader_size, 30, false, 5000);
        for (std::vector<NeuralTrainData>& batch : *loader)
        {
            SAIGA_ASSERT(batch.size() == 1);
            int scene_id_of_batch = batch.front()->scene_id;
            auto& scene_data      = train_scenes->data[scene_id_of_batch];
            train_scenes->Load(device, scene_id_of_batch);
            train_scenes->Train(epoch_id, false);
            int camera_id = batch.front()->img.camera_index;

            batch.front()->pixel_view_pos = ivec2(batch.front()->img.w / 2, batch.front()->img.h / 2);

            SAIGA_ASSERT(!torch::GradMode::is_enabled());

            auto result =
                pipeline->Forward(*scene_data.scene, batch, scene_data.eval_crop_mask[camera_id], true, epoch_id,
                                  write_test_images | params->train_params.write_images_at_checkpoint);

            if (params->train_params.write_images_at_checkpoint)
            {
                for (int i = 0; i < result.image_ids.size(); ++i)
                {
                    {
                        auto err = ImageTransformation::ErrorImage(result.outputs[i], result.targets[i]);
                        TemplatedImage<ucvec3> combined(err.h, err.w + result.outputs[i].w);
                        combined.getImageView().setSubImage(0, 0, result.outputs[i].getImageView());
                        combined.getImageView().setSubImage(0, result.outputs[i].w, err.getImageView());

                        result.outputs[i].save(ep_dir + "/test/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) +
                                               params->train_params.output_file_type);

                        result.targets[i].save(ep_dir + "/test/" + scene_data.scene->scene->scene_name + "_" +
                                               leadingZeroString(result.image_ids[i], 5) + "_gt" +
                                               params->train_params.output_file_type);
                    }
                }
            }
            if (write_test_images)
            {
                if (result.float_loss.loss_float < best_loss)
                {
                    best_loss  = result.float_loss.loss_float;
                    best_batch = result;
                }

                if (result.float_loss.loss_float > worst_loss)
                {
                    worst_loss  = result.float_loss.loss_float;
                    worst_batch = result;
                }
            }

            epoch_loss += result.float_loss.loss_float;
            scene_data.epoch_loss += result.float_loss;
            num_images += batch.size();

            std::vector<ReducedImageInfo> images_rii;
            for (int b_id = 0; b_id < batch.size(); ++b_id)
            {
                images_rii.push_back(batch[b_id]->img);
            }

            bar.addProgress(batch.size());
            bar.SetPostfix(" Cur=" + std::to_string(result.float_loss.loss_float) +
                           " Avg=" + std::to_string(epoch_loss / num_images));
        }
        train_scenes->Unload();

        bar.Quit();

        if (write_test_images)
        {
            console << "Best - Worst (Eval) [" << best_loss << ", " << worst_loss << "]" << std::endl;

            for (int i = 0; i < best_batch.targets.size(); ++i)
            {
                best_batch.targets[i].save(ep_dir + "/test/img_best_" + to_string(best_batch.image_ids[i]) + "_target" +
                                           params->train_params.output_file_type);
                best_batch.outputs[i].save(ep_dir + "/test/img_best_" + to_string(best_batch.image_ids[i]) + "_output" +
                                           params->train_params.output_file_type);
            }
            for (int i = 0; i < worst_batch.targets.size(); ++i)
            {
                worst_batch.targets[i].save(ep_dir + "/test/img_worst_" + to_string(worst_batch.image_ids[i]) +
                                            "_target" + params->train_params.output_file_type);
                worst_batch.outputs[i].save(ep_dir + "/test/img_worst_" + to_string(worst_batch.image_ids[i]) +
                                            "_output" + params->train_params.output_file_type);
            }
        }

        for (auto& sd : train_scenes->data)
        {
            console << "Loss: " << std::setw(20) << sd.scene->scene->scene_name << " ";
            sd.epoch_loss.Average().Print();
        }
        pipeline->Train(true);
    }
};

CombinedParams LoadParamsHybrid(int argc, const char** argv)
{
    CLI::App app{"Train on your Scenes", "train"};
    std::string config_file;
    app.add_option("--config", config_file)->required();
    auto params = CombinedParams();
    params.Load(app);
    try
    {
        app.parse(argc, argv);
    }
    catch (CLI::ParseError& error)
    {
        std::cout << "Parsing failed!" << std::endl;
        std::cout << error.what() << std::endl;
        CHECK(false);
    }

    std::cout << "Loading Config File " << config_file << std::endl;
    params.Load(config_file);
    app.parse(argc, argv);

    return params;
}

void call_handler(int signal)
{
    instance->signal_handler(signal);
    exit(0);
}


int main(int argc, const char* argv[])
{
    std::cout << "Git ref: " << GIT_SHA1 << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH
              << std::endl;
    long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
    std::cout << "The cuDNN version is " << cudnn_version << std::endl;
    std::cout << "cuDNN avail? " << torch::cuda::cudnn_is_available() << std::endl;
    // at::globalContext().setUserEnabledCuDNN(false);
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    std::cout << "The CUDA runtime version is " << runtimeVersion << std::endl;
    int version;
    cudaDriverGetVersion(&version);
    std::cout << "The driver version is " << version << std::endl;
    at::globalContext().setBenchmarkCuDNN(true);

    // at::globalContext().setAllowTF32CuBLAS(true);
    // at::globalContext().setAllowTF32CuDNN(true);
    //// at::globalContext().setAllowBF16ReductionCuBLAS(true);
    // at::globalContext().setAllowFP16ReductionCuBLAS(true);
    // at::globalContext().setFloat32MatmulPrecision("medium");

#ifdef LIVE_VIEWER

    WindowParameters windowParameters;
    OpenGLParameters openglParameters;
    DeferredRenderingParameters rendererParameters;
    windowParameters.fromConfigFile("config.ini");
    rendererParameters.hdr = true;
    MainLoopParameters mlp;
    viewer = std::make_shared<TrainViewer>(windowParameters, openglParameters, rendererParameters);

#endif

    bool finished = false;
    do
    {
        params = std::make_shared<CombinedParams>(LoadParamsHybrid(argc, argv));
        //    console << "Train Config: " << config_file << std::endl;
        //    SAIGA_ASSERT(std::filesystem::exists(config_file));
        if (params->train_params.random_seed == 0)
        {
            std::cout << "generating random seed..." << std::endl;
            params->train_params.random_seed = Random::generateTimeBasedSeed();
        }

        {
            std::cout << "Using Random Seed: " << params->train_params.random_seed << std::endl;
            Random::setSeed(params->train_params.random_seed);
            torch::manual_seed(params->train_params.random_seed * 937545);
        }
        params->Check();
        console << "torch::cuda::cudnn_is_available() " << torch::cuda::cudnn_is_available() << std::endl;
        std::filesystem::create_directories("experiments/");
        {
            // transfer image folder to local tmp folder (mostly for clusters)
            // images should be in scene folder
            if (!params->train_params.temp_image_dir.empty())
            {
                std::cout << "Copy scene to local temp folder" << std::endl;
                std::string job_id_sub_folder = "_x_/";
                if (std::getenv("SLURM_JOBID") != nullptr)
                {
                    job_id_sub_folder = "/_" + std::string(std::getenv("SLURM_JOBID")) + "_/";
                }
                std::filesystem::create_directory(params->train_params.temp_image_dir + job_id_sub_folder);
                for (int i = 0; i < params->train_params.scene_names.size(); ++i)
                {
                    std::string scene_name = params->train_params.scene_names[i];
                    std::string path_to_sc = params->train_params.scene_base_dir + scene_name;

                    std::string path_to_tmp = params->train_params.temp_image_dir + job_id_sub_folder + scene_name;

                    std::cout << "Copy " << path_to_sc << " to " << path_to_tmp << std::endl;

                    std::filesystem::remove_all(path_to_tmp);
                    std::filesystem::copy(path_to_sc, path_to_tmp, std::filesystem::copy_options::recursive);

                    {
                        std::string file_ini = path_to_tmp + "/dataset.ini";
                        SAIGA_ASSERT(std::filesystem::exists(file_ini));
                        auto dataset_params = SceneDatasetParams(file_ini);

                        if (!std::filesystem::exists(path_to_tmp + "/images/"))
                            std::filesystem::copy(dataset_params.image_dir, path_to_tmp + "/images/",
                                                  std::filesystem::copy_options::recursive);
                        // std::filesystem::remove_all(path_to_tmp+ "/images/");
                        std::cout << "replace image dir with " << path_to_tmp + "/images/" << std::endl;
                        dataset_params.image_dir = path_to_tmp + "/images/";
                        if (params->train_params.use_image_masks)
                        {
                            if (!std::filesystem::exists(path_to_tmp + "/masks/"))
                                std::filesystem::copy(dataset_params.mask_dir, path_to_tmp + "/masks/",
                                                      std::filesystem::copy_options::recursive);
                            // std::filesystem::remove_all(path_to_tmp+ "/images/");
                            std::cout << "replace mask dir with " << path_to_tmp + "/masks/" << std::endl;
                            dataset_params.mask_dir = path_to_tmp + "/masks/";
                        }

                        std::filesystem::remove(file_ini);
                        dataset_params.Save(file_ini);
                    }
                }
                params->train_params.scene_base_dir = params->train_params.temp_image_dir + job_id_sub_folder;
                std::cout << "Finished copying" << std::endl;
            }
        }

        {
            // signal handler for cluster
            std::signal(SIGTERM, call_handler);
            std::signal(SIGINT, call_handler);
#ifdef __linux__
            std::signal(SIGHUP, call_handler);
#endif
        }


        {
            NeuralTrainer trainer;
            finished = trainer.run();
        }
    } while (!finished);

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
