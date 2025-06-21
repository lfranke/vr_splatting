//
// Created by linus on 19.03.24.
//
#include "TrainScene.h"
extern std::shared_ptr<CombinedParams> params;
extern std::string full_experiment_dir;
extern torch::Device device;
//#undef LIVE_VIEWER

#ifdef LIVE_VIEWER
#    include "opengl/TrainLiveViewer.h"
extern std::shared_ptr<TrainViewer> viewer;
#endif


inline std::string EncodeImageToString(const Image& img, std::string type)
{
    auto data = img.saveToMemory(type);

    std::string result;
    result.resize(data.size());

    memcpy(result.data(), data.data(), data.size());
    return result;
}



torch::Tensor CropMask(int h, int w, int border)
{
    // Create a center crop mask so that the border is valued less during training.
    TemplatedImage<unsigned char> target_mask(h, w);
    target_mask.makeZero();

    int b     = border;
    auto crop = target_mask.getImageView().subImageView(b, b, h - b * 2, w - b * 2);
    crop.set(255);

    return ImageViewToTensor<unsigned char>(target_mask, true).unsqueeze(0);
}

TrainScene::TrainScene(std::vector<std::string> scene_dirs)
{
    for (int i = 0; i < scene_dirs.size(); ++i)
    {
        auto scene = std::make_shared<SceneData>(params->train_params.scene_base_dir + scene_dirs[i]);

        if (params->points_adding_params.dont_use_initial_pointcloud)
        {
            scene->point_cloud.position.clear();
            scene->point_cloud.color.clear();
            scene->point_cloud.data.clear();
            scene->point_cloud.normal.clear();
        }
        original_pc_size = scene->point_cloud.position.size();

        if (params->points_adding_params.scene_add_initially_random_points > 0)
        {
            AABB custom_aabb = scene->dataset_params.aabb;

            std::cout << "Adding random uniform points, "
                      << params->points_adding_params.scene_add_initially_random_points << " in bounding box "
                      << custom_aabb << std::endl;
            scene->AddRandomPoints(params->points_adding_params.scene_add_initially_random_points, custom_aabb);
        }

        // 1.  Separate indices
        auto all_indices                   = scene->Indices();
        auto [train_indices, test_indices] = params->train_params.Split(all_indices);

#ifdef LIVE_VIEWER
        std::cout << all_indices.size() << std::endl;
        viewer->setDataloaderMaxImages(all_indices.size());
        viewer->setupCamera(scene);
#endif

        if (std::filesystem::exists(params->train_params.split_index_file_train))
        {
            train_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_train);
        }


        if (std::filesystem::exists(params->train_params.split_index_file_test))
        {
            test_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_test);
        }


        if (params->train_params.duplicate_train_factor > 1)
        {
            // this multiplies the epoch size
            // increases performance for small epoch sizes
            auto cp = train_indices;
            for (int i = 1; i < params->train_params.duplicate_train_factor; ++i)
            {
                train_indices.insert(train_indices.end(), cp.begin(), cp.end());
            }
        }


        {
            std::ofstream strm(full_experiment_dir + "/train_indices_" + scene->scene_name + ".txt");
            for (auto i : train_indices)
            {
                strm << i << "\n";
            }
            std::ofstream strm2(full_experiment_dir + "/test_indices_" + scene->scene_name + ".txt");
            for (auto i : test_indices)
            {
                strm2 << i << "\n";
            }
        }

        std::cout << "Train(" << train_indices.size() << "): " << array_to_string(train_indices, ' ') << std::endl;
        std::cout << "Test(" << test_indices.size() << "): " << array_to_string(test_indices, ' ') << std::endl;

        PerSceneData scene_data;

        scene_data.not_training_indices = all_indices;
        for (auto i : train_indices)
        {
            auto it = std::find(scene_data.not_training_indices.begin(), scene_data.not_training_indices.end(), i);
            if (it != scene_data.not_training_indices.end())
            {
                scene_data.not_training_indices.erase(it);
            }
        }
        train_cropped_samplers.push_back(CroppedSampler(scene, train_indices));
        test_cropped_samplers.push_back(CroppedSampler(scene, test_indices));
        eval_samplers.push_back(FullSampler(scene, train_indices));
        test_samplers.push_back(FullSampler(scene, test_indices));


        if (params->train_params.train_mask_border > 0)
        {
            int i = 0;
            for (auto dims : test_samplers.back().image_size_crop)
            {
                int w              = dims.x();
                int h              = dims.y();
                torch::Tensor mask = CropMask(h, w, params->train_params.train_mask_border).to(device);
                TensorToImage<unsigned char>(mask).save(full_experiment_dir + "/eval_mask_" + scene->scene_name + "_" +
                                                        std::to_string(i) + ".png");
                scene_data.eval_crop_mask.push_back(mask);
                ++i;
            }
        }

        auto ns = std::make_shared<NeuralScene>(scene, params);
        if (params->train_params.noise_intr_k > 0 || params->train_params.noise_intr_d > 0)
        {
            scene->AddIntrinsicsNoise(params->train_params.noise_intr_k, params->train_params.noise_intr_d);

            torch::NoGradGuard ngg;
            auto intrinsics2 = IntrinsicsModule(scene);
            intrinsics2->to(device);
            ns->poses->to(device);

            PrintTensorInfo(ns->intrinsics->intrinsics);
            PrintTensorInfo(intrinsics2->intrinsics);
            ns->intrinsics->intrinsics.set_(intrinsics2->intrinsics);
        }


        if (params->train_params.noise_pose_r > 0 || params->train_params.noise_pose_t > 0)
        {
            scene->AddPoseNoise(radians(params->train_params.noise_pose_r), params->train_params.noise_pose_t / 1000.);

            torch::NoGradGuard ngg;
            auto poses2 = PoseModule(scene);
            poses2->to(device);
            ns->poses->to(device);

            PrintTensorInfo(ns->poses->poses_se3);
            PrintTensorInfo(poses2->poses_se3);
            ns->poses->poses_se3.set_(poses2->poses_se3);
        }

        if (params->train_params.noise_point > 0)
        {
            std::cout << "Adding point noise of " << params->train_params.noise_point << " on positions:" << std::endl;
            torch::NoGradGuard ngg;
            auto noise =
                torch::normal(0, params->train_params.noise_point, ns->point_cloud_cuda->t_position.sizes()).to(device);
            noise.slice(1, 3, 4).zero_();

            ns->point_cloud_cuda->t_position += noise;
        }


        if (params->train_params.noise_point_size > 0)
        {
            std::cout << "Adding point size noise of " << params->train_params.noise_point_size << std::endl;

            torch::NoGradGuard ngg;
            auto noise =
                torch::normal(0, params->train_params.noise_point_size, ns->point_cloud_cuda->t_point_size.sizes())
                    .to(device);

            ns->point_cloud_cuda->t_point_size += noise;
        }



        scene_data.scene = ns;

        data.push_back(scene_data);
    }
}
SceneDataTrainSampler TrainScene::CroppedSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
{
    ivec2 crop(params->train_params.train_crop_size, params->train_params.train_crop_size);

    SceneDataTrainSampler sampler(scene, indices, params->train_params.train_use_crop, crop,
                                  params->train_params.inner_batch_size, params->train_params.use_image_masks,
                                  params->train_params.crop_rotation, params->train_params.crop_max_dis_center,
                                  params->train_params.warmup_epochs, false);
    sampler.min_max_zoom(0)   = params->train_params.min_zoom * scene->dataset_params.render_scale;
    sampler.min_max_zoom(1)   = params->train_params.max_zoom * scene->dataset_params.render_scale;
    sampler.prefere_border    = params->train_params.crop_prefere_border;
    sampler.inner_sample_size = params->train_params.inner_sample_size;
    sampler.sample_gaussian   = params->train_params.crop_gaussian_sample;
    std::cout << "cropped sampler " << sampler.image_size_input[0].x() << "x" << sampler.image_size_input[0].y()
              << " to " << sampler.image_size_crop[0].x() << " x " << sampler.image_size_crop[0].y() << " render scale "
              << scene->dataset_params.render_scale << std::endl;
    return sampler;
}

SceneDataTrainSampler TrainScene::FullSampler(std::shared_ptr<SceneData> scene, std::vector<int> indices)
{
    int w = scene->scene_cameras.front().w * scene->dataset_params.render_scale;
    int h = scene->scene_cameras.front().h * scene->dataset_params.render_scale;

    int max_eval_size = iAlignUp(params->train_params.max_eval_size, 32);

    std::cout << "full sampler " << w << "x" << h << " render scale " << scene->dataset_params.render_scale
              << std::endl;

    int min_scene_size = std::min(w, h);
    if (min_scene_size > max_eval_size && params->train_params.train_use_crop)
    {
        w = std::min(w, max_eval_size);
        h = std::min(h, max_eval_size);

        SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks, false,
                                  -1);
        sdf.random_zoom        = true;
        sdf.min_max_zoom(0)    = max_eval_size / double(min_scene_size);
        sdf.min_max_zoom(1)    = max_eval_size / double(min_scene_size);
        sdf.random_translation = false;
        return sdf;
    }
    else if (scene->dataset_params.render_scale != 1)
    {
        SceneDataTrainSampler sdf(scene, indices, true, ivec2(w, h), 1, params->train_params.use_image_masks, false,
                                  -1);
        sdf.random_zoom        = true;
        sdf.min_max_zoom(0)    = scene->dataset_params.render_scale;
        sdf.min_max_zoom(1)    = scene->dataset_params.render_scale;
        sdf.random_translation = false;
        return sdf;
    }
    else
    {
        SceneDataTrainSampler sdf(scene, indices, false, ivec2(-1, -1), 1, params->train_params.use_image_masks, false,
                                  -1);
        sdf.random_translation = false;
        return sdf;
    }
}

//  typedef std::unique_ptr<torch::data::StatelessDataLoader<TorchSingleSceneDataset, torch::MultiDatasetSampler>>
//      dataloader_t;

dataloader_type_t TrainScene::DataLoader(std::vector<SceneDataTrainSampler>& train_cropped_samplers, bool train, int& n)
{
    std::vector<uint64_t> sizes;
    for (auto& t : train_cropped_samplers)
    {
        sizes.push_back(t.Size());
    }

    int batch_size  = train ? params->train_params.batch_size : 1;
    int num_workers = train ? params->train_params.num_workers_train : params->train_params.num_workers_eval;
    bool shuffle    = train ? params->train_params.shuffle_train_indices : false;
    auto options    = torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(false).workers(num_workers);

    auto sampler = torch::MultiDatasetSampler(sizes, options.batch_size(), shuffle);
    n            = sampler.NumImages();

    dataloader_type_t d_l =
        torch::data::make_data_loader(TorchSingleSceneDataset(train_cropped_samplers), std::move(sampler), options);

    return d_l;
}

void TrainScene::Load(torch::DeviceType device, int scene)
{
    if (scene == current_scene) return;
    Unload();
    std::cout << "Load " << scene << std::endl;
    current_scene = scene;
    data[current_scene].scene->to(device);
}

void TrainScene::Unload(bool always_unload)
{
    if (params->train_params.keep_all_scenes_in_memory && !always_unload) return;
    if (current_scene != -1)
    {
        std::cout << "Unload " << current_scene << std::endl;
        data[current_scene].scene->to(torch::kCPU);
    }
    current_scene = -1;
}

void TrainScene::SetScene(int id)
{
    current_scene = id;
}

void TrainScene::Train(int epoch, bool v)
{
    SAIGA_ASSERT(current_scene != -1);
    data[current_scene].scene->Train(epoch, v);
}

void TrainScene::StartEpoch()
{
    for (auto& sd : data)
    {
        sd.epoch_loss = {};
    }
}
