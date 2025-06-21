﻿/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "SceneData.h"

#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/sophus/Sophus.h"
#include "saiga/core/util/BinaryFile.h"

#include "rendering/PointBlending.h"


Saiga::Camera FrameData::GLCamera() const
{
    Saiga::Camera cam;
    cam.setModelMatrix(OpenglModel());

    cam.updateFromModel();
    cam.zNear = 0.1;
    cam.zFar  = 1000;

    cam.proj = CVCamera2GLProjectionMatrix(K.matrix(), ivec2(w, h), cam.zNear, cam.zFar);
    return cam;
}

std::pair<vec2, float> FrameData::Project3(vec3 wp) const
{
    vec3 p    = pose.inverse().cast<float>() * wp;
    vec2 np   = p.head<2>() / p(2);
    vec2 disp = distortNormalizedPoint(np, distortion);
    // vec2 disp = np;
    vec2 ip = K.normalizedToImage(disp);
    return {ip, p(2)};
}

SceneData::SceneData(std::string _scene_path, bool full_preprocess)
{
    std::cout << "Scene path: " << _scene_path << std::endl;
    scene_path = std::filesystem::canonical(_scene_path).string();
    scene_name = std::filesystem::path(scene_path).filename().string();
    std::cout << "====================================" << std::endl;
    std::cout << "Scene Loaded" << std::endl;
    std::cout << "  Name       " << scene_name << std::endl;
    std::cout << "  Path       " << scene_path << std::endl;
    SAIGA_ASSERT(!scene_name.empty());

    {
        std::string file_ini = scene_path + "/dataset.ini";
        SAIGA_ASSERT(std::filesystem::exists(file_ini));
        dataset_params = SceneDatasetParams(file_ini);
    }

    for (auto ini_file : dataset_params.camera_files)
    {
        std::string file_ini = scene_path + "/" + ini_file;
        SAIGA_ASSERT(std::filesystem::exists(file_ini));
        scene_cameras.push_back(SceneCameraParams(file_ini));
        // scene_cameras.back().camera_model_type = dataset_params.camera_model;
        CameraModel camMod =
            scene_cameras.back().ocam.poly_cam2world.size() > 0 ? CameraModel::OCAM : CameraModel::PINHOLE_DISTORTION;
        scene_cameras.back().camera_model_type = camMod;

        std::cout << "CAM model: "
                  << (scene_cameras.back().ocam.poly_cam2world.size() > 0 ? "CameraModel::OCAM"
                                                                          : "CameraModel::PINHOLE_DISTORTION")
                  << std::endl;
    }
    SAIGA_ASSERT(!scene_cameras.empty());

    file_dataset_base           = scene_path;
    file_intrinsics             = scene_path + "/intrinsics/intrinsic_color.txt";
    file_point_cloud            = scene_path + "/" + dataset_params.file_point_cloud;
    file_point_cloud_compressed = scene_path + "/" + dataset_params.file_point_cloud_compressed;
    file_pose                   = scene_path + "/poses.txt";
    file_exposure               = scene_path + "/exposure.txt";
    file_white_balance          = scene_path + "/white_balance.txt";
    file_image_names            = scene_path + "/images.txt";
    file_mask_names             = scene_path + "/masks.txt";
    file_camera_indices         = scene_path + "/camera_indices.txt";

#define COMPRESS
#ifdef COMPRESS
    if (std::filesystem::exists(file_point_cloud_compressed))
    {
        point_cloud.LoadCompressed(file_point_cloud_compressed);
        // UnifiedModel(point_cloud).Save(scene_path + "/point_cloud_initial_preprocess.ply");
        SAIGA_ASSERT(point_cloud.NumVertices() > 0);
    }
    else if (std::filesystem::exists(file_point_cloud))
    {
        std::cout << ">> Loading initial PLY point cloud and preprocessing it (done only once)" << std::endl;
        std::cout << ">> This can take a while..." << std::endl;
        point_cloud = Saiga::UnifiedModel(file_point_cloud).mesh[0];
        SAIGA_ASSERT(point_cloud.NumVertices() > 0);
        if (!point_cloud.HasColor())
        {
            std::cout << "No Point Color found... Setting to white!" << std::endl;
            point_cloud.SetVertexColor(vec4(1, 1, 1, 1));
        }
        if (full_preprocess)
        {
            auto box = point_cloud.BoundingBox();
            std::cout << "Bounding Box point cloud: " << std::endl << box << std::endl;

            SAIGA_ASSERT(box.min.array().allFinite());
            SAIGA_ASSERT(box.max.array().allFinite());
            SAIGA_ASSERT(box.maxSize() < 1e20);

            // some initial processing
            point_cloud.RemoveDoubles(0.0001);
            RemoveLonelyPoints(5, 0.02);
            // RemoveClosePoints(0.0005);

            RemovePointsInCloseArea(dataset_params.point_factor);
            point_cloud.ReorderMorton64();
            point_cloud.RandomBlockShuffle(256);
        }
        ComputeRadius();
        if (full_preprocess)
        {
            SortBlocksByRadius(256);

            std::cout << "Save point cloud " << file_point_cloud_compressed << std::endl;
        }
        point_cloud.SaveCompressed(file_point_cloud_compressed);
        if (full_preprocess)
        {
            UnifiedModel(point_cloud).Save(scene_path + "/point_cloud_initial_preprocess.ply");
        }
    }


    if (!point_cloud.HasData() || point_cloud.data[0](0) == 0)
    {
        ComputeRadius();
        std::cout << "Save point cloud " << file_point_cloud_compressed << std::endl;
        point_cloud.SaveCompressed(file_point_cloud_compressed);
    }
    // UnifiedModel(point_cloud).Save(scene_path + "/point_cloud_initial_preprocess.ply");


    if (!point_cloud.HasNormal())
    {
        std::cout << "  No normals: fill with data" << std::endl;
        std::vector<vec3> normals;
        for (auto p : point_cloud.position)
        {
            normals.push_back(p.normalized());
        }
        point_cloud.normal = normals;
    }
#else

    point_cloud = Saiga::UnifiedModel(file_point_cloud).mesh[0];
    SAIGA_ASSERT(point_cloud.NumVertices() > 0);
    if (!point_cloud.HasColor())
    {
        std::cout << "No Point Color found... Setting to white!" << std::endl;
        point_cloud.SetVertexColor(vec4(1, 1, 1, 1));
    }
    ComputeRadius();

    std::cout << point_cloud.color[0] << std::endl;
#endif

    std::cout << "  Points     " << point_cloud.NumVertices() << std::endl;
    std::cout << "  Colors     " << point_cloud.HasColor() << std::endl;
    std::cout << "  Normals    " << point_cloud.HasNormal() << std::endl;


    auto read_pose_file = [](std::string pose_file, std::vector<Sophus::SE3d>& poses)
    {
        std::ifstream strm(pose_file);

        std::string line;
        while (std::getline(strm, line))
        {
            std::stringstream sstream(line);

            Quat q;
            Vec3 t;

            sstream >> q.x() >> q.y() >> q.z() >> q.w() >> t.x() >> t.y() >> t.z();
            // sstream >> q.w() >> q.x() >> q.y() >> q.z() >> t.x() >> t.y() >> t.z();

            poses.push_back({q, t});
            // poses.back() = poses.back().inverse();
        }
    };

    std::vector<Sophus::SE3d> poses;
    if (std::filesystem::exists(file_pose))
    {
        read_pose_file(file_pose, poses);
    }
    else
    {
        SAIGA_EXIT_ERROR("the pose file is required!");

        // BinaryFile strm(scene_path + "/poses.dat", std::ios_base::in);
        // strm >> poses;
    }

    std::vector<float> exposures;
    if (std::filesystem::exists(file_exposure))
    {
        std::ifstream strm(file_exposure);

        std::string line;
        while (std::getline(strm, line))
        {
            std::stringstream sstream(line);
            double ex;
            sstream >> ex;
            exposures.push_back(ex);
        }
        std::cout << "  Avg. EV  " << Statistics(exposures).mean << std::endl;
    }
    else
    {
        // BinaryFile strm(scene_path + "/exposure.dat", std::ios_base::in);
        // strm >> exposures;
    }

    std::vector<vec3> wbs;
    if (std::filesystem::exists(file_white_balance))
    {
        std::ifstream strm(file_white_balance);

        std::string line;
        while (std::getline(strm, line))
        {
            std::stringstream sstream(line);
            vec3 wb;
            sstream >> wb.x() >> wb.y() >> wb.z();

            wbs.push_back(wb);
        }
    }

    std::vector<std::string> images;
    std::vector<std::string> masks;


    if (std::filesystem::exists(file_image_names))
    {
        std::ifstream strm(file_image_names);

        std::string line;
        while (std::getline(strm, line))
        {
            images.push_back(line);
        }
    }

    if (std::filesystem::exists(file_mask_names))
    {
        std::ifstream strm(file_mask_names);

        std::string line;
        while (std::getline(strm, line))
        {
            masks.push_back(line);
        }
    }

    std::vector<int> camera_indices;
    if (std::filesystem::exists(file_camera_indices))
    {
        std::ifstream strm(file_camera_indices);

        std::string line;
        while (std::getline(strm, line))
        {
            camera_indices.push_back(to_int(line));
        }
    }


    int n_frames = std::max({images.size(), poses.size()});
    frames.resize(n_frames);

    std::cout << "  Num Images " << frames.size() << std::endl;
    std::cout << "  Num Cameras " << scene_cameras.size() << std::endl;

    SAIGA_ASSERT(poses.empty() || poses.size() == frames.size());
    SAIGA_ASSERT(images.empty() || images.size() == frames.size());
    SAIGA_ASSERT(masks.empty() || masks.size() == frames.size());
    SAIGA_ASSERT(exposures.empty() || exposures.size() == frames.size());
    SAIGA_ASSERT(wbs.empty() || wbs.size() == frames.size());
    SAIGA_ASSERT(camera_indices.empty() || camera_indices.size() == frames.size());

    for (int i = 0; i < frames.size(); ++i)
    {
        auto& fd        = frames[i];
        fd.image_index  = i;
        fd.camera_index = 0;

        if (!camera_indices.empty()) fd.camera_index = camera_indices[i];
        if (!poses.empty()) fd.pose = poses[i];
        if (!exposures.empty()) fd.exposure_value = exposures[i];
        if (!wbs.empty()) fd.white_balance = wbs[i];
        if (!images.empty()) fd.target_file = images[i];
        if (!masks.empty()) fd.mask_file = masks[i];

        auto cam      = scene_cameras[fd.camera_index];
        fd.K          = cam.K;
        fd.distortion = cam.distortion;
        fd.w          = cam.w;
        fd.h          = cam.h;
    }

    std::string extra_eval_poses = scene_path + "/extra_eval_poses.txt";
    std::string extra_eval_cam   = scene_path + "/extra_eval_cam.ini";
    if (std::filesystem::exists(extra_eval_poses) && std::filesystem::exists(extra_eval_cam))
    {
        // extra cameras for eval exist

        std::vector<Sophus::SE3d> extra_poses;
        if (std::filesystem::exists(extra_eval_poses))
        {
            read_pose_file(extra_eval_poses, extra_poses);
        }
        if (std::filesystem::exists(extra_eval_cam))
        {
            SceneCameraParams extra_eval_scene_cam_param;
            {
                std::string file_ini = extra_eval_cam;
                SAIGA_ASSERT(std::filesystem::exists(file_ini));
                extra_eval_scene_cam_param                   = (SceneCameraParams(file_ini));
                extra_eval_scene_cam_param.camera_model_type = CameraModel::PINHOLE_DISTORTION;
                extra_eval_cameras.push_back(extra_eval_scene_cam_param);
            }
        }

        extra_eval_frames.resize(extra_poses.size());

        for (int i = 0; i < extra_eval_frames.size(); ++i)
        {
            auto& fd        = extra_eval_frames[i];
            fd.image_index  = i;
            fd.camera_index = 0;

            fd.pose = extra_poses[i];

            auto cam      = extra_eval_cameras[0];
            fd.K          = cam.K;
            fd.distortion = cam.distortion;
            fd.w          = cam.w;
            fd.h          = cam.h;
        }
    }
#if 0
    if (dataset_params.aabb.maxSize() < 0.0001)
    {
        double percent_int = 0.95;
        std::cout << "Compute scene importance bounding box as " << percent_int * 100
                  << "% of points interval around center of mass" << std::endl;
        Vec3 center_of_mass(0.0, 0.0, 0.0);
        {
            ScopedTimerPrintLine tim("Compute center of mass");
            double fac = double(1) / double(point_cloud.NumVertices());
            for (int i = 0; i < point_cloud.NumVertices(); ++i)
            {
                Vec3 p = point_cloud.position[i].cast<double>();
                center_of_mass += p * fac;
            }
            std::cout << "center of mass:" << center_of_mass << std::endl;
        }

        AABB aabb_percentile;
        {
            std::vector<std::pair<float, int>> distance_pos_pairs(point_cloud.NumVertices());
            {
                ScopedTimerPrintLine tim("Build range vec");

                for (int i = 0; i < point_cloud.NumVertices(); ++i)
                {
                    distance_pos_pairs[i] = std::make_pair(length((point_cloud.position[i] - center_of_mass)), i);
                }
            }
            {
                ScopedTimerPrintLine tim("Sort range vec");

                std::sort(distance_pos_pairs.begin(), distance_pos_pairs.end(),
                          [](std::pair<float, int> a, std::pair<float, int> b) { return a.first < b.first; });
            }
            {
                ScopedTimerPrintLine tim("Extend box");

                for (int i = 0; i < std::ceil(percent_int * point_cloud.position.size()); ++i)
                {
                    vec3 pos_in_b = point_cloud.position[distance_pos_pairs[i].second];
                    aabb_percentile.growBox(pos_in_b);
                }
            }
        }
        std::cout << "Box: " << aabb_percentile << std::endl;
        dataset_params.aabb = aabb_percentile;
    }
#endif
    // exit(1);
    std::cout << "====================================" << std::endl;
}

void SceneData::Save(bool extended_save)
{
    ScopedTimerPrintLine tim("SceneData::Save");
    SAIGA_ASSERT(!frames.empty());

    std::filesystem::create_directories(file_dataset_base);


    {
        std::ofstream strm2(file_exposure, std::ios_base::out);
        for (auto f : frames)
        {
            strm2 << std::scientific << std::setprecision(15);
            strm2 << f.exposure_value << "\n";
        }
    }

    {
        std::ofstream ostream1(file_white_balance, std::ios_base::out);
        for (auto& fd : frames)
        {
            vec3 V = fd.white_balance;
            ostream1 << V(0) << " " << V(1) << " " << V(2) << "\n";
        }
    }

    {
        std::vector<Sophus::SE3d> posesd;
        for (auto f : frames)
        {
            SE3 p = f.pose;
            posesd.push_back(p);
        }
        SavePoses(posesd, file_pose);

        if (extended_save)
        {
            std::ofstream ostream1(scene_path + "/poses_view_matrix.txt");
            for (auto m : posesd)
            {
                auto V = m.inverse().matrix();
                ostream1 << V << "\n";
            }
        }
    }

    if (extended_save)
    {
        std::ofstream ostream2(scene_path + "/K_matrix.txt");
        for (auto m : scene_cameras)
        {
            auto K = m.K.matrix();
            ostream2 << K << "\n";
        }
    }


    {
        std::ofstream ostream1(file_image_names);
        std::ofstream ostream2(file_mask_names);
        for (auto f : frames)
        {
            ostream1 << f.target_file << std::endl;
            ostream2 << f.mask_file << std::endl;
        }
    }


    point_cloud.SaveCompressed(file_point_cloud_compressed);
}

void SceneData::SavePoses(std::vector<SE3> poses, std::string file)
{
    std::ofstream strm2(file, std::ios_base::out);
    for (auto p : poses)
    {
        Quat q = p.unit_quaternion();
        Vec3 t = p.translation();
        strm2 << std::scientific << std::setprecision(15);
        strm2 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << t.x() << " " << t.y() << " " << t.z()
              << "\n";
    }
}

void SceneData::AddPointNoise(float sdev)
{
    if (sdev > 0)
    {
        for (auto& p : point_cloud.position)
        {
            p += Random::MatrixGauss<vec3>(0, sdev);
        }
    }
}

void SceneData::AddPoseNoise(float sdev_rot, float sdev_trans)
{
    if (sdev_rot > 0)
    {
        std::cout << "Adding rotational noise: " << degrees(sdev_rot) << " deg" << std::endl;
        for (auto& p : frames)
        {
            Quat q = p.pose.unit_quaternion();
            q      = Sophus::SO3d::exp(Random::MatrixGauss<Vec3>(0, sdev_rot)).unit_quaternion() * q;
            p.pose.setQuaternion(q);
        }
    }

    if (sdev_trans > 0)
    {
        std::cout << "Adding translational noise: " << sdev_trans * 1000.0 << " mm." << std::endl;
        for (auto& p : frames)
        {
            p.pose.translation() += Random::MatrixGauss<Vec3>(0, sdev_trans);
        }
    }
}


void SceneData::AddIntrinsicsNoise(float sdev_K, float sdev_dist)
{
    vec5 noise_K;
    vec8 noise_D;
    float one_over_cams = 1.f / float(scene_cameras.size());

    for (auto& cam : scene_cameras)
    {
        if (sdev_K > 0)
        {
            vec5 dk    = cam.K.coeffs();
            vec5 noise = Random::MatrixGauss<vec5>(0, sdev_K);
            noise(4) *= 0.1;

            noise_K += abs(noise) * (one_over_cams);
            dk += noise;
            cam.K = dk;
        }

        if (sdev_dist > 0)
        {
            vec8 dd    = cam.distortion.Coeffs();
            vec8 noise = Random::MatrixGauss<vec8>(0, sdev_dist);

            // k3
            noise(2) *= 0.25;

            // k4 - 6
            noise(3) *= 0.1;
            noise(4) *= 0.1;
            noise(5) *= 0.1;

            // tangential distortion
            noise(6) *= 0.1;
            noise(7) *= 0.1;

            noise_D += abs(noise) * one_over_cams;
            dd += noise;
            cam.distortion = dd;
        }
    }
    std::cout << "Added noise to intrinsics: on average K:" << noise_K.array()
              << "; on average distortion: " << noise_D.array() << std::endl;
}

Saiga::UnifiedMesh SceneData::OutlierPointCloud(int n, float distance)
{
    n = std::min(n, point_cloud.NumVertices());

    auto bb               = point_cloud.BoundingBox();
    auto [center, radius] = bb.BoundingSphere();

    radius += radius * distance;

    UnifiedMesh result;
    for (int i = 0; i < n; ++i)
    {
        vec3 p = center + Random::sphericalRand(radius).cast<float>();
        vec3 n = (center - p).normalized();
        vec4 c = vec4(1, 0, 0, 1);
        result.position.push_back(p);
        result.color.push_back(c);
        result.normal.push_back(n);
    }
    return result;
}

void SceneData::ComputeRadius(int n)
{
    ComputeRadius(point_cloud, n);
    /*
    point_cloud.data.resize(point_cloud.NumVertices(), vec4::Zero());
    KDTree<3, vec3> tree;

    {
        ScopedTimerPrintLine tim("Build KDTree");
        tree = KDTree<3, vec3>(point_cloud.position);
    }
    {
        ScopedTimerPrintLine tim("Compute radius");

        std::vector<float> nearest_distance_squared(point_cloud.NumVertices());
#pragma omp parallel for
        for (int i = 0; i < point_cloud.position.size(); ++i)
        {
            // N includes the own point here!
            auto v  = tree.KNearestNeighborSearch(point_cloud.position[i], n + 1);
            int idx = v.back();
            // SAIGA_ASSERT(idx != i);
            vec3 nn                = point_cloud.position[idx];
            float dis              = (point_cloud.position[i] - nn).norm();
            point_cloud.data[i](0) = dis;
        }
    }


    double average_dis = 0;
    for (int i = 0; i < point_cloud.position.size(); ++i)
    {
        average_dis += point_cloud.data[i](0);
    }
    average_dis /= point_cloud.position.size();
    std::cout << "average_dis = " << average_dis << std::endl;

    {
        ScopedTimerPrintLine tim("Computing random range sample");
        for (auto& d : point_cloud.data)
        {
            d[2] = sqrt(1 / Random::sampleDouble(0, 1));
            d[3] = d[0] * d[2];
        }
    }*/
}

void SceneData::ComputeRadius(Saiga::UnifiedMesh& pc, int n)
{
    pc.data.resize(pc.NumVertices(), vec4::Zero());
    KDTree<3, vec3> tree;

    {
        ScopedTimerPrintLine tim("Build KDTree");
        tree = KDTree<3, vec3>(pc.position);
    }
    {
        ScopedTimerPrintLine tim("Compute radius");

        std::vector<float> nearest_distance_squared(pc.NumVertices());
#pragma omp parallel for
        for (int i = 0; i < pc.position.size(); ++i)
        {
            // N includes the own point here!
            auto v  = tree.KNearestNeighborSearch(pc.position[i], n + 1);
            int idx = v.back();
            // SAIGA_ASSERT(idx != i);
            vec3 nn       = pc.position[idx];
            float dis     = (pc.position[i] - nn).norm();
            pc.data[i](0) = dis;
        }
    }


    double average_dis = 0;
    for (int i = 0; i < pc.position.size(); ++i)
    {
        average_dis += pc.data[i](0);
    }
    average_dis /= pc.position.size();
    std::cout << "average_dis = " << average_dis << std::endl;

    {
        ScopedTimerPrintLine tim("Computing random range sample");
        for (auto& d : pc.data)
        {
            d[2] = sqrt(1 / Random::sampleDouble(0, 1));
            d[3] = d[0] * d[2];
        }
    }
}

void SceneData::AddRandomPoints(int num_points_added, AABB custom_bounding_box)
{
    ScopedTimerPrintLine tim("AddRandomPoints");

    // std::cout << "SceneData setup: add " << num_points_added << " points" << std::endl;
    int points_to_add = num_points_added;
    //    KDTree<3, vec3> tree(point_cloud.position);

    // point_cloud.position.resize(N + points_to_add);
    // point_cloud.normal.resize(N + points_to_add);
    // point_cloud.data.resize(N + points_to_add);
    // point_cloud.color.resize(N + points_to_add);

    auto bb = custom_bounding_box;

    if (custom_bounding_box.maxSize() < 0.1)
    {
        bb = point_cloud.BoundingBox();
    }

    for (int i = 0; i < points_to_add; ++i)
    {
        float r1    = float(rand()) / float(RAND_MAX);
        float r2    = float(rand()) / float(RAND_MAX);
        float r3    = float(rand()) / float(RAND_MAX);
        vec3 rand_p = {r1, r2, r3};
        // rand.setRandom();
        vec3 pos = rand_p.array() * (bb.max - bb.min).array() + bb.min.array();
        point_cloud.position.push_back(pos);
        point_cloud.normal.push_back(rand_p);
        point_cloud.data.push_back(vec4(0, 1, 0, 1));
        point_cloud.color.push_back(vec4(0, 1, 0, 0));
    }
    ComputeRadius();
}

void SceneData::AddRandomPoints(Saiga::UnifiedMesh& point_cloud, float fac)
{
    SAIGA_ASSERT(point_cloud.HasNormal());
    // SAIGA_ASSERT(point_cloud.HasColor());
    ScopedTimerPrintLine tim("AddRandomPoints");
    int N = point_cloud.NumVertices();

    int points_to_add = std::ceil(N * fac);
    //    KDTree<3, vec3> tree(point_cloud.position);

    // point_cloud.position.resize(N + points_to_add);
    // point_cloud.normal.resize(N + points_to_add);
    // point_cloud.data.resize(N + points_to_add);
    // point_cloud.color.resize(N + points_to_add);

    auto bb = point_cloud.BoundingBox();
    std::cout << "Scene bounding box: " << bb << std::endl;

    for (int i = 0; i < points_to_add; ++i)
    {
        float r1  = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float r2  = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float r3  = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        vec3 rand = {r1, r2, r3};
        // rand.setRandom();
        vec3 pos = rand.array() * (bb.max - bb.min).array() + bb.min.array();
        point_cloud.position.push_back(pos);
        point_cloud.normal.push_back(rand);
        point_cloud.data.push_back(vec4(0, 1, 0, 1));
        // point_cloud.color.push_back(vec4(0,1,0,0));
    }
    ComputeRadius(point_cloud);
}

void SceneData::AddPoints(Saiga::UnifiedMesh& point_cloud, std::vector<vec3> positions, std::vector<vec3> normals)
{
    // SAIGA_ASSERT(point_cloud.HasNormal());
    // SAIGA_ASSERT(point_cloud.HasColor());
    ScopedTimerPrintLine tim("AddRandomPoints");
    std::vector<vec4> placeholder_colors(positions.size(), vec4(1, 1, 1, 0));

    for (auto& n : normals)
    {
        if (length(n) > 0.01)
            n.normalized();
        else
            n = vec3(1, 0, 0);
    }

    point_cloud.position.insert(point_cloud.position.end(), positions.begin(), positions.end());
    point_cloud.normal.insert(point_cloud.normal.end(), normals.begin(), normals.end());
    point_cloud.color.insert(point_cloud.color.end(), placeholder_colors.begin(), placeholder_colors.end());
    std::vector<vec4> data(positions.size(), vec4(0, 1, 0, 1));
    point_cloud.data.insert(point_cloud.data.end(), data.begin(), data.end());

    ComputeRadius(point_cloud);
}

void SceneData::DuplicatePointsRand(int factor, float dis = 0.5)
{
    DuplicatePointsRand(point_cloud, factor, dis);
}

void SceneData::DuplicatePoints(int factor, float dis = 0.5)
{
    DuplicatePoints(point_cloud, factor, dis);
}

void SceneData::DuplicatePoints(Saiga::UnifiedMesh& pc, int factor, float dis = 0.5)
{
    SAIGA_ASSERT(pc.HasNormal());
    SAIGA_ASSERT(pc.HasColor());
    ScopedTimerPrintLine tim("DuplicatePoints");
    int N = pc.NumVertices();
    KDTree<3, vec3> tree(pc.position);

    pc.position.resize(N * factor);
    pc.normal.resize(N * factor);
    pc.data.resize(N * factor);
    pc.color.resize(N * factor);

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        auto p = pc.position[i];
        auto n = pc.normal[i];
        auto d = pc.data[i];
        auto c = pc.color[i];

        float r = d(0);

        for (int k = 0; k < (factor - 1); ++k)
        {
            vec3 best_p    = vec3(0, 0, 0);
            float best_dis = 0;

            for (int sam = 0; sam < 10; ++sam)
            {
                mat3 T = onb(n);

                // p += T.col(0) * Random::sampleDouble(0, r * 0.25) + T.col(1) * Random::sampleDouble(0, r * 0.25);

                vec2 rand = vec2(Random::sampleDouble(-1, 1), Random::sampleDouble(-1, 1)).normalized();
                vec3 t    = T.col(0) * rand.x() + T.col(1) * rand.y();
                vec3 p2   = p + dis * r * t * Random::sampleDouble(0.25, 2);

                float dis = (pc.position[tree.NearestNeighborSearch(p2)] - p).squaredNorm();
                if (dis > best_dis)
                {
                    best_p = p2;
                }
            }

            int new_index = i + (k + 1) * N;

            pc.position[new_index] = best_p;
            pc.normal[new_index]   = (n);
            pc.data[new_index]     = (d);
            pc.color[new_index]    = c;
        }
    }
    ComputeRadius(pc);
}


void SceneData::DuplicatePointsRand(Saiga::UnifiedMesh& pc, int factor, float dis, bool pointgrow)
{
    SAIGA_ASSERT(pc.HasNormal());
    SAIGA_ASSERT(pc.HasColor());
    ScopedTimerPrintLine tim("DuplicatePoints");
    int N = pc.NumVertices();
    KDTree<3, vec3> tree(pc.position);

    pc.position.resize(N * factor);
    pc.normal.resize(N * factor);
    pc.data.resize(N * factor);
    pc.color.resize(N * factor);

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        auto p = pc.position[i];
        auto n = pc.normal[i];
        auto d = pc.data[i];
        auto c = pc.color[i];

        float r = d(0);

        for (int k = 0; k < (factor - 1); ++k)
        {
            vec3 best_p    = vec3(0, 0, 0);
            float best_dis = 0;

            {
                // p += T.col(0) * Random::sampleDouble(0, r * 0.25) + T.col(1) * Random::sampleDouble(0, r * 0.25);

                vec3 rand = vec3(Random::sampleDouble(-1, 1), Random::sampleDouble(-1, 1), Random::sampleDouble(-1, 1))
                                .normalized();
                vec3 p2 = p + r * dis * rand * Random::sampleDouble(0.25, 2);

                {
                    best_p = p2;
                }
            }

            int new_index = i + (k + 1) * N;

            pc.position[new_index] = best_p;
            pc.normal[new_index]   = (n);
            pc.data[new_index]     = (d);
            if (pointgrow)
                pc.color[new_index] = vec4(1, 1, 1, 0);

            else
                pc.color[new_index] = c;
        }
    }
    ComputeRadius(pc);
}

void SceneData::DownsamplePoints(float factor)
{
    int n           = point_cloud.NumVertices();
    int n_remaining = n / factor;
    int n_remove    = n - n_remaining;

    auto remove_indices = Random::uniqueIndices(n_remove, n);
    point_cloud.EraseVertices(remove_indices);
}

void SceneData::RemoveLonelyPoints(int n, float dis_th)
{
    int bef = point_cloud.NumVertices();

    PointDistanceToCamera();

    KDTree<3, vec3> tree;
    std::vector<int> to_erase;

    {
        ScopedTimerPrintLine tim("Build KDTree");
        tree = KDTree<3, vec3>(point_cloud.position);
    }

    {
        ScopedTimerPrintLine tim("RemoveLonelyPoints");

        std::vector<float> point_dis(point_cloud.NumVertices());

#pragma omp parallel for
        for (int i = 0; i < point_cloud.position.size(); ++i)
        {
            // N includes the own point here!
            auto v  = tree.KNearestNeighborSearch(point_cloud.position[i], n + 1);
            int idx = v.back();
            SAIGA_ASSERT(idx != i);
            vec3 nn      = point_cloud.position[idx];
            float d      = (point_cloud.position[i] - nn).norm();
            point_dis[i] = d;
        }

        for (int i = 0; i < point_cloud.position.size(); ++i)
        {
            float d       = point_dis[i];
            float cam_dis = point_cloud.data[i](0);
            if (d / cam_dis > dis_th)
            {
                to_erase.push_back(i);
            }
        }
    }

    point_cloud.EraseVertices(to_erase);
    std::cout << "remove lonely dis " << dis_th << " Points " << bef << " -> " << point_cloud.NumVertices()
              << std::endl;
}


void SceneData::RemoveClosePoints(float dis_th)
{
    int bef = point_cloud.NumVertices();
    PointDistanceToCamera();

    KDTree<3, vec3> tree;

    {
        ScopedTimerPrintLine tim("Build KDTree");
        tree = KDTree<3, vec3>(point_cloud.position);
    }

    {
        ScopedTimerPrintLine tim("RemoveClosePoints");


        std::vector<int> to_merge(point_cloud.NumVertices());

        std::vector<int> to_erase;
        std::vector<int> valid(point_cloud.NumVertices(), 0);
        for (int i = 0; i < point_cloud.position.size(); ++i)
        {
            float distance = dis_th / (point_cloud.data[i](0));
            auto ps        = tree.RadiusSearch(point_cloud.position[i], distance);
            bool found     = false;
            to_merge[i]    = i;
            for (auto pi : ps)
            {
                if (valid[pi])
                {
                    to_erase.push_back(i);
                    to_merge[i] = pi;
                    found       = true;
                    break;
                }
            }
            if (!found)
            {
                valid[i] = true;
            }
        }
        point_cloud.EraseVertices(to_erase);
    }
    std::cout << "remove close z dis " << dis_th << " Points " << bef << " -> " << point_cloud.NumVertices()
              << std::endl;
}

void SceneData::RemovePointsInCloseArea(float dis_mm)
{
    RemovePointsInCloseArea(point_cloud, frames, dis_mm);
}


void SceneData::RemovePointsInCloseArea(Saiga::UnifiedMesh& point_cloud, std::vector<FrameData>& frames, float dis_mm)
{
    int bef = point_cloud.NumVertices();
    PointDistanceToCamera(point_cloud, frames);

    KDTree<3, vec3> tree;

    {
        ScopedTimerPrintLine tim("Build KDTree");
        tree = KDTree<3, vec3>(point_cloud.position);
    }

    {
        ScopedTimerPrintLine tim("RemovePointsInCloseArea");


        std::vector<int> to_merge(point_cloud.NumVertices());

        std::vector<int> to_erase;
        std::vector<int> valid(point_cloud.NumVertices(), 0);
        for (int i = 0; i < point_cloud.position.size(); ++i)
        {
            float distance = dis_mm;
            auto ps        = tree.RadiusSearch(point_cloud.position[i], distance);
            bool found     = false;
            to_merge[i]    = i;
            for (auto pi : ps)
            {
                if (valid[pi])
                {
                    to_erase.push_back(i);
                    to_merge[i] = pi;
                    found       = true;
                    break;
                }
            }
            if (!found)
            {
                valid[i] = true;
            }
        }
        point_cloud.EraseVertices(to_erase);
    }
    std::cout << "remove close z dis " << dis_mm << " Points " << bef << " -> " << point_cloud.NumVertices()
              << std::endl;
}

void SceneData::PointDistanceToCamera()
{
    PointDistanceToCamera(point_cloud, frames);
}


void SceneData::PointDistanceToCamera(Saiga::UnifiedMesh& point_cloud, std::vector<FrameData>& frames)
{
    point_cloud.data.resize(point_cloud.NumVertices());
#pragma omp parallel for
    for (int i = 0; i < point_cloud.NumVertices(); ++i)
    {
        float best_dis = 105235545;

        vec3 p = point_cloud.position[i];

        for (auto& f : frames)
        {
            auto [ip, z] = f.Project3(p);
            if (z <= 0 || !f.inImage(ip)) continue;
            float dis = (p - f.pose.translation().cast<float>()).squaredNorm();

            if (dis < best_dis)
            {
                best_dis = dis;
            }
        }

        point_cloud.data[i](0) = sqrt(best_dis);
    }
}

void SceneData::SortBlocksByRadius(int block_size)
{
    int n_blocks = point_cloud.NumVertices() / block_size;

    std::vector<int> indices(point_cloud.NumVertices());
    std::iota(indices.begin(), indices.end(), 0);

    for (int b = 0; b < n_blocks; ++b)
    {
        std::sort(indices.begin() + b * block_size, indices.begin() + (b + 1) * block_size,
                  [this](int i1, int i2) { return point_cloud.data[i1](3) < point_cloud.data[i2](3); });
    }
    point_cloud.ReorderVertices(indices);
}
// Unprojection to normalized image space
vec3 SceneCameraParams::ImageToNormalized(vec2 ip, float z)
{
    vec3 np;
    if (camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        vec2 ip2 = K.unproject2(ip);
        ip2      = Saiga::undistortNormalizedPointSimple(ip2, distortion);

        np = vec3(ip2(0) * z, ip2(1) * z, z);
    }
    else
    {
        SAIGA_EXIT_ERROR("sdlf");
    }
    return np;
}

std::pair<vec2, float> SceneCameraParams::NormalizedToImage(vec3 p)
{
    vec2 ip;
    float z;

    if (camera_model_type == CameraModel::OCAM)
    {
        ip = ocam.cast<float>().Project(p);
        z  = p.norm();
        //
        //            std::swap(p(0), p(1));
        //            p(2) *= -1;
        //
        //            ip = ProjectOCam2<float>(p, cam.ocam.AffineParams(), cam.ocam.poly_world2cam);
        //
        //            std::swap(ip(0), ip(1));
    }
    else if (camera_model_type == CameraModel::PINHOLE_DISTORTION)
    {
        z        = p.z();
        vec2 ipz = vec2(p(0) / p(2), p(1) / p(2));
        vec2 np  = distortNormalizedPoint(ipz, distortion);
        ip       = K.normalizedToImage(np);
    }
    else
    {
        throw std::runtime_error("not implemented");
    }
    return {ip, z};
}

TemplatedImage<ucvec3> SceneData::CPURenderFrame(int id, float scale)
{
    auto& f = frames[id];


    TemplatedImage<ucvec3> img(f.h * scale, f.w * scale);
    TemplatedImage<float> depth(f.h * scale, f.w * scale);
    img.makeZero();
    depth.getImageView().set(100000);

    auto cam = scene_cameras[f.camera_index];

    // std::swap(cam.ocam.cx, cam.ocam.cy);
    std::cout << "Rendering cpu frame " << id << std::endl;
    std::cout << "pose " << f.pose << std::endl;
    std::cout << "view " << f.pose.inverse() << std::endl;
    std::cout << "ocam " << cam.ocam << std::endl;
    std::cout << "K " << cam.K << std::endl;
    std::cout << "distortion " << cam.distortion << std::endl;

    for (int i = 0; i < point_cloud.NumVertices(); ++i)
    {
        vec3 p = point_cloud.position[i];
        vec3 c = point_cloud.color[i].head<3>();


        // vec4 p4 = make_vec4(p, 1);
        // p4      = view * p4;

        p = f.pose.inverse().cast<float>() * p;
        // std::cout << p.transpose() << std::endl;
        // std::cout << p4.transpose() << std::endl;
        // exit(0);


        // p = p4.head<3>();


        auto [ip, z] = cam.NormalizedToImage(p);
        ip *= scale;
        ivec2 ipi = ip.array().round().cast<int>();

        // std::cout << "proj point " << ipi.transpose() << std::endl;
        if (img.inImage(ipi))
        {
            auto img_z = depth(ipi(1), ipi(0));
            if (z < img_z)
            {
                depth(ipi(1), ipi(0)) = z;
                img(ipi(1), ipi(0))   = (c * 255).cast<unsigned char>();
            }
        }
    }
    return img;
}

TemplatedImage<ucvec3> SceneData::CPURenderFrameBilinear(int id, float scale)
{
    auto& f = frames[id];


    TemplatedImage<ucvec3> img(f.h * scale, f.w * scale);
    // TemplatedImage<float> depth(f.h * scale, f.w * scale);
    img.makeZero();
    // depth.getImageView().set(100000);

    auto cam = scene_cameras[f.camera_index];

    // std::swap(cam.ocam.cx, cam.ocam.cy);
    std::cout << "Rendering cpu frame " << id << std::endl;
    std::cout << "pose " << f.pose << std::endl;
    std::cout << "view " << f.pose.inverse() << std::endl;
    std::cout << "ocam " << cam.ocam << std::endl;
    std::cout << "K " << cam.K << std::endl;
    std::cout << "distortion " << cam.distortion << std::endl;

    for (int i = 0; i < point_cloud.NumVertices(); ++i)
    {
        vec3 p = point_cloud.position[i];
        vec3 c = point_cloud.color[i].head<3>();


        // vec4 p4 = make_vec4(p, 1);
        // p4      = view * p4;

        p = f.pose.inverse().cast<float>() * p;
        // std::cout << p.transpose() << std::endl;
        // std::cout << p4.transpose() << std::endl;
        // exit(0);


        // p = p4.head<3>();


        auto [ip, z] = cam.NormalizedToImage(p);
        ip *= scale;
        ivec2 ipix = ip.array().floor().cast<int>();


        // std::cout << "proj point " << ipi.transpose() << std::endl;
        vec4 blend_facs = compute_blending_fac(ip);

        for (int yj = 0; yj <= 1; ++yj)
        {
            for (int xi = 0; xi <= 1; ++xi)
            {
                int index_blend_vec = yj * 2 + xi;
                float f             = blend_facs[index_blend_vec];
                ivec2 ipi           = ipix + ivec2(xi, yj);
                if (img.inImage(ipi)) img(ipi(1), ipi(0)) = (f * c * 255).cast<unsigned char>();
            }
        }
    }
    return img;
}

SceneData::SceneData(int w, int h, IntrinsicsPinholef K)
{
    SceneCameraParams cam;
    cam.w = w;
    cam.h = h;
    cam.K = K;
    scene_cameras.push_back(cam);
}
/*
 * <<<<<<< HEAD


void SceneDatasetParams::Params(Saiga::SimpleIni* ini, CLI::App* app)
{
    SAIGA_PARAM(file_model);
    SAIGA_PARAM(image_dir);
    SAIGA_PARAM(mask_dir);
    SAIGA_PARAM_LIST(camera_files, ' ');

    SAIGA_PARAM(file_point_cloud);
    SAIGA_PARAM(file_point_cloud_compressed);


    int camera_type_int = (int)camera_model;
    SAIGA_PARAM(camera_type_int);
    camera_model = (CameraModel)camera_type_int;

    SAIGA_PARAM(znear);
    SAIGA_PARAM(zfar);
    SAIGA_PARAM(render_scale);
    SAIGA_PARAM(scene_exposure_value);
    SAIGA_PARAM(point_factor);
    SAIGA_PARAM(random_points_factor);


    {
        std::vector<std::string> up_vector = split(array_to_string(this->scene_up_vector), ' ');
        SAIGA_PARAM_LIST(up_vector, ' ');
        SAIGA_ASSERT(up_vector.size() == 3);
        for (int i = 0; i < 3; ++i)
        {
            double d           = to_double(up_vector[i]);
            scene_up_vector(i) = d;
        }
    }
}


void SceneCameraParams::Params(Saiga::SimpleIni* ini, CLI::App* app)
{
    SAIGA_PARAM(w);
    SAIGA_PARAM(h);

    auto vector2string = [](auto vector)
    {
        std::stringstream sstrm;
        sstrm << std::setprecision(15) << std::scientific;
        for (unsigned int i = 0; i < vector.size(); ++i)
        {
            sstrm << vector[i];
            if (i < vector.size() - 1) sstrm << " ";
        }
        return sstrm.str();
    };



    {
        std::vector<std::string> K = split(vector2string(this->K.cast<double>().coeffs()), ' ');
        SAIGA_PARAM_LIST_COMMENT(K, ' ', "# fx fy cx cy s");
        SAIGA_ASSERT(K.size() == 5);

        Vector<float, 5> K_coeffs;
        for (int i = 0; i < 5; ++i)
        {
            double d    = to_double(K[i]);
            K_coeffs(i) = d;
        }
        this->K = IntrinsicsPinholef(K_coeffs);
    }

    {
        std::vector<std::string> distortion = split(vector2string(this->distortion.cast<double>().Coeffs()), ' ');
        SAIGA_PARAM_LIST_COMMENT(distortion, ' ', "# 8 paramter distortion model. see distortion.h");

        SAIGA_ASSERT(distortion.size() == 8);


        Vector<float, 8> coeffs;
        for (int i = 0; i < 8; ++i)
        {
            double d  = to_double(distortion[i]);
            coeffs(i) = d;
        }
        this->distortion = Distortionf(coeffs);
    }


    {
        ocam.h = h;
        ocam.w = w;

        {
            std::vector<std::string> op = split(vector2string(this->ocam.cast<double>().AffineParams()), ' ');
            SAIGA_PARAM_LIST_COMMENT(op, ' ', "# c d e cx cy");
            SAIGA_ASSERT(op.size() == 5);

            Vector<double, 5> a_coeffs;
            for (int i = 0; i < 5; ++i)
            {
                double d    = to_double(op[i]);
                a_coeffs(i) = d;
            }
            this->ocam.SetAffineParams(a_coeffs);
        }
        {
            std::vector<std::string> poly_world2cam =
                split(vector2string(this->ocam.cast<double>().poly_world2cam), ' ');
            SAIGA_PARAM_LIST(poly_world2cam, ' ');

            std::vector<double> a_coeffs;
            for (int i = 0; i < poly_world2cam.size(); ++i)
            {
                double d = to_double(poly_world2cam[i]);
                a_coeffs.push_back(d);
            }
            this->ocam.SetWorld2Cam(a_coeffs);
        }

        {
            std::vector<std::string> poly_cam2world =
                split(vector2string(this->ocam.cast<double>().poly_cam2world), ' ');
            SAIGA_PARAM_LIST(poly_cam2world, ' ');

            std::vector<double> a_coeffs;
            for (int i = 0; i < poly_cam2world.size(); ++i)
            {
                double d = to_double(poly_cam2world[i]);
                a_coeffs.push_back(d);
            }
            this->ocam.SetCam2World(a_coeffs);
        }

        SAIGA_PARAM(ocam_cutoff);
    }


    std::cout << "  Image Size " << w << "x" << h << std::endl;
    std::cout << "  Aspect     " << float(w) / h << std::endl;
    std::cout << "  K          " << K << std::endl;
    std::cout << "  ocam       " << ocam << std::endl;
    std::cout << "  ocam cut   " << ocam_cutoff << std::endl;
    std::cout << "  normalized center " << (vec2(K.cx / w, K.cy / h) - vec2(0.5, 0.5)).transpose() << std::endl;
    std::cout << "  dist       " << distortion << std::endl;
}
=======
*/