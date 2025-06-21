/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/exif/TinyEXIF.h"
#include "saiga/core/util/file.h"
#include "saiga/vision/cameraModel/OCam.h"
#include "saiga/vision/util/ColmapReader.h"

#include "data/SceneData.h"

/*
void ColmapReader::Save(const std::string& dir)
{
    SAIGA_ASSERT(Check());
    std::filesystem::create_directories(dir);

std::string img_file = dir + "/images.bin";
std::string cam_file = dir + "/cameras.bin";
std::string poi_file = dir + "/points3D.bin";


{
    // Read cameras
    BinaryFile file(cam_file, std::ios::out);

    uint64_t num_cameras = cameras.size();
    file << num_cameras;

    for (auto& c : cameras)
    {
        file << c.camera_id << c.model_id;
        file << c.w << c.h;


        // FULL_OPENCV
        // fx, fy, cx, cy,   k1, k2, p1, p2,   k3, k4, k5, k6
        std::array<double, 12> coeffs;
        coeffs[0] = c.K.fx;
        coeffs[1] = c.K.fy;
        coeffs[2] = c.K.cx;
        coeffs[3] = c.K.cy;

        coeffs[4] = c.dis.k1;
        coeffs[5] = c.dis.k2;
        coeffs[6] = c.dis.p1;
        coeffs[7] = c.dis.p2;

        coeffs[8]  = c.dis.k3;
        coeffs[9]  = c.dis.k4;
        coeffs[10] = c.dis.k5;
        coeffs[11] = c.dis.k6;
        file << coeffs;
    }
}


{
    // save images
    BinaryFile file(img_file, std::ios::out);

    uint64_t num_images = images.size();
    file << num_images;

    for (auto& ci : images)
    {
        file << ci.image_id;

        file << ci.q.w() << ci.q.x() << ci.q.y() << ci.q.z();
        file << ci.t.x() << ci.t.y() << ci.t.z();
        file << ci.camera_id;


        for (char x : ci.name)
        {
            file << x;
        }
        file << '\0';

        uint64_t num_points = ci.obvservations.size();
        file << num_points;

        for (auto& p : ci.obvservations)
        {
            file << p.keypoint.x() << p.keypoint.y();
            file << p.world_point_index;
        }
    }
}



{
    // save points
    BinaryFile file(poi_file, std::ios::out);

    uint64_t num_points = points.size();
    file << num_points;

    for (auto p : points)
    {
        file << p.world_point_index;
        file << p.position.x() << p.position.y() << p.position.z();
        file << p.color.x() << p.color.y() << p.color.z();
        file << p.error;

        uint64_t num_tracks = p.tracks.size();
        file << num_tracks;

        for (auto& t : p.tracks)
        {
            file << t.image_id << t.keypoint_id;
        }
    }
}
}

*/

int main(int argc, char* argv[])
{
    std::string scene_dir, output_dir;

    bool do_colmap_sparse = false;

    CLI::App app{"ADOP to Colmap Scene Converter", "adop2colmap"};
    app.add_option("--scene_dir", scene_dir)->required();
    app.add_option("--output_dir", output_dir)->required();
    app.add_option("--do_colmap_sparse", do_colmap_sparse);

    CLI11_PARSE(app, argc, argv);

    std::filesystem::create_directories(output_dir + "/tmp/");


    /*
        std::vector<Sophus::SE3d> poses;
        if (std::filesystem::exists(file_pose))
        {
            std::ifstream strm(file_pose);

            std::string line;
            while (std::getline(strm, line))
            {
                std::stringstream sstream(line);

                Quat q;
                Vec3 t;

                sstream >> q.x() >> q.y() >> q.z() >> q.w() >> t.x() >> t.y() >> t.z();


                SE3 view(q, t);

                poses.push_back(view.inverse());
            }
        }*/


    auto scene = std::make_shared<SceneData>(scene_dir);


    std::string img_file = output_dir + "/tmp/" + "/images.txt";
    std::string cam_file = output_dir + "/tmp/" + "/cameras.txt";
    std::string poi_file = output_dir + "/tmp/" + "/points3D.txt";


    {
        // Read cameras
        // BinaryFile file(cam_file, std::ios::out);
        std::ofstream file;
        file.open(cam_file);

        uint64_t num_cameras = scene->scene_cameras.size();
        // file << num_cameras;

        file << "# A->col: Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                "# Number of cameras: "
             << num_cameras << "\n";

        int cam_id = 0;


        for (auto& c : scene->scene_cameras)
        {
            std::string camera_type = "PINHOLE";
            auto dis_coeff          = c.distortion.Coeffs();
            for (int i = 0; i < dis_coeff.size(); ++i)
            {
                if (dis_coeff[i] != 0.f)
                {
                    camera_type = "RADIAL";
                    std::cout << "RADIAL CAM" << std::endl;
                    break;
                }
            }
            SAIGA_ASSERT(c.camera_model_type == CameraModel::PINHOLE_DISTORTION, "NOT IMPLEMENTED");

            file << cam_id + 1 << " " << camera_type << " ";
            file << c.w << " " << c.h << " ";


            // FULL_OPENCV
            // fx, fy, cx, cy,   k1, k2, p1, p2,   k3, k4, k5, k6
            file << c.K.fx << " ";
            file << c.K.fy << " ";
            file << c.K.cx << " ";
            file << c.K.cy << " ";

            if (camera_type == "RADIAL")
            {
                file << c.distortion.k1 << " ";
                file << c.distortion.k2 << " ";
                file << c.distortion.p1 << " ";
                file << c.distortion.p2 << " ";

                file << c.distortion.k3 << " ";
                file << c.distortion.k4 << " ";
                file << c.distortion.k5 << " ";
                file << c.distortion.k6 << " ";
            }

            ++cam_id;
        }
        file.close();
    }


    {
        // save images
        // BinaryFile file(img_file, std::ios::out);

        std::ofstream file;
        file.open(img_file);


        uint64_t num_images = scene->frames.size();
        file << "# A->col: Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID); num img: "
             << num_images << "\n";

        for (auto& frame : scene->frames)
        {
            file << frame.image_index + 1 << " ";

            Sophus::SE3d pose = frame.pose.inverse();
            Quat q            = pose.unit_quaternion();
            Vec3 t            = pose.translation();

            file << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
            file << t.x() << " " << t.y() << " " << t.z() << " ";
            file << frame.camera_index + 1 << " ";

            file << frame.target_file << std::endl;

            // no keypoints
            file << std::endl;
        }
        file.close();
    }



    {
        // save points
        //  BinaryFile file(poi_file, std::ios::out);

        std::ofstream file;
        file.open(poi_file);

        file << std::endl;
        file.close();
    }


    if (do_colmap_sparse)
    {
        std::cout << "doing sparse reco with poses for feature point extractions" << std::endl;
        std::string colmap_db = output_dir + "/database.db";


        float default_focal_length_fac = scene->scene_cameras.front().w / scene->scene_cameras.front().K.fx;

        std::string feat_extraction = std::string("colmap feature_extractor ");
        feat_extraction += "--database_path " + colmap_db + " --image_path " + scene->dataset_params.image_dir;
        feat_extraction += " SiftExtraction.default_focal_length_factor " + std::to_string(default_focal_length_fac);

        std::system(feat_extraction.c_str());

        std::string feature_matcher = "";
        feature_matcher += "colmap exhaustive_matcher --database_path " + colmap_db + " --SiftMatching.use_gpu 1";
        feature_matcher += " --ExhaustiveMatching.block_size 150";
        std::system(feature_matcher.c_str());



        std::string point_tri = "";
        point_tri += "colmap point_triangulator --database_path " + colmap_db + " --image_path " +
                     scene->dataset_params.image_dir + " --input_path " + output_dir + "/tmp/" + " --output_path " +
                     output_dir;

        std::system(point_tri.c_str());

        std::string undistorter = "";
        //        undistorter += "colmap image_undistorter --image_path " + args.source_path + "/input --input_path " +
        //                       args.source_path + "/distorted/sparse/0     --output_path " + args.source_path +
        //            ;
    }

    // app.add_option("--image_dir", image_dir)->required();
    // app.add_option("--point_cloud_file", point_cloud_file)->required();
    // app.add_option("--output_path", output_path)->required();
    // app.add_option("--scale_intrinsics", scale_intrinsics)->required();
    // app.add_option("--render_scale", render_scale)->required();


    // ColmapScene(sparse_dir, image_dir, point_cloud_file, output_path, scale_intrinsics, render_scale);


    return 0;
}
