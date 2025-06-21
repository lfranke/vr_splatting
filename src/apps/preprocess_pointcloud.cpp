﻿/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/commandLineArguments.h"

#include "data/SceneData.h"

void PreprocessPointcloud(std::string scene_dir, int dup_factor)
{
    auto scene = std::make_shared<SceneData>(scene_dir);

    scene->DuplicatePoints(dup_factor, 1);
    scene->ComputeRadius();
    scene->RemoveClosePoints(0.00005);
    scene->RemoveLonelyPoints(5, 0.02);
    scene->point_cloud.RemoveDoubles(0.0002);
    std::cout << "Remaining Points: " << scene->point_cloud.NumVertices() << std::endl;

    scene->point_cloud.ReorderMorton64();
    scene->point_cloud.RandomBlockShuffle(default_point_block_size);
    scene->ComputeRadius();
    scene->Save();


    UnifiedModel(scene->point_cloud).Save(scene_dir + "/point_cloud_dup.ply");
}

int main(int argc, char* argv[])
{
    CLI::App app{"Point Cloud Preprocessor", "preprocess_pointcloud"};

    std::string scene_path;
    int point_factor = 1;
    app.add_option("--scene_path", scene_path)->required();
    app.add_option("--point_factor", point_factor,
                   "Increase number of point by this factor. Set to 1 to keep the number of points.")
        ->required();


    CLI11_PARSE(app, argc, argv);

    PreprocessPointcloud(scene_path, point_factor);

    return 0;
}
