//
// Created by linus on 13.06.24.
//
#pragma once
#include "saiga/core/model/UnifiedModel.h"

#include <filesystem>
#include <fstream>

namespace Saiga
{
template <typename T>
T read_binary_value(std::istream& file)
{
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

// Reads and preloads a binary file into a string stream
// file_path: path to the file
// returns: a unique pointer to a string stream
std::unique_ptr<std::istream> read_binary(std::filesystem::path file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    std::unique_ptr<std::istream> file_stream;
    if (file.fail())
    {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    // preload
    std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
    file_stream = std::make_unique<std::stringstream>(std::string(buffer.begin(), buffer.end()));
    return file_stream;
}

Saiga::UnifiedMesh read_point3D_binary(std::filesystem::path file_path)
{
    if (!std::filesystem::exists(file_path))
    {
        std::cerr << "No Points3D.bin file for GS under " << file_path << std::endl;
        return Saiga::UnifiedMesh();
    }
    auto point3D_stream_buffer = read_binary(file_path);
    const size_t point3D_count = read_binary_value<uint64_t>(*point3D_stream_buffer);

    struct Track
    {
        uint32_t _image_ID;
        uint32_t _max_num_2D_points;
    };

    Saiga::UnifiedMesh point_cloud;
    point_cloud.position = std::vector<vec3>(point3D_count);
    point_cloud.color    = std::vector<vec4>(point3D_count);
    //  point_cloud._normals.reserve(point3D_count); <- no normals saved. Just ignore.
    for (size_t i = 0; i < point3D_count; ++i)
    {
        // just ignore the point3D_ID
        read_binary_value<uint64_t>(*point3D_stream_buffer);
        // vertices
        point_cloud.position[i].x() = static_cast<float>(read_binary_value<double>(*point3D_stream_buffer));
        point_cloud.position[i].y() = static_cast<float>(read_binary_value<double>(*point3D_stream_buffer));
        point_cloud.position[i].z() = static_cast<float>(read_binary_value<double>(*point3D_stream_buffer));

        // colors
        point_cloud.color[i].x() = static_cast<float>(read_binary_value<uint8_t>(*point3D_stream_buffer));
        point_cloud.color[i].y() = static_cast<float>(read_binary_value<uint8_t>(*point3D_stream_buffer));
        point_cloud.color[i].z() = static_cast<float>(read_binary_value<uint8_t>(*point3D_stream_buffer));

        // the rest can be ignored.
        read_binary_value<double>(*point3D_stream_buffer);  // ignore

        const auto track_length = read_binary_value<uint64_t>(*point3D_stream_buffer);
        std::vector<Track> tracks;
        tracks.resize(track_length);
        point3D_stream_buffer->read(reinterpret_cast<char*>(tracks.data()), track_length * sizeof(Track));
    }

    // write_ply_file(file_path.parent_path() / "points3D.ply", point_cloud);
    return point_cloud;
}
}  // namespace Saiga