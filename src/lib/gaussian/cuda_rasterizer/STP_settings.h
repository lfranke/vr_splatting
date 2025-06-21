//
// Created by linus on 17.06.24.
//
#pragma once

enum SortMode
{
    GLOBAL      = 0,
    PPX_FULL    = 1,
    PPX_KBUFFER = 2,
    HIER        = 3,
};

enum GlobalSortOrder
{
    Z_DEPTH    = 0,
    DISTANCE   = 1,
    PTD_CENTER = 2,
    PTD_MAX    = 3,
};

struct SortQueueSizes
{
    int tile_4x4  = 64;
    int tile_2x2  = 8;
    int per_pixel = 4;
};

struct CullingSettings
{
    bool rect_bounding            = false;
    bool tight_opacity_bounding   = false;
    bool tile_based_culling       = false;
    bool hierarchical_4x4_culling = false;
};

struct SortSettings
{
    SortQueueSizes queue_sizes = SortQueueSizes();
    SortMode sort_mode         = SortMode::GLOBAL;
    GlobalSortOrder sort_order = GlobalSortOrder::Z_DEPTH;
};
struct ExtendedSettings
{
    SortSettings sort_settings       = SortSettings();
    CullingSettings culling_settings = CullingSettings();
    bool load_balancing              = false;
    bool proper_ewa_scaling          = false;
};