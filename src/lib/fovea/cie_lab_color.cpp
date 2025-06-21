//
// Created by Linus on 27/08/2024.
//
#include "cie_lab_color.h"
#include "saiga/core/util/assert.h"
torch::Tensor rgb_to_xyz(const torch::Tensor rgb)
{
    const int color_dim = rgb.sizes().size() - 3;

    auto r = rgb.slice(color_dim, 0, 1);  //.index({"...", 0, Slice(), Slice()});
    auto g = rgb.slice(color_dim, 1, 2);  //.index({"...", 1, Slice(), Slice()});
    auto b = rgb.slice(color_dim, 2, 3);  //.index({"...", 2, Slice(), Slice()});

    auto x = 0.41245f * r + 0.35758f * g + 0.18042f * b;
    auto y = 0.21267f * r + 0.71516f * g + 0.07216f * b;
    auto z = 0.01933f * r + 0.11919f * g + 0.95022f * b;

    return torch::cat({x, y, z}, color_dim);
}

torch::Tensor xyz_to_rgb(const torch::Tensor xyz)
{
    const int color_dim = xyz.sizes().size() - 3;

    auto x = xyz.slice(color_dim, 0, 1);
    auto y = xyz.slice(color_dim, 1, 2);
    auto z = xyz.slice(color_dim, 2, 3);

    auto r = 3.24048f * x + -1.53715f * y + -0.49854f * z;
    auto g = -0.96925f * x + 1.87599f * y + 0.04156f * z;
    auto b = 0.05565f * x + -0.20404f * y + 1.05731f * z;

    return torch::cat({r, g, b}, color_dim);
}

torch::Tensor srgb_to_lab(const torch::Tensor srgb_norm)
{
    SAIGA_ASSERT(false);
    auto srgb= srgb_norm* 255.f;
    auto small_values = srgb / 12.92f;
    auto big_values = (torch::exp((srgb + 0.055f) / 1.055f), 2.4f);
    auto rgb = torch::where(srgb <= 0.04045f, small_values, big_values);

    return rgb_to_lab(rgb/255.f);
}

torch::Tensor rgb_to_lab(const torch::Tensor rgb)
{
    const int color_dim = rgb.sizes().size() - 3;

    auto xyz            = rgb_to_xyz(rgb);

    auto xyz_norm = xyz;
    xyz_norm.slice(color_dim, 0, 1) /= 0.9505f;
    xyz_norm.slice(color_dim, 2, 3) /= 1.0888f;

    float threshold = 0.00886f;

    auto xyz_pow = torch::pow(xyz_norm.clamp(threshold), 1.f / 3.f);

    //auto xyz_scale = 7.787f * xyz_norm + 4.f / 29.f;
    auto xyz_scale = (1.f /116.f) * (24389.f/27.f) * xyz_norm + 16.f;
    auto xyz_r     = torch::where(xyz_norm > threshold, xyz_pow, xyz_scale);

    auto L = 116.f * xyz_r.slice(color_dim, 1, 2) - 16.f;
    auto a = 500.f * (xyz_r.slice(color_dim, 0, 1) - xyz_r.slice(color_dim, 1, 2));
    auto b = 200.f * (xyz_r.slice(color_dim, 1, 2) - xyz_r.slice(color_dim, 2, 3));

    return torch::cat({L, a, b}, color_dim);
}

static torch::Tensor scale_lab_space(torch::Tensor lab){
    const int color_dim = lab.sizes().size() - 3;

    const float scale_val = 128.f;

    lab.slice(color_dim, 0, 1) /= 100.f;
    //[unbounded, but scale [-scale_val,scale_val] to [0,1]
    lab.slice(color_dim, 1, 3) /= 2.f * scale_val;
    lab.slice(color_dim, 1, 3) += 1.f / 2.f;

    return lab;
    }

torch::Tensor srgb_to_lab_normalized(const torch::Tensor srgb)
{
    SAIGA_ASSERT(false);

    auto lab            = srgb_to_lab(srgb).clamp(0.01,0.99);
    return scale_lab_space(lab);
}

torch::Tensor rgb_to_lab_normalized(const torch::Tensor rgb)
{

    auto lab            = rgb_to_lab(rgb);
    return scale_lab_space(lab);
}

torch::Tensor lab_to_rgb(const torch::Tensor lab)
{
    const int color_dim = lab.sizes().size() - 3;

    auto L = lab.slice(color_dim, 0, 1);
    auto a = lab.slice(color_dim, 1, 2);
    auto b = lab.slice(color_dim, 2, 3);

    auto fy   = (L + 16.0) / 116.0;
    auto fx   = (a / 500.0) + fy;
    auto fz   = (fy - (b / 200.0)).clamp(0.f);
    auto fxyz = torch::cat({fx, fy, fz}, color_dim);

    auto xyz_pow   = torch::pow(fxyz, 3.f);
    auto xyz_scale = (fxyz - 4.f / 29.f) / 7.787f;
    auto xyz       = torch::where(fxyz > 0.2068966, xyz_pow, xyz_scale);

    auto xyz_norm = xyz;
    xyz_norm.slice(color_dim, 0, 1) *= 0.9505f;
    xyz_norm.slice(color_dim, 2, 3) *= 1.0888f;

    auto rgb = xyz_to_rgb(xyz_norm);

    return rgb;
}
