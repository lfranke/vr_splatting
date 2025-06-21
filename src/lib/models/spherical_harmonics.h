/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once


#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/torch.h"

#include <torch/torch.h>
#ifdef USE_TINYCUDANN
#    include "tiny-cuda-nn/torch_wrapper.h"
#endif

class SphericalHarmonicsEncoding
{
   public:
    SphericalHarmonicsEncoding(int degree);


    int OutputChannels() { return degree * degree; }

    // Input:
    //      direction [ <shape> , 3]
    // Output:
    //      coefficients [ <shape> , OutputChannels()]
    torch::Tensor forward(torch::Tensor direction);

   public:
#ifdef USE_TINYCUDANN
    TcnnTorchModule module = nullptr;
#endif
    int degree;
};
