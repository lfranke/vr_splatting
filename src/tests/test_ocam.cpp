/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/math/math.h"
#include "saiga/core/sophus/Sophus.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/cameraModel/OCam.h"

#include "data/Dataset.h"
#include "models/Pipeline.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>



int main(int argc, char* argv[])
{
    int w   = 5472;
    int h   = 3648;
    vec5 ap = vec5(1.000185014003372e+00, -7.174764051563908e-05, 2.430793224923302e-04, 1.824925597696699e+03,
                   2.725090333359710e+03);
    std::vector<float> world2cam = {2.180444628737742e+03,  1.372696159664295e+03,  -1.964975565512464e+02,
                                    -2.830049694074260e+02, 1.390210198260174e+02,  5.003730656049156e+02,
                                    -9.982626212551259e+01, -6.872935942306067e+02, 4.693052243604325e+01,
                                    7.775401120588649e+02,  1.204269438651390e+02,  -5.871922589847818e+02,
                                    -2.274466314525343e+02, 2.422745032222010e+02,  1.580266170813702e+02,
                                    -2.658486489066854e+01, -3.989550079533365e+01, -8.118421039340319e+00};
    std::vector<float> cam2world = {-1.372785698018784e+03, 0.000000000000000e+00, 3.614401850212424e-04,
                                    -2.817037238699711e-07, 2.644392660936918e-10, -1.074142104512164e-13,
                                    1.759999176278743e-17};
    // vec5 ap = vec5(1.000120000000000e+00, 3.123290000000000e-03,
    // -3.110610000000000e-03, 1.826300000000000e+03, 2.725010000000000e+03); std::vector<float> world2cam =
    // {2.185330000000000e+03, 1.374650000000000e+03, -1.956840000000000e+02,
    // -2.779350000000000e+02, 1.402440000000000e+02, 4.893720000000000e+02, -1.035840000000000e+02,
    // -6.679500000000000e+02, 5.614510000000000e+01, 7.562140000000001e+02, 1.058190000000000e+02,
    // -5.744420000000000e+02, -2.145730000000000e+02, 2.400390000000000e+02 ,1.528420000000000e+02,
    // -2.750980000000000e+01, -3.936660000000000e+01, -7.958420000000000e+00}; std::vector<float> cam2world =
    // {-1.376070000000000e+03, 0.000000000000000e+00, 3.545170000000000e-04,
    // -2.687470000000000e-07, 2.537960000000000e-10, -1.033720000000000e-13, 1.699990000000000e-17};
    OCam<float> ocam(w, h, ap, cam2world, world2cam);

    const int num_tests = 100;

    for (int t_num = 0; t_num < num_tests; ++t_num)
    {
        // vec2 ip   = vec2(2.725090333359710e+03 , 1.824925597696699e+03);
        vec2 ip   = vec2(Saiga::linearRand(0, w), Saiga::linearRand(0, h));
        float dep = Saiga::linearRand(0, 100);
        //!!!!!!!!!!!!!!
        // depth may not be < 0
        SAIGA_ASSERT(dep >= 0);

        // dep = 1.f;
        std::cout << ip.x() << " " << ip.y() << " " << dep << " - comp - ";
        vec3 world_p = UnprojectOCam<float>(ip, dep, ap, cam2world);
        //   std::cout << world_p.x() << " "<< world_p.y() << " "<< world_p.z() << " "  << std::endl;
        vec3 screen_p = ProjectOCam<float>(world_p, ap, world2cam);
        std::cout << screen_p.x() << " " << screen_p.y() << " " << screen_p.z() << " " << std::endl;

        SAIGA_ASSERT(length((screen_p - vec3(ip.x(), ip.y(), dep))) < 0.5);
    }
}
