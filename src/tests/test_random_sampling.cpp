/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include <iostream>
#include <vector>
// #include "../lib/rendering/PointRendererHelper.h"

#define HD

// loosely following park-miller
HD inline int intrnd(int seed)  // 1<=seed<=m
{
    int const a = 16807;       // ie 7**5
    int const m = 2147483647;  // ie 2**31-1
    seed        = (long(seed * a)) % m;
    return seed;
}

HD inline float getRandForPointAndEpoch(int point_id, int epoch)
{
    int a1      = intrnd(point_id + 1);
    int a2      = intrnd(epoch + 1);
    int const m = 2147483647;  // ie 2**31-1
    int r       = intrnd((a1 * a2 * (point_id + 1) * (epoch + 1))) % m;
    return (float(r) / float(m)) * 0.5 + 0.5;
}


int main(int argc, char* argv[])
{
    std::cout << "Test random sampling for point discards" << std::endl;

    constexpr int epochs    = 10000;
    constexpr int point_num = 100000;

    std::vector<float> randoms(point_num, 0);
    for (int ep = 0; ep < epochs; ++ep)
    {
#pragma omp parallel for
        for (int p_id = 0; p_id < point_num; ++p_id)
        {
            float rand = getRandForPointAndEpoch(p_id, ep);
            randoms[p_id] += rand;
        }
    }
    for (int i = 0; i < randoms.size(); ++i)
    {
        randoms[i] /= float(epochs);
    }
    float conf_interval      = 0.05;
    int num_outside_interval = 0;
    for (int i = 0; i < randoms.size(); ++i)
    {
        // std::cout << std::abs(randoms[i]-0.5) << std::endl;
        if (std::abs(randoms[i] - 0.5) > conf_interval)
        {
            //   std::cout << randoms[i] << std::endl;
            num_outside_interval++;
        }
    }
    std::cout << "Tested " << point_num << " points over " << epochs << " epochs. " << num_outside_interval
              << " points are outside of the confidence interval of 0.5 +- " << conf_interval << std::endl;


    return 0;
}