/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/math/all.h"

#include "data/NeuralScene.h"
#include "gtest/gtest.h"
#include "rendering/PointBlending.h"

#include <iostream>
#include <vector>

#include "../../External/saiga/tests/numeric_derivative.h"
// #include "compare_numbers.h"
#include <saiga/core/tests/test.h>

#include "numerical_testing.h"


torch::Tensor t_blend_color(std::vector<torch::Tensor> colors, std::vector<torch::Tensor> confidences)
{
    torch::Tensor alpha_dest = torch::tensor({1.f});
    torch::Tensor color_out  = torch::tensor({0});
    for (int i = 0; i < colors.size(); ++i)
    {
        color_out  = alpha_dest * confidences[i] * colors[i] + color_out;
        alpha_dest = (1 - confidences[i]) * alpha_dest;
    }
    color_out = color_out / (1 - alpha_dest);
    return color_out;
}

torch::Tensor col_to_tensor(vec3 c)
{
    torch::Tensor out = torch::from_blob(&c, {3}, torch::TensorOptions().dtype(torch::kFloat32)).contiguous().clone();
    out               = out.requires_grad_(true);
    return out;
}

torch::Tensor conf_to_tensor(float c)
{
    torch::Tensor out =
        torch::from_blob(&c, {1}, torch::TensorOptions().dtype(torch::kFloat32)).contiguous().clone().cpu();
    out = out.requires_grad_(true);
    return out;
}



#define NUM_COL 8

Matrix<double, 1, 1> blend_color_red(vec8 colors, vec8 confidences, Matrix<double, 1, NUM_COL>* J_colors = nullptr,
                                     Matrix<double, 1, NUM_COL>* J_confidences = nullptr)
{
    float alpha_dest = 1.f;
    float color_out  = 0.f;
    Matrix<double, 1, 1> J_alphadest_alpha[NUM_COL];
    Matrix<double, 1, 1> J_cdest_alpha[NUM_COL];
    Matrix<double, 1, 1> J_cdest_col[NUM_COL];
    Matrix<double, 1, 1> J_cdest_oldcdest[NUM_COL];
    Matrix<double, 1, 1> J_cdest_alphadest[NUM_COL];
    Matrix<double, 1, 1> J_alphadest_alphadestold[NUM_COL];

    Matrix<double, 1, 1> run_Js_colorpart[NUM_COL];
    Matrix<double, 1, 1> run_Js_alphadest[NUM_COL];

    //  Matrix<double, 1, 1> run_Js_colors_only[NUM_COL];

    for (int i = 0; i < NUM_COL; ++i)
    {
        J_alphadest_alpha[i].setZero();
        J_cdest_alpha[i].setZero();
        J_cdest_col[i].setZero();
        J_cdest_oldcdest[i].setZero();
        J_cdest_alphadest[i].setZero();
        J_alphadest_alphadestold[i].setZero();
        run_Js_colorpart[i].setZero();
        run_Js_alphadest[i].setZero();
        //   run_Js_colors_only[i].setZero();
    }
    for (int i = 0; i < NUM_COL; ++i)
    {
        // int i=0;
        color_out = compute_blend_d(alpha_dest, confidences(i), colors(i), color_out, &J_cdest_alpha[i],
                                    &J_cdest_col[i], &J_cdest_alphadest[i],
                                    &J_cdest_oldcdest[i]);  // alpha_dest * confidences[i] * colors[i] + color_out;

        alpha_dest =
            compute_new_alphadest_d(alpha_dest, confidences(i), &J_alphadest_alpha[i], &J_alphadest_alphadestold[i]);

        run_Js_colorpart[i] = J_cdest_alpha[i];
        run_Js_alphadest[i] = J_alphadest_alpha[i];
        if (J_colors)
        {
            //     run_Js_colors_only[i] = J_cdest_col[i];
            auto& J = *J_colors;
            J(0, i) = J_cdest_col[i](0, 0);
        }

        for (int j = 0; j < i; ++j)
        {
            // J_cdest_oldcdest[i] == 1
            run_Js_colorpart[j] = 1 * run_Js_colorpart[j] + J_cdest_alphadest[i] * run_Js_alphadest[j];
            run_Js_alphadest[j] = J_alphadest_alphadestold[i] * run_Js_alphadest[j];
            // if(J_colors)
            //{
            //  //   run_Js_colors_only[j] = 1 * run_Js_colors_only[j];
            // }
        }
    }
    // Matrix<double,1,1> J_cout_alphadest;
    // J_cout_alphadest.setZero();
    // Matrix<double,1,1> J_cout_cdest;
    // J_cout_alphadest.setZero();
    //
    // color_out  = normalize_by_alphadest(color_out,alpha_dest,&J_cout_alphadest, &J_cout_cdest);
    // if(J_confidences)
    //     *J_confidences = J_cout_alphadest * J_alphadest_alpha + J_cout_cdest * J_cdest_alpha;

    if (J_confidences)
    {
        auto& J = *J_confidences;

        // J(0,1) = J_cdest_alpha[1](0,0)
        // J(0, 0) = J_cdest_oldcdest[1](0,0) * J_cdest_alpha[0](0,0) + J_cdest_alphadest[1](0,0) *
        // J_alphadest_alpha[0](0,0);

        // J(0, 2) = J_cdest_alpha[2](0,0);
        // J(0, 1) = J_cdest_oldcdest[2](0,0) * J_cdest_alpha[1](0,0) + J_cdest_alphadest[2](0,0) *
        // J_alphadest_alpha[1](0,0); J(0,0) = J_cdest_oldcdest[2](0,0) * (J_cdest_oldcdest[1](0,0) *
        // J_cdest_alpha[0](0,0) + J_cdest_alphadest[1](0,0)*J_alphadest_alpha[0](0,0))
        //     + J_cdest_alphadest[2](0,0)*J_alphadest_alphadestold[1](0,0)*J_alphadest_alpha[0](0,0);

        for (int i = 0; i < NUM_COL; ++i)
        {
            J(0, i) = run_Js_colorpart[i](0, 0);
        }
    }
    // if(J_colors){
    //     auto& J = *J_colors;
    //     for(int i=0; i< NUM_COL; ++i){
    //         J(0,i) = run_Js_colors_only[i](0,0);
    //     }
    // }
    Matrix<double, 1, 1> out;
    out(0, 0) = color_out;
    return out;
}

struct TestList
{
    std::vector<vec4> descriptors;
    std::vector<float> confidences;
    static constexpr int size_of_desc = 4;
    static constexpr int num_elems    = 8;
    TestList()
    {
        std::vector<float> r_vals;
        for (int i_r = 0; i_r < num_elems * (size_of_desc + 1); ++i_r)
        {
            r_vals.push_back(float(Saiga::Random::sampleDouble(0.0, 1.0)));
        }

        for (int i = 0; i < num_elems; ++i)
        {
            descriptors.push_back(vec4());
            for (int j = 0; j < size_of_desc; ++j)
            {
                descriptors[i][j] = r_vals[i * size_of_desc + j];
            }
            confidences.push_back(num_elems * size_of_desc);
        }
    }
};


vec4 blend_descriptors(std::vector<vec4> descriptors, std::vector<float> confidences)
{
    float alpha_dest = 1.f;
    vec4 result      = vec4(0, 0, 0, 0);
    for (int i = 0; i < descriptors.size(); ++i)
    {
        result     = compute_blend_vec<vec4, 4>(alpha_dest, confidences[i], descriptors[i], result);
        alpha_dest = compute_new_alphadest(alpha_dest, confidences[i]);
    }
    return result;
}

template <typename desc_vec, int size_of_desc_vec>
vec4 blend_descriptors_and_back(std::vector<desc_vec> descriptors, std::vector<float> confidences,
                                std::vector<Matrix<double, size_of_desc_vec, size_of_desc_vec>>* J_colors,
                                std::vector<Matrix<double, size_of_desc_vec, 1>>* J_confidences)
{
    std::cout << "not implemented" << std::endl;
    return vec4(0, 0, 0, 0);
    float alpha_dest = 1.f;
    vec4 color_out   = vec4(0, 0, 0, 0);
    Matrix<double, size_of_desc_vec, size_of_desc_vec> J_cdest_col, J_cdest_oldcdest;
    Matrix<double, size_of_desc_vec, 1> J_cdest_alpha, J_cdest_alphadest;
    Matrix<double, 1, 1> J_alphadest_alpha, J_alphadest_alphadestold;
    std::vector<Matrix<double, size_of_desc_vec, 1>> run_Js_colorpart(descriptors.size());
    std::vector<Matrix<double, 1, 1>> run_Js_alphadest(descriptors.size());
    for (int i = 0; i < descriptors.size(); ++i)
    {
        run_Js_colorpart[i].setZero();
        run_Js_alphadest[i].setZero();
    }
    for (int i = 0; i < descriptors.size(); ++i)
    {
        {
            J_cdest_alpha.setZero();
            J_cdest_col.setZero();
            J_cdest_alphadest.setZero();
            J_cdest_oldcdest.setZero();
        }
        J_alphadest_alpha.setZero();
        J_alphadest_alphadestold.setZero();


        // int i=0;
        color_out = compute_blend_vec<desc_vec, size_of_desc_vec>(alpha_dest, confidences[i], descriptors[i], color_out,
                                                                  &J_cdest_alpha, &J_cdest_col, &J_cdest_alphadest,
                                                                  &J_cdest_oldcdest);

        alpha_dest = compute_new_alphadest_d(alpha_dest, confidences[i], &J_alphadest_alpha, &J_alphadest_alphadestold);

        run_Js_colorpart[i] = J_cdest_alpha;
        run_Js_alphadest[i] = J_alphadest_alpha;
        if (J_colors)
        {
            //     run_Js_colors_only[i] = J_cdest_col[i];
            auto& J = (*J_colors)[i];
            for (int t = 0; t < size_of_desc_vec; ++t) J(0, t) = J_cdest_col(0, t);
        }

        for (int j = 0; j < i; ++j)
        {
            // J_cdest_oldcdest[i] == 1
            run_Js_colorpart[j] = 1 * run_Js_colorpart[j] + J_cdest_alphadest[i] * run_Js_alphadest[j];
            run_Js_alphadest[j] = J_alphadest_alphadestold[i] * run_Js_alphadest[j];
            // if(J_colors)
            //{
            //  //   run_Js_colors_only[j] = 1 * run_Js_colors_only[j];
            // }
        }
    }
    if (J_confidences)
    {
        for (int i = 0; i < descriptors.size(); ++i)
        {
            auto& J = (*J_confidences)[i];
            J(0, 0) = run_Js_colorpart[i](0, 0);
        }
    }
    return color_out;
}

TEST(AlphaBlend, Forward)
{
    for (int t_id = 0; t_id < 100; ++t_id)
    {
        TestList testlist;
        blend_descriptors(testlist.descriptors, testlist.confidences);
    }
}

TEST(AlphaBlend, BlendBack)
{
    std::cout << "Test alpha blend " << std::endl;

    for (int t_id = 0; t_id < 100; ++t_id)
    {
        TestList testlist;


        std::vector<Matrix<double, 1, testlist.size_of_desc>> J_own_col;
        std::vector<Matrix<double, 1, testlist.size_of_desc>> J_own_conf;
        std::vector<Matrix<double, 1, testlist.size_of_desc>> J_numeric_conf;
        std::vector<Matrix<double, 1, testlist.size_of_desc>> J_numeric_color;
        for (int i = 0; i < testlist.descriptors.size(); ++i)
        {
            J_own_col[i].setZero();
            J_own_conf[i].setZero();
            J_numeric_conf[i].setZero();
            J_numeric_color[i].setZero();
        }
        return;
        // auto res = blend_descriptors_and_back(testlist.descriptors, testlist.confidences, &J_own_col, &J_own_conf);

#define EPSILON 1e-3
        /*
                std::cout << "_____________________________________________________" << std::endl;

                auto res1 = blend_color_red(colors_red_channel, confidences, &J_own_col, &J_own_conf);
                std::cout << "CONF: OWN IMPL: res " << res1 << "_______ J: " << J_own_conf << std::endl;

                auto res2 = EvaluateNumeric([&](auto p) { return blend_color_red(colors_red_channel, p); }, confidences,
                                            &J_numeric_conf);
                std::cout << "CONF: NUMBERIC_IMPL: res " << res2 << "_______ J: " << J_numeric_conf << std::endl;

                bool not_ok = false;
                not_ok |= ExpectCloseRelative(res1, res2, EPSILON, false);
                not_ok |= ExpectCloseRelative(J_own_conf, J_numeric_conf, EPSILON, false);

                std::cout << "COLOR: OWN IMPL: res " << res1 << "_______ J: " << J_own_col << std::endl;
                res2 = EvaluateNumeric([&](auto p) { return blend_color_red(p, confidences); }, colors_red_channel,
                                       &J_numeric_color);
                std::cout << "COLOR: NUMBERIC_IMPL: res " << res2 << "_______ J: " << J_numeric_color << std::endl;

                not_ok |= ExpectCloseRelative(res1, res2, EPSILON, false);
                not_ok |= ExpectCloseRelative(J_own_col, J_numeric_color, EPSILON, false);

                EXPECT_FALSE(not_ok);
                */
        // if (not_ok)
        //{
        //     return;
        // }
    }

    std::cout << "ALL OK" << std::endl;

    //  return 0;
}