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



/*
vec3 blend_color(std::vector<vec3> colors, std::vector<float> confidences, std::vector<Matrix<double,3,1>> v_of_J_c =
nullptr){ float alpha_dest = 1.f; vec3 color_out = vec3(0,0,0); for(int i=0; i< colors.size(); ++i){ color_out =
alpha_dest * confidences[i] * colors[i] + color_out; alpha_dest = (1-confidences[i]) * alpha_dest;
    }
    color_out  = color_out/(1-alpha_dest);
    return color_out;
}
*/



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
        color_out = compute_blend(alpha_dest, confidences(i), colors(i), color_out, &J_cdest_alpha[i], &J_cdest_col[i],
                                  &J_cdest_alphadest[i],
                                  &J_cdest_oldcdest[i]);  // alpha_dest * confidences[i] * colors[i] + color_out;

        alpha_dest =
            compute_new_alphadest(alpha_dest, confidences(i), &J_alphadest_alpha[i], &J_alphadest_alphadestold[i]);

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

/*
int main(int argc, char* argv[])
{
    std::cout << "Test alpha blend " << std::endl;

    for (int i = 0; i < 100; ++i)
    {
        //  std::vector<vec3> colors = {{1,0.01,0.01},{0.01,1,0.01},{0.01,0.01,1}};
        std::vector<float> r_vals;
        for (int i_r = 0; i_r < 16; ++i_r)
        {
            r_vals.push_back(float(Saiga::Random::sampleDouble(0.0, 1.0)));
        }
        vec8 colors_red_channel = {r_vals[0], r_vals[1], r_vals[2], r_vals[3],
                                   r_vals[4], r_vals[5], r_vals[6], r_vals[7]};

        vec8 confidences = {r_vals[8],  r_vals[9],  r_vals[10], r_vals[11],
                            r_vals[12], r_vals[13], r_vals[14], r_vals[15]};

        std::cout << "colors: " << std::endl
                  << colors_red_channel << std::endl
                  << " - confidences: " << confidences << std::endl;

#if 0
        auto out_col_forward = blend_color_red(colors_red_channel, confidences);

        std::cout << "Blended color forward:" << std::endl << out_col_forward(0) << std::endl;
    std::vector<torch::Tensor> t_colors;
    std::vector<torch::Tensor> t_confidences;
    for(int i=0; i<colors_red_channel.size(); ++i){
        t_colors.push_back(conf_to_tensor(colors_red_channel[i]));
        t_confidences.push_back(conf_to_tensor(confidences[i]));
    }

 //   =
{conf_to_tensor(colors_red_channel[0]),conf_to_tensor(colors_red_channel[1]),conf_to_tensor(colors_red_channel[2])};
 //   = {conf_to_tensor(confidences[0]),conf_to_tensor(confidences[1]),conf_to_tensor(confidences[2])};

    torch::Tensor t_out_col_forward = t_blend_color(t_colors, t_confidences);
    std::cout << "Blended color tensor forward:" << std::endl << t_out_col_forward << std::endl;
    t_out_col_forward.backward(torch::tensor({1}));

    //for(auto& t : t_colors)
    //{
    //   0 std::cout << "Blended color tensor backwards:" << t.grad()<< std::endl;
    //}
    for(auto& t : t_confidences)
    {
        std::cout << "Blended conf tensor backwards:" << t.grad() << std::endl;
    }
#endif
        Matrix<double, 1, NUM_COL> J_own_col;
        Matrix<double, 1, NUM_COL> J_own_conf;
        Matrix<double, 1, 8> J_numeric_conf;
        Matrix<double, 1, 8> J_numeric_color;
        J_own_col.setZero();
        J_own_conf.setZero();
        J_numeric_conf.setZero();
        J_numeric_color.setZero();

#define EPSILON 1e-3

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
        if (not_ok)
        {
            return 1;
        }
    }

    std::cout << "ALL OK" << std::endl;

    return 0;
}*/
TEST(AlphaBlend, BlendBack)
{
    std::cout << "Test alpha blend " << std::endl;

    for (int i = 0; i < 100; ++i)
    {
        //  std::vector<vec3> colors = {{1,0.01,0.01},{0.01,1,0.01},{0.01,0.01,1}};
        std::vector<float> r_vals;
        for (int i_r = 0; i_r < 16; ++i_r)
        {
            r_vals.push_back(float(Saiga::Random::sampleDouble(0.0, 1.0)));
        }
        vec8 colors_red_channel = {r_vals[0], r_vals[1], r_vals[2], r_vals[3],
                                   r_vals[4], r_vals[5], r_vals[6], r_vals[7]};

        vec8 confidences = {r_vals[8],  r_vals[9],  r_vals[10], r_vals[11],
                            r_vals[12], r_vals[13], r_vals[14], r_vals[15]};

        std::cout << "colors: " << std::endl
                  << colors_red_channel << std::endl
                  << " - confidences: " << confidences << std::endl;

#if 0
        auto out_col_forward = blend_color_red(colors_red_channel, confidences);

        std::cout << "Blended color forward:" << std::endl << out_col_forward(0) << std::endl;
    std::vector<torch::Tensor> t_colors;
    std::vector<torch::Tensor> t_confidences;
    for(int i=0; i<colors_red_channel.size(); ++i){
        t_colors.push_back(conf_to_tensor(colors_red_channel[i]));
        t_confidences.push_back(conf_to_tensor(confidences[i]));
    }

 //   = {conf_to_tensor(colors_red_channel[0]),conf_to_tensor(colors_red_channel[1]),conf_to_tensor(colors_red_channel[2])};
 //   = {conf_to_tensor(confidences[0]),conf_to_tensor(confidences[1]),conf_to_tensor(confidences[2])};

    torch::Tensor t_out_col_forward = t_blend_color(t_colors, t_confidences);
    std::cout << "Blended color tensor forward:" << std::endl << t_out_col_forward << std::endl;
    t_out_col_forward.backward(torch::tensor({1}));

    //for(auto& t : t_colors)
    //{
    //   0 std::cout << "Blended color tensor backwards:" << t.grad()<< std::endl;
    //}
    for(auto& t : t_confidences)
    {
        std::cout << "Blended conf tensor backwards:" << t.grad() << std::endl;
    }
#endif
        Matrix<double, 1, NUM_COL> J_own_col;
        Matrix<double, 1, NUM_COL> J_own_conf;
        Matrix<double, 1, 8> J_numeric_conf;
        Matrix<double, 1, 8> J_numeric_color;
        J_own_col.setZero();
        J_own_conf.setZero();
        J_numeric_conf.setZero();
        J_numeric_color.setZero();

#define EPSILON 1e-3

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
        // if (not_ok)
        //{
        //     return;
        // }
    }

    std::cout << "ALL OK" << std::endl;

    //  return 0;
}