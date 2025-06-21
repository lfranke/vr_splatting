//
// Created by Linus on 14/09/2024.
//
#pragma once
#include "opengl/RealTimeRenderer.h"

#include <iostream>
#include <string>
#include <vector>

#include "viewer_base.h"

#define CHOICES 2

class UserStudy
{
   public:
    struct StudyElement
    {
        std::string scene_name;
        int view_num;
        std::string fov_exp_str;
        std::string gs_exp_str;
        RealTimeRenderer::Experiment fov_ex;
        RealTimeRenderer::Experiment gs_ex;
        StudyElement(std::string sc_name, int v_num, std::string fov_ex_str, std::string gs_ex_str,
                     std::string experiment_base_dir)
            : scene_name(sc_name),
              view_num(v_num),
              fov_exp_str(fov_ex_str),
              gs_exp_str(gs_ex_str),
              fov_ex(experiment_base_dir + "/" + fov_ex_str + "/", fov_ex_str, scene_name),
              gs_ex(experiment_base_dir + "/" + gs_exp_str + "/", gs_exp_str, scene_name)
        {
        }
        friend std::ostream& operator<<(std::ostream& o, StudyElement const& se)
        {
            o << "StudyElement: Scene[" << se.scene_name << "], view number[" << se.view_num << "], fov_exp["
              << se.fov_exp_str << "], gs_exp[" << se.gs_exp_str << "].";
            return o;
        }
    };
    int random_index() { return Saiga::Random::rand() % CHOICES; }

    UserStudy() {}
    UserStudy(ViewerBase* vb, std::string conf_file);
    StudyElement* get_next()
    {
        study_index++;
        if (study_index >= study_elements.size()) return nullptr;
        return &study_elements[study_index];
    }
    std::vector<StudyElement> get_all() { return study_elements; };

   private:
    int study_index = -1;
    ViewerBase* viewer_base;
    std::vector<StudyElement> study_elements;
};