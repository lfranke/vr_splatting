//
// Created by Linus on 14/09/2024.
//
#include "UserStudy.h"

#include <fstream>
#include <sstream>

UserStudy::UserStudy(ViewerBase* vb, std::string conf_file) : viewer_base(vb)
{
    std::ifstream strm(conf_file);

    std::string line;
    while (std::getline(strm, line))
    {
        std::stringstream sstream(line);
        if (line.empty() || line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;

        std::string name;
        int view_num;
        std::string fov_exp_str;
        std::string gs_exp_str;

        sstream >> name >> view_num >> fov_exp_str >> gs_exp_str;

        study_elements.emplace_back(name, view_num, fov_exp_str, gs_exp_str, "experiments/");
        // std::cout << "Study list: " << name << " : " << view_num << std::endl;
        std::cout << study_elements.back() << std::endl;
    }
}


/*
void UserStudy::check_all(RealTimeRenderer& rtr)
{
    for (auto& i : study_elements)
    {
        std::cout << "Test: " << i << std::endl;
        std::cout << "test fov" << std::endl;
        viewer_base->LoadScene(i.scene_name, false);
        rtr.scene = viewer_base->scene;
        if (rtr.experiments.empty())
            rtr.experiments.push_back(i.fov_ex);
        else
            rtr.experiments.front() = i.fov_ex;
        rtr.current_ex = 0;
        rtr.current_ep = i.fov_ex.eps.size() - 1;

        rtr.LoadNets();
        std::cout << "test gs" << std::endl;
        rtr.experiments.front() = i.gs_ex;
        rtr.current_ep          = i.gs_ex.eps.size() - 1;

        rtr.LoadNets();
    }
}*/