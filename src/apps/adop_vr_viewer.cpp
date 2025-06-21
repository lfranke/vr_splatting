/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/rendering/VRRendering/VRRenderer.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/window/WindowTemplate.h"
#include "saiga/opengl/window/glfw_window.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/opengl/world/proceduralSkybox.h"

#include "../lib/opengl/EyeTracking.h"
#include "UserStudy.h"

// #include "viewer_base.h"
using namespace Saiga;
class DoubleFastVRRenderer : public VRRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = VRRenderingParameters;
    VRRenderingParameters params;
    ViewPort viewport;
    DoubleFastVRRenderer(OpenGLWindow& window, const VRRenderingParameters& params = VRRenderingParameters())
        : VRRenderer(window, params)
    {
    }
    virtual ~DoubleFastVRRenderer() {}
    virtual void render(const RenderInfo& renderInfo) override
    {
        if (!rendering) return;
        //    std::cout << "FAST VR RENDERER"<< std::endl;
        SAIGA_ASSERT(rendering);
        auto camera = dynamic_cast<PerspectiveCamera*>(renderInfo.camera);
        SAIGA_ASSERT(camera);
        PrepareImgui(false);
        VR().update();
        {
            SAIGA_ASSERT(timer);
            timer->BeginFrame();
            auto [cameraLeft, cameraRight]         = VR().getEyeCameras(*camera);
            RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
            SAIGA_ASSERT(renderingInterface);
            viewport_offset = ivec2(0, 0);
            viewport_size   = ivec2(VR().renderWidth(), VR().renderHeight());
            viewport        = ViewPort(viewport_offset, viewport_size);
            Resize(viewport.size.x(), viewport.size.y());
            RenderInfo rInfo({&cameraLeft, RenderPass::Final});
            std::vector<PerspectiveCamera> eyes_mats = std::vector<PerspectiveCamera>(2);
            eyes_mats[0]                             = cameraLeft;
            eyes_mats[1]                             = cameraRight;

            rInfo.camera = eyes_mats.data();
            // static std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
            // time                                              = std::chrono::steady_clock::now();
            {
                renderingInterface->render(rInfo);
                glFinish();
            }
            {
                VR().submitImage(vr::Hmd_Eye::Eye_Left, get_Tex(0).get());
                VR().submitImage(vr::Hmd_Eye::Eye_Right, get_Tex(1).get());
                glFlush();
                // vr::VRCompositor()->PostPresentHandoff();
            }
            // glFinish();
            // auto t2   = std::chrono::steady_clock::now();
            // auto t_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - time).count()) / 1000.f;
            // std::cout << t_ms << std::endl;
            // time = t2;

            timer->EndFrame();
            {
            }
        }
        if (imgui)
        {
            if (editor_gui.enabled && render_viewport)
            {
                ImGui::Begin("3DView");
                ImGui::BeginChild("viewer_child");
                auto size = ImGui::GetWindowContentRegionMax();
                size.x    = size.x / 2 - 2;
                ImGui::Texture(get_Tex(0).get(), size, true);
                ImGui::SameLine();
                ImGui::Texture(get_Tex(1).get(), size, true);
                ImGui::EndChild();
                ImGui::End();
            }
            // The imgui frame is now done
            // -> Render it to the screen (default FB)
            imgui->endFrame();
            default_framebuffer.bind();
            imgui->render();
        }
        if (params.useGlFinish) glFinish();
    }
};

class ADOPVRViewer : public StandaloneWindow<WindowManagement::GLFW, DoubleFastVRRenderer>,
                     public glfw_KeyListener,
                     ViewerBase
{
    UserStudy user_study;
    bool swap_scene           = false;
    bool swap_scene_show_gray = false;

    bool swap_experiment           = false;
    bool swap_experiment_show_gray = false;
    std::shared_ptr<Texture> grey_vr_tex[2];
    std::ofstream user_study_file;
    UserStudy::StudyElement* current_element;



   public:
    void loadSceneFromUserStudy(UserStudy::StudyElement exp, int id = 0)
    {
        LoadScene("scenes/" + exp.scene_name);
        LoadSceneImpl();
        neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene, false);
        neural_renderer->experiments.clear();
        neural_renderer->experiments.push_back(exp.fov_ex);
        neural_renderer->experiments.push_back(exp.gs_ex);

        neural_renderer->current_ex = id;
        neural_renderer->current_ep = neural_renderer->experiments[neural_renderer->current_ex].eps.size() - 1;
        neural_renderer->LoadNets();


        auto& f = scene->scene->frames[exp.view_num];

        camera->setModelMatrix(f.OpenglModel());
        camera->updateFromModel();
    }

    void swapSceneUserStudy()
    {
        int id                      = (neural_renderer->current_ex + 1) % 2;
        neural_renderer->current_ex = id;
        neural_renderer->current_ep = neural_renderer->experiments[neural_renderer->current_ex].eps.size() - 1;
    }

    void switchToNextUserScene() { swap_experiment = true; }

    void UserStudyRecordDecision()
    {
        user_study_file << current_element->scene_name << ", " << current_element->view_num << ", ";

        int choice_of_mode = neural_renderer->current_ex;
        user_study_file << ((choice_of_mode == 0) ? "fov" : "gs");
        user_study_file << std::endl;
        user_study_file.flush();
        switchToNextUserScene();
    }

    ADOPVRViewer(std::string scene_dir) : StandaloneWindow("config.ini")
    {
        main_menu.AddItem("Saiga", "MODEL", [this]() { view_mode = ViewMode::MODEL; }, GLFW_KEY_F1, "F1");
        main_menu.AddItem("Saiga", "NEURAL", [this]() { view_mode = ViewMode::NEURAL; }, GLFW_KEY_F2, "F2");

        EyeTracking::initEyeTracking(this->renderer->VR_ptr());

        LoadScene(scene_dir);
        LoadSceneImpl();


        view_mode = ViewMode::NEURAL;
        std::cout << "Program Initialized!" << std::endl;

        static bool USER_STUDY = false;
        if (!USER_STUDY) return;

        user_study = UserStudy(this, "user_study.txt");

        if (!neural_renderer)
        {
            neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
        }
        auto user_exps = user_study.get_all();
        for (auto& i : user_exps)
        {
            std::cout << "Test: " << i << std::endl;
            std::cout << "test fov" << std::endl;

            loadSceneFromUserStudy(i);
            std::cout << "test gs" << std::endl;
            loadSceneFromUserStudy(i, 1);
        }

        int w = this->renderer->VR_ptr()->renderWidth();
        int h = this->renderer->VR_ptr()->renderHeight();
        TemplatedImage<vec4> img;
        img.create(h, w);
        img.getImageView().set(vec4(0.5, 0.5, 0.5, 1));
        for (int i = 0; i < 2; ++i)
        {
            grey_vr_tex[i] = std::make_shared<Texture>(img);
        }

        // https://stackoverflow.com/a/12468109
        auto random_string = [](size_t length)
        {
            auto randchar = []() -> char
            {
                const char charset[] =
                    "0123456789"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz";
                const size_t max_index = (sizeof(charset) - 1);
                return charset[rand() % max_index];
            };
            std::string str(length, 0);
            std::generate_n(str.begin(), length, randchar);
            return str;
        };

        std::string userfilename = Saiga::CurrentTimeString("%F_%H-%M-%S") + random_string(8);
        user_study_file.open("userstudy/" + userfilename);
        user_study_file << Saiga::CurrentTimeString("%F_%H-%M-%S") << std::endl;
        user_study_file.flush();
        current_element = user_study.get_next();

        loadSceneFromUserStudy(*current_element, user_study.random_index());
    }



    ~ADOPVRViewer() {}

    void LoadSceneImpl()
    {
        ::camera = &scene->scene_camera;
        window->setCamera(camera);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;

        auto& f = scene->scene->frames.front();
        ::camera->setModelMatrix(f.OpenglModel());
        ::camera->updateFromModel();
        // ::camera->setView(vec3(0.58,1.54,1.22), vec3(1,0,0), vec3(-0.100000, -0.9950000, -0.015000));
        // ::camera->getUpVector();


        renderer->lighting.directionalLights.clear();
        sun                   = std::make_shared<DirectionalLight>();
        sun->ambientIntensity = exp2(scene->scene->dataset_params.scene_exposure_value);
        sun->intensity        = 0;
        renderer->lighting.AddLight(sun);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;

        // glfwSwapInterval(0);
    }

    void update(float dt) override
    {
        int FORWARD = keyboard.getKeyState(GLFW_KEY_W) - keyboard.getKeyState(GLFW_KEY_S);
        {
            float speed = dt * FORWARD;
            if (keyboard.getKeyState(GLFW_KEY_LEFT_SHIFT))
            {
                speed *= 5;
            }
            vec3 dir                   = camera->rot * renderer->VR().LookingDirection();
            camera->position.head<3>() = camera->position.head<3>() + dir * speed;
            camera->calculateModel();
        }

        if (keyboard.getKeyState(GLFW_KEY_SPACE))
        {
            swap_scene = true;
            std::cout << "SWAP!!!" << std::endl;
        }
    }
    void interpolate(float dt, float interpolation) override
    {
        if (renderer->use_mouse_input_in_3dview || renderer->use_mouse_input_in_3dview)
        {
            scene->scene_camera.interpolate(dt, interpolation);
        }
    }
    enum VR_EYES
    {
        LEFT  = 0,
        RIGHT = 1,
        COUNT = 2
    };

    void render(RenderInfo render_info) override
    {
        // if (view_mode == ViewMode::MODEL && render_info.render_pass == RenderPass::Deferred)
        //{
        //     if (!object_tex)
        //     {
        //         object_tex = std::make_shared<TexturedAsset>(scene->model);
        //     }
        //     object_tex->render(render_info.camera, mat4::Identity());
        // }

        if (render_info.render_pass == RenderPass::Final && view_mode == ViewMode::NEURAL)
        {
            auto show_gray = [&]()
            {
                for (auto eye : {VR_EYES::LEFT, VR_EYES::RIGHT})
                {
                    renderer->get_FBO(eye).bind();
                    display.render(grey_vr_tex[0].get(), ivec2(0, 0), renderer->viewport_size, true);
                    renderer->get_FBO(eye).unbind();
                }
            };
            if (swap_scene)
            {
                show_gray();
                swap_scene           = false;
                swap_scene_show_gray = true;
                return;
            }
            if (swap_scene_show_gray)
            {
                swapSceneUserStudy();
                neural_renderer->LoadNets();
                swap_scene_show_gray = false;
                return;
            }
            if (swap_experiment)
            {
                show_gray();
                swap_experiment_show_gray = true;
                swap_experiment           = false;
                return;
            }
            if (swap_experiment_show_gray)
            {
                current_element = user_study.get_next();
                if (current_element == nullptr)
                {
                    user_study_file.close();
                    exit(1);
                }
                loadSceneFromUserStudy(*current_element, user_study.random_index());
                swap_experiment_show_gray = false;
                return;
            }

            torch::NoGradGuard ngg;
            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
                // neural_renderer->tone_mapper = &renderer->tone_mapper;
                neural_renderer->timer_system = new CUDA::CudaTimerSystem();
                std::cout << "renderer init" << std::endl;
            }
            if (neural_renderer->timer_system) neural_renderer->timer_system->BeginFrame();
            std::vector<ImageInfo> fds;
            std::vector<vec2> foveas;
            for (auto eye : {VR_EYES::LEFT, VR_EYES::RIGHT})
            {
                auto fd = scene->CurrentFrameData();
                // std::cout << render_info.camera->model << std::endl;
                fd.eye = eye;

                fd.w = renderer->viewport_size.x() * render_scale;
                fd.h = renderer->viewport_size.y() * render_scale;
                // fd.w = iAlignUp(fd.w, 32);
                // fd.h = iAlignUp(fd.h, 32);
                PerspectiveCamera* cameras_ptr = (PerspectiveCamera*)&(*render_info.camera);
                PerspectiveCamera cam          = cameras_ptr[eye];
                fd.distortion                  = Distortionf();

                fd.K    = GLProjectionMatrix2CVCamera(cam.proj, fd.w, fd.h);
                fd.pose = Sophus::SE3f::fitToSE3(cam.model * GL2CVView()).cast<double>();



                fd.proj_mat  = cam.proj;
                float z_flip = -1.f;

                fd.proj_mat(3, 2) *= z_flip;
                fd.proj_mat(2, 2) *= z_flip;
                fd.proj_mat(1, 2) *= z_flip;
                fd.proj_mat(0, 2) *= z_flip;
                fd.own_proj_mat = true;

                foveas.push_back(EyeTracking::getEyeLookPositionInPixelsForEye(eye, fd.w, fd.h));
                // std::cout << "eyepos: " << fovea.x() << " " << fovea.y() << std::endl;

                fd.exposure_value = renderer->tone_mapper.params.exposure_value;
                fd.white_balance  = renderer->tone_mapper.params.white_point;
                fds.push_back(fd);
            }
            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;

            // neural_renderer->Render(fd, fovea);
            {
                neural_renderer->RenderTwo(fds[0], fds[1], foveas[0], foveas[1]);
            }

            for (auto eye : {VR_EYES::LEFT, VR_EYES::RIGHT})
            {
                renderer->get_FBO(eye).bind();

                if (neural_renderer->use_gl_tonemapping)
                {
                    display.render(neural_renderer->vr_output_ldr[eye].get(), ivec2(0, 0), renderer->viewport_size,
                                   true);
                }
                else
                {
                    display.render(neural_renderer->vr_output[eye].get(), ivec2(0, 0), renderer->viewport_size, true);
                }
                renderer->get_FBO(eye).unbind();
            }

            if (neural_renderer->timer_system) neural_renderer->timer_system->EndFrame();

            // if (neural_renderer->use_gl_tonemapping)
            //{
            //     display.render(neural_renderer->output_texture_ldr.get(), ivec2(0, 0), renderer->viewport_size,
            //                    true);
            // }
            // else
            //{
            //     display.render(neural_renderer->output_texture.get(), ivec2(0, 0), renderer->viewport_size, true);
            // }
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::Begin("Model Viewer");
            ImGui::SliderFloat("render_scale", &render_scale, 0.01, 1);

            if (ImGui::Button("startcalib"))
            {
                EyeTracking::calibrateET();
            }
            ImGui::End();
            ViewerBase::imgui();
        }
    }


    void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                window->close();
                break;
            case GLFW_KEY_ENTER:
            {
                UserStudyRecordDecision();
                break;
            }
            default:
                break;
        }
    }

   private:
    TextureDisplay display;
    std::shared_ptr<DirectionalLight> sun;

    float render_scale = 1.0;  // 0.15;
    ViewMode view_mode = ViewMode::MODEL;
};



int main(int argc, char* argv[])
{
    std::string scene_dir;
    CLI::App app{"ADOP VR Viewer for Scenes", "adop_vr_viewer"};
    app.add_option("--scene_dir", scene_dir)->required();
    CLI11_PARSE(app, argc, argv);

    initSaigaSample();
    ADOPVRViewer window(scene_dir);
    window.run();
    return 0;
}
#if 0

/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#    include "saiga/core/imgui/imgui.h"
#    include "saiga/core/model/model_from_shape.h"
#    include "saiga/core/util/commandLineArguments.h"
#    include "saiga/opengl/assets/all.h"
#    include "saiga/opengl/rendering/VRRendering/VRRenderer.h"
#    include "saiga/opengl/rendering/renderer.h"
#    include "saiga/opengl/shader/shaderLoader.h"
#    include "saiga/opengl/window/SampleWindowForward.h"
#    include "saiga/opengl/window/WindowTemplate.h"
#    include "saiga/opengl/window/glfw_window.h"
#    include "saiga/opengl/world/TextureDisplay.h"
#    include "saiga/opengl/world/proceduralSkybox.h"

#    include "viewer_base.h"
using namespace Saiga;

class SAIGA_OPENGL_API DoubleFastVRRenderer : public VRRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = VRRenderingParameters;
    VRRenderingParameters params;
    ViewPort viewport;
    DoubleFastVRRenderer(OpenGLWindow& window, const VRRenderingParameters& params = VRRenderingParameters())
        : VRRenderer(window, params)
    {
    }
    virtual ~DoubleFastVRRenderer() {}
    virtual void render(const RenderInfo& renderInfo) override
    {
        if (!rendering) return;
        //    std::cout << "FAST VR RENDERER"<< std::endl;
        SAIGA_ASSERT(rendering);
        auto camera = dynamic_cast<PerspectiveCamera*>(renderInfo.camera);
        SAIGA_ASSERT(camera);
        PrepareImgui(false);
        VR().update();
        {
            SAIGA_ASSERT(timer);
            timer->BeginFrame();
            auto [cameraLeft, cameraRight]         = VR().getEyeCameras(*camera);
            RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
            SAIGA_ASSERT(renderingInterface);
            viewport_offset = ivec2(0, 0);
            viewport_size   = ivec2(VR().renderWidth(), VR().renderHeight());
            viewport        = ViewPort(viewport_offset, viewport_size);
            Resize(viewport.size.x(), viewport.size.y());
            RenderInfo rInfo({&cameraLeft, RenderPass::Final});
            std::vector<PerspectiveCamera> eyes_mats = std::vector<PerspectiveCamera>(2);
            eyes_mats[0]                             = cameraLeft;
            eyes_mats[1]                             = cameraRight;

            rInfo.camera = eyes_mats.data();
            {
                renderingInterface->render(rInfo);
                glFinish();
            }
            {
                VR().submitImage(vr::Hmd_Eye::Eye_Left, textures[0].get());
                VR().submitImage(vr::Hmd_Eye::Eye_Right, textures[1].get());
            }
            timer->EndFrame();
            {
            }
        }
        if (imgui)
        {
            if (editor_gui.enabled && render_viewport)
            {
                ImGui::Begin("3DView");
                ImGui::BeginChild("viewer_child");
                auto size = ImGui::GetWindowContentRegionMax();
                size.x    = size.x / 2 - 2;
                ImGui::Texture(textures[0].get(), size, true);
                ImGui::SameLine();
                ImGui::Texture(textures[1].get(), size, true);
                ImGui::EndChild();
                ImGui::End();
            }
            // The imgui frame is now done
            // -> Render it to the screen (default FB)
            imgui->endFrame();
            default_framebuffer.bind();
            imgui->render();
        }
        if (params.useGlFinish) glFinish();
    }
};


class ADOPVRViewer : public StandaloneWindow<WindowManagement::GLFW, DoubleFastVRRenderer>,
                     public glfw_KeyListener,
                     ViewerBase
{
   public:
    ADOPVRViewer(std::string scene_dir) : StandaloneWindow("config.ini")
    {
        main_menu.AddItem("Saiga", "MODEL", [this]() { view_mode = ViewMode::MODEL; }, GLFW_KEY_F1, "F1");

        main_menu.AddItem("Saiga", "NEURAL", [this]() { view_mode = ViewMode::NEURAL; }, GLFW_KEY_F2, "F2");

        LoadScene(scene_dir);
        LoadSceneImpl();

        auto& f = scene->scene->frames[18];
        camera->setModelMatrix(f.OpenglModel());
        camera->updateFromModel();

        view_mode = ViewMode::NEURAL;
        std::cout << "Program Initialized!" << std::endl;
    }
    ~ADOPVRViewer() {}

    void LoadSceneImpl()
    {
        ::camera = &scene->scene_camera;
        window->setCamera(camera);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;

        auto& f = scene->scene->frames.front();
        camera->setModelMatrix(f.OpenglModel());
        camera->updateFromModel();

        renderer->lighting.directionalLights.clear();
        sun                   = std::make_shared<DirectionalLight>();
        sun->ambientIntensity = exp2(scene->scene->dataset_params.scene_exposure_value);
        sun->intensity        = 0;
        renderer->lighting.AddLight(sun);
        renderer->tone_mapper.params.exposure_value = scene->scene->dataset_params.scene_exposure_value;
    }

    void update(float dt) override
    {
        int FORWARD = keyboard.getKeyState(GLFW_KEY_W) - keyboard.getKeyState(GLFW_KEY_S);
        {
            float speed = dt * FORWARD;
            if (keyboard.getKeyState(GLFW_KEY_LEFT_SHIFT))
            {
                speed *= 5;
            }
            vec3 dir                   = camera->rot * renderer->VR().LookingDirection();
            camera->position.head<3>() = camera->position.head<3>() + dir * speed;
            camera->calculateModel();
        }
    }
    void interpolate(float dt, float interpolation) override
    {
        if (renderer->use_mouse_input_in_3dview || renderer->use_mouse_input_in_3dview)
        {
            scene->scene_camera.interpolate(dt, interpolation);
        }
    }

    enum VR_EYES
    {
        LEFT  = 0,
        RIGHT = 1,
        COUNT = 2
    };

    void render(RenderInfo render_info) override
    {
#    if 0
        if (view_mode == ViewMode::MODEL && render_info.render_pass == RenderPass::Deferred)
        {
            if (!object_tex)
            {
                object_tex = std::make_shared<TexturedAsset>(scene->model);
            }
            object_tex->render(camera, mat4::Identity());
        }
#    endif
        if (render_info.render_pass == RenderPass::Final && view_mode == ViewMode::NEURAL)
        {
            torch::NoGradGuard ngg;
            if (!neural_renderer)
            {
                neural_renderer = std::make_unique<RealTimeRenderer>(scene->scene);
                // neural_renderer->tone_mapper = &renderer->tone_mapper;
            }

            neural_renderer->timer_system.BeginFrame();
            std::vector<ImageInfo> fds;
            for (auto eye : {VR_EYES::LEFT, VR_EYES::RIGHT})
            {
                auto fd = scene->CurrentFrameData();
                fd.w    = renderer->viewport_size.x() * render_scale;
                fd.h    = renderer->viewport_size.y() * render_scale;
                // fd.w = iAlignUp(fd.w, 32);

                // fd.h = iAlignUp(fd.h, 32);

                PerspectiveCamera* cameras_ptr = (PerspectiveCamera*)&(*render_info.camera);
                fd.distortion                  = Distortionf();

                fd.K    = GLProjectionMatrix2CVCamera(cameras_ptr[eye].proj, fd.w, fd.h);
                fd.pose = Sophus::SE3f::fitToSE3(cameras_ptr[eye].model * GL2CVView()).cast<double>();

                fd.exposure_value = renderer->tone_mapper.params.exposure_value;
                fd.white_balance  = renderer->tone_mapper.params.white_point;

                fds.push_back(fd);
            }

            neural_renderer->tone_mapper.params      = renderer->tone_mapper.params;
            neural_renderer->tone_mapper.tm_operator = renderer->tone_mapper.tm_operator;
            neural_renderer->tone_mapper.params.exposure_value -= scene->scene->dataset_params.scene_exposure_value;
            neural_renderer->tone_mapper.params_dirty = true;

            for (auto eye : {VR_EYES::LEFT, VR_EYES::RIGHT})
            {
                auto fd = fds[eye];
                neural_renderer->Render(fd);
            }
            if (neural_renderer->use_gl_tonemapping)
            {
                display.render(neural_renderer->output_texture_ldr.get(), ivec2(0, 0), renderer->viewport_size, true);
            }
            else
            {
                display.render(neural_renderer->output_texture.get(), ivec2(0, 0), renderer->viewport_size, true);
            }

            neural_renderer->timer_system.EndFrame();
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::Begin("Model Viewer");
            ImGui::SliderFloat("render_scale", &render_scale, 0.01, 1);
            ImGui::End();
            ViewerBase::imgui();
        }
    }


    void keyPressed(int key, int scancode, int mods) override
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                window->close();
                break;
            default:
                break;
        }
    }

   private:
    TextureDisplay display;
    std::shared_ptr<DirectionalLight> sun;

    float render_scale = 1.0;
    ViewMode view_mode = ViewMode::MODEL;
};



int main(int argc, char* argv[])
{
    std::string scene_dir;
    CLI::App app{"ADOP VR Viewer for Scenes", "adop_vr_viewer"};
    app.add_option("--scene_dir", scene_dir)->required();
    CLI11_PARSE(app, argc, argv);

    initSaigaSample();
    ADOPVRViewer window(scene_dir);
    window.run();
    return 0;
}
#endif