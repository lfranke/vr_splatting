#include "EyeTracking.h"
// code adapted from https://doi.org/10.1111/cgf.14176

#include <../../../External/SRanipal/include/SRanipal.h>

#include <../../../External/SRanipal/include/SRanipal_Enums.h>
#include <../../../External/SRanipal/include/SRanipal_Eye.h>
#include <../../../External/SRanipal/include/SRanipal_Lip.h>

// #include "glm\gtx\vector_angle.hpp"
// #include "glm\gtx\intersect.hpp"
#include <atomic>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
// #pragma comment(lib, "SRanipal.lib")
//  #define FOVE_HEADSET
//  #define VIVE_PRO_EYE_HEADSET
using namespace Saiga;


#ifdef VIVE_PRO_EYE_HEADSET
#    define VPE_ET_V2

using namespace ViveSR;

#endif
#ifdef FOVE_HEADSET

#    include "FoveAPI.h"
#endif
namespace EyeTracking
{
// int w, h;
float z_near = 0.1f;
float z_far  = 1000.f;

#ifdef FOVE_HEADSET
Fove::Headset headset;

Fove::Result<Fove::GazeConvergenceData> gazeOrError;
Fove::Result<Fove::Stereo<Fove::Vec2>> gazeOrError2Eye;
#endif

#ifdef VIVE_PRO_EYE_HEADSET
volatile bool calibFinished = false;

#    ifdef VPE_ET_V2
std::atomic<ViveSR::anipal::Eye::EyeData_v2> eye_data;
#    else
ViveSR::anipal::Eye::EyeData eye_data;
#    endif
volatile bool looping          = false;
std::thread* loopingthread     = nullptr;
std::thread* calibratingThread = nullptr;

#    ifdef VPE_ET_V2
void eye_data_callback(ViveSR::anipal::Eye::EyeData_v2 const& eye_data_t)
{
#        if 0
    static std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();
    auto t2                                           = std::chrono::steady_clock::now();
    auto t_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - time).count()) / 1000.f;
    std::cout << t_ms << std::endl;
    time = t2;
#        endif
    eye_data.store(eye_data_t);
    // float const* gaze = eye_data_t.verbose_data.left.gaze_direction_normalized.elem_;
    // printf("[Eye] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
}
#    else

void eye_data_callback(ViveSR::anipal::Eye::EyeData const& eye_data_t)
{
    eye_data          = eye_data_t;
    float const* gaze = eye_data_t.verbose_data.left.gaze_direction_normalized.elem_;
    printf("[Eye] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
}
#    endif
void streaming()
{
#    ifdef VPE_ET_V2
    ViveSR::anipal::Eye::EyeData_v2 eye_data_thread;
#    else
    ViveSR::anipal::Eye::EyeData eye_data_thread;
#    endif
    std::chrono::steady_clock::time_point start;
    start      = std::chrono::steady_clock::now();
    int result = ViveSR::Error::WORK;
    while (looping)
    {
#    ifdef VPE_ET_V2
        int result = ViveSR::anipal::Eye::GetEyeData_v2(&eye_data_thread);
#    else
        int result = ViveSR::anipal::Eye::GetEyeData(&eye_data_thread);
#    endif

        if (result != ViveSR::Error::WORK)
        {
            // float *gaze = eye_data.verbose_data.left.gaze_direction_normalized.elem_;
            // printf("[Eye] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
            std::cerr << "failed receiving eye tracking data" << std::endl;
        }
        else
        {
            //	printf("TRACK\n");
        }
        eye_data = eye_data_thread;
        std::chrono::steady_clock::time_point end;
        end       = std::chrono::steady_clock::now();
        auto t_ms = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.f;
        std::cout << t_ms << std::endl;
        start = end;
    }
}

#endif
std::shared_ptr<OpenVRWrapper> vrWrapper;

}  // namespace EyeTracking


/*
//FROM NEWER GLM VERSION
template<typename genType>
GLM_FUNC_QUALIFIER bool intersectRayPlane
    (
        genType const& orig, genType const& dir,
        genType const& planeOrig, genType const& planeNormal,
        typename genType::value_type & intersectionDistance
    )
{
    typename genType::value_type d = glm::dot(dir, planeNormal);
    typename genType::value_type Epsilon = std::numeric_limits<typename genType::value_type>::epsilon();

    if (d < -Epsilon)
    {
        intersectionDistance = glm::dot(planeOrig - orig, planeNormal) / d;
        return true;
    }

    return false;
}
*/

//////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////


bool EyeTracking::ETCalibFinished()
{
#ifdef VIVE_PRO_EYE_HEADSET

    calibFinished = true;
    return true;
#else
    return true;
#endif
}

void EyeTracking::calibrateET()
{
#ifdef VIVE_PRO_EYE_HEADSET

    ViveSR::anipal::Eye::LaunchEyeCalibration(ETCalibFinished);

    calibFinished = true;
#else

#endif
}

bool EyeTracking::checkIfCalibFinished()
{
#ifdef VIVE_PRO_EYE_HEADSET

    if (calibFinished)
    {
        if (calibratingThread != nullptr)
        {
            calibratingThread->join();
            std::cout << "Calib Thread Joined" << std::endl;
        }
    }
    return calibFinished;
#else
    return true;
#endif
}


void EyeTracking::update()
{
#ifdef FOVE_HEADSET

    gazeOrError     = headset.getGazeConvergence();
    gazeOrError2Eye = headset.getGazeVectors2D();
#endif
}

void EyeTracking::initEyeTracking(std::shared_ptr<OpenVRWrapper> vrWrapperPtr)
{
    vrWrapper = vrWrapperPtr;
#ifdef FOVE_HEADSET

    headset = Fove::Headset::create(Fove::ClientCapabilities::Orientation | Fove::ClientCapabilities::Position |
                                    Fove::ClientCapabilities::Gaze)
                  .getValue();
#endif

#ifdef VIVE_PRO_EYE_HEADSET

    if (!ViveSR::anipal::Eye::IsViveProEye())
    {
        std::cout << "\n\nthis device does not have eye-tracker, please change your HMD\n" << std::endl;
        return;
    }
    std::cout << "Start initializing VR Eye Tracking" << std::endl;
#    ifdef VPE_ET_V2
    int error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE_V2, NULL);

#    else
    int error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
#    endif
    if (error == ViveSR::Error::WORK)
    {
        std::cout << "Successfully initialize Eye engine.\n" << std::endl;
    }
    else if (error == ViveSR::Error::RUNTIME_NOT_FOUND)
    {
        std::cout << "please follows SRanipal SDK guide to install SR_Runtime first\n" << std::endl;
        exit(13);
    }
    else
    {
        std::cout << "Fail to initialize Eye engine. please refer the code " << error << " of ViveSR::Error.\n"
                  << std::endl;
        exit(13);
    }

    //  calibratingThread = new std::thread(EyeTracking::calibrateET);

    ViveSR::anipal::Eye::RegisterEyeDataCallback_v2(eye_data_callback);

    /*if (loopingthread == nullptr)
    {
        looping       = true;
        loopingthread = new std::thread(EyeTracking::streaming);
        if (loopingthread)
        {
            std::cout << "ET THREAD START" << std::endl;
        }
        else
        {
            std::cout << "no ET THREAD" << std::endl;
        }
    }*/



#endif

    // w = width;
    // h = height;
    // headset.startEyeTrackingCalibration(true);

    // std::cout << "[Eye Tracking] Initialized, resolution: " << w << "x" << h << std::endl;
}

void EyeTracking::destroyTracker()
{
#ifdef VIVE_PRO_EYE_HEADSET
    looping = false;
    if (loopingthread != nullptr)
    {
        loopingthread->join();
        delete loopingthread;
        loopingthread = nullptr;
    }
    ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE);
    ViveSR::anipal::Release(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP);
#endif
}



vec2 EyeTracking::getEyeLookPositionInPixelsForEye(int eye, int wid, int hei)
{
    vec2 resHalf = vec2(wid / 2, hei / 2);

    static vec2 eye_pos_static[2] = {resHalf, resHalf};
#ifdef FOVE_HEADSET

    Fove::Vec2 left, right;


    // fove doesnt really care
    return getEyeLookPositionInPixels();


    if (gazeOrError2Eye.isValid())
    {
        Fove::Stereo<Fove::Vec2> gaze = gazeOrError2Eye.getValue();
        // gaze vectors [-1,1], with (0,0) center
        if (eye == 0)
        {
            return glm::vec2(gaze.l.x, gaze.l.y) * resHalf + resHalf;
        }
        else
        {
            return glm::vec2(gaze.r.x, gaze.r.y) * resHalf + resHalf;
        }
    }
#endif


#ifdef VIVE_PRO_EYE_HEADSET

    auto eye_data_tl = eye_data.load();
    vec3 gaze;
    if (eye == 0)
    {
        gaze.x() = -eye_data_tl.verbose_data.left.gaze_direction_normalized.x;
        gaze.y() = eye_data_tl.verbose_data.left.gaze_direction_normalized.y;
        gaze.z() = eye_data_tl.verbose_data.left.gaze_direction_normalized.z;
    }
    else
    {
        gaze.x() = -eye_data_tl.verbose_data.right.gaze_direction_normalized.x;
        gaze.y() = eye_data_tl.verbose_data.right.gaze_direction_normalized.y;
        gaze.z() = eye_data_tl.verbose_data.right.gaze_direction_normalized.z;
    }
    bool valid_measure = gaze.x() != 0 || gaze.y() != 0 || gaze.z() != 0;

    vec2 res_eye = eye_pos_static[eye];
    if (valid_measure) res_eye = vec2(gaze.x() * resHalf.x() + resHalf.x(), -gaze.y() * resHalf.y() + resHalf.y());

    eye_pos_static[eye] = res_eye;

    return res_eye;
    // gaze[0] = -eye_data.verbose_data.combined.eye_data.gaze_direction_normalized.x;
    // gaze[1] = eye_data.verbose_data.combined.eye_data.gaze_direction_normalized.y;
    // gaze[2] = eye_data.verbose_data.combined.eye_data.gaze_direction_normalized.z;
    gaze = gaze.normalize();

    // printf("[Eye] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
    // result_pos = glm::vec2(eye_data.verbose_data.combined.eye_data.pupil_position_in_sensor_area.x,
    // eye_data.verbose_data.combined.eye_data.pupil_position_in_sensor_area.y)* glm::vec2(w, h);

    mat4 proj = vrWrapper->GetHMDProjectionMatrix((vr::Hmd_Eye)eye, z_near, z_far);

    // std::cout << w << "x" << h << "; " << resHalf.x() << " " << resHalf.y() << std::endl;
    // project point in frustum to image plane
    float distance_proj_point = (z_far * 0.5f);
    vec4 gaze_hom =
        vec4(gaze[0] * distance_proj_point, gaze[1] * distance_proj_point, gaze[2] * distance_proj_point, 1);
    vec4 projected_gaze = proj * (gaze_hom);
    // std::cout <<"prehom: " <<projected_gaze.x()  <<" "<<projected_gaze.y()  <<" "<<projected_gaze.z()  <<"
    // "<<projected_gaze.w()  <<" " << std::endl;
    projected_gaze /= projected_gaze.w();
    // std::cout <<"posthom: " <<projected_gaze.x()  <<" "<<projected_gaze.y()  <<" "<<projected_gaze.z()  <<"
    // "<<projected_gaze.w()  <<" " << std::endl;


    vec2 result = vec2(projected_gaze.x() * resHalf.x(), projected_gaze.y() * resHalf.y()) + resHalf;
    if (result.x() != result.x()) result = resHalf;

    return result;

    //	return glm::vec2(gaze[0], gaze[1])*resHalf + resHalf;

#endif
    return resHalf;
}

vec2 EyeTracking::getEyeLookPositionInPixels(int w, int h)
{
    SAIGA_ASSERT(false);

#ifdef FOVE_HEADSET

    if (!ukeys.pauseTracking)
    {
        // return glm::vec2(800, 900);
        // const Fove::Result<Stereo<GazeVector>> gazeOrError = headset.getGazeVectors();

        if (gazeOrError.isValid())
        {
            Fove::Ray ray = gazeOrError->ray;
            // if(abs(ray.direction.x)>0.05 || abs(ray.direction.y) > 0.05)
            //	std::cout << ray.direction.x << ", " << ray.direction.y <<", " <<ray.direction.z << std::endl;

            glm::vec3 near_plane_normal = glm::vec3(0, 0, -1);
            glm::vec3 near_plane_origin = glm::vec3(0, 0, hmdCams->m_fNearClip);
            glm::vec3 gaze_origin       = glm::vec3(0, 0, 0);
            glm::vec3 gaze_dir          = glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z);

            float intersect_t = 0.f;
            bool intersected =
                intersectRayPlane(gaze_origin, gaze_dir, near_plane_origin, near_plane_normal, intersect_t);
            glm::vec3 intersect_point = gaze_origin + gaze_dir * intersect_t;
            //	std::cout << intersect_point.x << "," << intersect_point.y << ", " << intersect_point.z << std::endl;

            // JUST LEFT EYE ATM
            glm::mat4 proj    = hmdCams->P(0);
            float aspect      = proj[1][1] / proj[0][0];
            float fovy        = 2.0 * atan(1.0 / proj[1][1]);
            float fovy_degree = fovy * 180.0 / M_PI;

            float fovx        = aspect * fovy;
            float fovx_degree = fovx * 180.0 / M_PI;

            float res2y      = tan(fovy / 2.0) * hmdCams->m_fNearClip;
            float pixelSizeY = res2y / (h / 2);

            float res2x      = tan(fovx / 2.0) * hmdCams->m_fNearClip;
            float pixelSizeX = res2x / (w / 2);
            // if (abs(ray.direction.x) > 0.05 || abs(ray.direction.y) > 0.05) {
            // std::cout << "asp: " << aspect << " , fovy " << fovy_degree << ", fovx? " << fovx_degree << std::endl;
            // std::cout << "pixelSizeY " << pixelSizeY << "pixelSizeX " << pixelSizeX << std::endl;

            float pixels_y = intersect_point.y / pixelSizeY;
            float pixels_x = intersect_point.x / pixelSizeX;

            // std::cout << "pixelsx " << pixels_x << "pixelsy " << pixels_y << std::endl;
            result_pos = glm::vec2(w / 2 + pixels_x, h / 2 + pixels_y);
            //}
            static_result_pos = result_pos;
        }
        getFoveaAngle();
    }

    result_pos = static_result_pos;

    return result_pos;

#endif

#ifdef VIVE_PRO_EYE_HEADSET

    auto eye_data_tl = eye_data.load();
    float gaze[3];
    gaze[0] = -eye_data_tl.verbose_data.combined.eye_data.gaze_direction_normalized.x;
    gaze[1] = eye_data_tl.verbose_data.combined.eye_data.gaze_direction_normalized.y;
    gaze[2] = eye_data_tl.verbose_data.combined.eye_data.gaze_direction_normalized.z;


    int eye   = 0;
    mat4 proj = vrWrapper->GetHMDProjectionMatrix((vr::Hmd_Eye)eye, z_near, z_far);

    vec4 gaze_hom = vec4(gaze[0], gaze[1], gaze[2], 1);

    vec2 resHalf = vec2(w / 2, h / 2);

    vec2 result_pos =
        vec2(gaze[0] * resHalf.x(), gaze[1] * resHalf.y()) +
        resHalf;  // glm::vec2(eye_data.verbose_data.combined.eye_data.pupil_position_in_sensor_area.x,
                  // eye_data.verbose_data.combined.eye_data.pupil_position_in_sensor_area.y)* glm::vec2(w, h);

    // pupil_position_in_sensor_area
    return result_pos;

#endif
}


float EyeTracking::getFoveaAngleInPixels()
{
#if 0
    //float near_plane = hmdCams->m_fNearClip;
    int eye = 0;
    mat4 projectionMatrix = vrWrapper->GetHMDProjectionMatrix((vr::Hmd_Eye)eye, z_near, z_far);

    float aspectRatio = projectionMatrix(1,1) / projectionMatrix(0,0)
    //std::cout << "aspect: " << aspectRatio << std::endl;

    float fov_y = 2.0*atan(1.0 / projectionMatrix(1,1)) * 180.0 / 3.1415f;
    //std::cout << "fov_y: " << fov_y << std::endl;
    float fov_x = fov_y*aspectRatio;
    //std::cout << "fov_x: " << fov_x << std::endl;

    //std::cout << "resolution: " << HMD_screen_width << " x " << HMD_screen_height << std::endl;

    float angle_per_pixel = fov_y / float(HMD_screen_height);
    //std::cout << "Y: angle per pixel " << fov_y/float(HMD_screen_height) <<std::endl;
    //std::cout << "X: angle per pixel " << fov_x / float(HMD_screen_width) << std::endl;

    float extend_in_pixels = ukeys.fovea_angle_in_degree / angle_per_pixel;

    ukeys.fovea_extend = extend_in_pixels;

    return ukeys.fovea_extend;
#endif
    std::cout << "not implemented" << std::endl;
    return 100;
}

float EyeTracking::getFoveaAngle()
{
    float angle = 20.f;

#ifdef FOVE_HEADSET
    //	const Fove::Result<Fove::GazeConvergenceData> gazeOrError = headset.getGazeConvergence();
    if (gazeOrError.isValid())
    {
        Fove::Ray ray         = gazeOrError->ray;
        glm::vec3 gaze_vector = glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z);
        angle                 = glm::angle(glm::normalize(gaze_vector), glm::vec3(0, 0, 1));
        //	if(angle > 5)
        //	std::cout << angle << std::endl;
    }
#endif


#ifdef VIVE_PRO_EYE_HEADSET

#endif
    return angle;
}
