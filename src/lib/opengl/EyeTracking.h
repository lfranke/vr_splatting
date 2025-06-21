#pragma once
// code adapted from https://doi.org/10.1111/cgf.14176

#ifndef HEADLESS
#    include <saiga/opengl/vr/OpenvrWrapper.h>
#endif

#include <saiga/core/math/Types.h>
// #include "saiga/submodules/glog/config.h"


namespace EyeTracking
{
#ifndef HEADLESS

void initEyeTracking(std::shared_ptr<Saiga::OpenVRWrapper> vrWrapperPtr);
#endif
void update();
Saiga::vec2 getEyeLookPositionInPixels(int w, int h);
Saiga::vec2 getEyeLookPositionInPixelsForEye(int eye, int w, int h);

// glm::vec2 getEyeLookPositionInPixels_nonSDK();
float getFoveaAngleInPixels();
float getFoveaAngle();
void destroyTracker();
bool ETCalibFinished();
bool checkIfCalibFinished();
void calibrateET();
}  // namespace EyeTracking
