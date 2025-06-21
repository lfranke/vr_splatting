#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <thread>
#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Lip.h"
#include "SRanipal_Enums.h"
#include "SRanipal_NotRelease.h"
#pragma comment (lib, "SRanipal.lib")
using namespace ViveSR;

#define EnableEyeTracking 1
#define DisableEyeTracking 0

//#define UseEyeCallback
//#define UseEyeCallback_v2

bool GetBitMaskValidation(uint64_t mask, ViveSR::anipal::Eye::SingleEyeDataValidity SingleEyeDataType);
std::string CovertErrorCode(int error);

std::thread *t = nullptr;
bool EnableEye = false, EnableEyeV2 = false;
bool EnableLip = false, EnableLipV2 = false;
bool looping = false;
void streaming() {
    // Eye 
	ViveSR::anipal::Eye::EyeData eye_data;
    ViveSR::anipal::Eye::EyeData_v2 eye_data_v2;
    // Lip
	char lip_image[800 * 400];
	ViveSR::anipal::Lip::LipData lip_data;
    ViveSR::anipal::Lip::LipData_v2 lip_data_v2;
	lip_data.image = lip_image;
    lip_data_v2.image = lip_image;

	int result = ViveSR::Error::WORK;
	while (looping) {
#ifndef UseEyeCallback || UseEyeCallback_v2
		if (EnableEye) {
			int result = ViveSR::anipal::Eye::GetEyeData(&eye_data);
			if (result == ViveSR::Error::WORK) {
				float *gaze = eye_data.verbose_data.left.gaze_direction_normalized.elem_;
				printf("[Eye] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
				bool needCalibration = false;
				int error = ViveSR::anipal::Eye::IsUserNeedCalibration(&needCalibration);
				if (needCalibration) {
					printf("[Eye] Need to do calibration\n");
				}
				else {
					printf("[Eye] Don't need to do calibration\n");
				}
			}
		}
        if (EnableEyeV2) {
            int result = ViveSR::anipal::Eye::GetEyeData_v2(&eye_data_v2);
            if (result == ViveSR::Error::WORK) {
                float *gaze = eye_data_v2.verbose_data.left.gaze_direction_normalized.elem_;
                printf("[Eye v2] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
            }
        }
#endif
		if (EnableLip) {
			result = ViveSR::anipal::Lip::GetLipData(&lip_data);
            if (result == ViveSR::Error::WORK) {
                float *weightings = lip_data.prediction_data.blend_shape_weight;
                printf("Lip: \n");
                for (int i = 0; i < 27; i++) {
                    printf("%.2f ", weightings[i]);
                }
                printf("\n");
            }
		}
        if (EnableLipV2) {
            result = ViveSR::anipal::Lip::GetLipData_v2(&lip_data_v2);
            if (result == ViveSR::Error::WORK) {
                float *weightings = lip_data_v2.prediction_data.blend_shape_weight;
                printf("Lip Ver2: \n");
                for (int i = 0; i < ViveSR::anipal::Lip::blend_shape_nums; i++) {
                    printf("%.2f ", weightings[i]);
                }
                printf("\n");
            }
        }
	}
}

#ifdef UseEyeCallback
void TestEyeCallback(ViveSR::anipal::Eye::EyeData const &eye_data) {
    const float *gaze = eye_data.verbose_data.left.gaze_direction_normalized.elem_;
    printf("[Eye callback] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
    bool needCalibration = false;
    int error = ViveSR::anipal::Eye::IsUserNeedCalibration(&needCalibration);
    if (needCalibration) {
        printf("[Eye callback] Need to do calibration\n");
    }
    else {
        printf("[Eye callback] Don't need to do calibration\n");
    }
    float openness = eye_data.verbose_data.right.eye_openness;
    printf("[Eye callback] Openness %.2f\n", openness);
}
#endif
#ifdef UseEyeCallback_v2
void TestEyeCallback_v2(ViveSR::anipal::Eye::EyeData_v2 const &eye_data) {
    const float *gaze = eye_data.verbose_data.left.gaze_direction_normalized.elem_;
    printf("[Eye callback v2] Gaze: %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2]);
    bool needCalibration = false;
    int error = ViveSR::anipal::Eye::IsUserNeedCalibration(&needCalibration);
    if (needCalibration) {
        printf("[Eye callback v2] Need to do calibration\n");
    }
    else {
        printf("[Eye callback v2] Don't need to do calibration\n");
    }
    const float wide_left = eye_data.expression_data.left.eye_wide;
    const float wide_right = eye_data.expression_data.right.eye_wide;
    printf("[Eye callback v2] Wide: %.2f %.2f\n", wide_left, wide_right);
}
#endif

int main() {
	char *version = "";
	ViveSR::anipal::SRanipal_GetVersion(version);
	printf("SRanipal Sample (version: %s)\n\nPlease refer the below hotkey list to try APIs.\n", version);
	printf("[`] Exit this program.\n");
	printf("[0] Release all anipal engines.\n");
	printf("[1] Initial Eye engine\n");
	printf("[2] Initial Lip engine\n");
	printf("[3] Launch a thread to query data.\n");
	printf("[4] Stop the thread.\n");
    printf("[5] Initial Eye v2 engine\n");
    printf("[6] Initial Lip v2 engine\n");
	char str = 0;
	int error, id = NULL;   
	while (true) {
		if (str != '\n' && str != EOF) { printf("\nWait for key event :"); }
		str = getchar();
		if (str == '`') break;
		else if (str == '0') {
#ifdef UseEyeCallback
            ViveSR::anipal::Eye::UnregisterEyeDataCallback(TestEyeCallback);
#endif
#ifdef UseEyeCallback_v2
            ViveSR::anipal::Eye::UnregisterEyeDataCallback_v2(TestEyeCallback_v2);
#endif
			error = ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE);
            error = ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE_V2);
			error = ViveSR::anipal::Release(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP);
            error = ViveSR::anipal::Release(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP_V2);
			printf("Successfully release all anipal engines.\n");
			EnableEye = EnableLip = false;
		}
		else if (str == '1') {
			error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
			if (error == ViveSR::Error::WORK) {
                EnableEye = true; printf("Successfully initialize Eye engine.\n");
#ifdef UseEyeCallback
                ViveSR::anipal::Eye::RegisterEyeDataCallback(TestEyeCallback);
#endif
            }
            else if (error == ViveSR::Error::RUNTIME_NOT_FOUND) printf("please follows SRanipal SDK guide to install SR_Runtime first\n");
            else if (error == ViveSR::Error::NOT_SUPPORT_EYE_TRACKING) printf("This HMD do not have eye tracking feature!\n");
			else printf("Fail to initialize Eye engine. please refer the code %d %s.\n", error, CovertErrorCode(error).c_str());
		}
		else if (str == '2') {
			error = ViveSR::anipal::Initial(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP, NULL);
			if (error == ViveSR::Error::WORK) { EnableLip = true; printf("Successfully initialize Lip engine.\n"); }
			else printf("Fail to initialize Lip engine. please refer the code %d %s.\n", error, CovertErrorCode(error).c_str());
		}
		else if (str == '3') {
            if (t == nullptr) {
                t = new std::thread(streaming);
                if(t)   looping = true;
            }
		}
		else if (str == '4') {
            looping = false;
			if (t == nullptr) continue;
			t->join();
			delete t;
			t = nullptr;
		}
        else if (str == '5') {
            error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE_V2, NULL);
            if (error == ViveSR::Error::WORK) {
                EnableEyeV2 = true;
                printf("Successfully initialize version2 Eye engine.\n");
#ifdef UseEyeCallback_v2
                ViveSR::anipal::Eye::RegisterEyeDataCallback_v2(TestEyeCallback_v2);
#endif
            }
            else printf("Fail to initialize version2 Eye engine. please refer the code %d %s.\n", error, CovertErrorCode(error).c_str());
        }
        else if (str == '6') {
            error = ViveSR::anipal::Initial(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP_V2, NULL);
            if (error == ViveSR::Error::WORK) { 
                EnableLipV2 = true;
                printf("Successfully initialize version2 Lip engine.\n"); 
            }
            else printf("Fail to initialize version2 Lip engine. please refer the code %d %s.\n", error, CovertErrorCode(error).c_str());
        }
	}
	ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE);
	ViveSR::anipal::Release(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP);
    ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE_V2);
    ViveSR::anipal::Release(ViveSR::anipal::Lip::ANIPAL_TYPE_LIP_V2);
    return 0;
}

// For test
bool GetBitMaskValidation(uint64_t mask, ViveSR::anipal::Eye::SingleEyeDataValidity SingleEyeDataType) {
    return mask & (1 << (int)SingleEyeDataType) > 0;
}

std::string CovertErrorCode(int error) {
    std::string result = "";
    switch (error) {
    case(RUNTIME_NOT_FOUND):     result = "RUNTIME_NOT_FOUND"; break;
    case(NOT_INITIAL):           result = "NOT_INITIAL"; break;
    case(FAILED):                result = "FAILED"; break;
    case(WORK):                  result = "WORK"; break;
    case(INVALID_INPUT):         result = "INVALID_INPUT"; break;
    case(FILE_NOT_FOUND):        result = "FILE_NOT_FOUND"; break;
    case(DATA_NOT_FOUND):        result = "DATA_NOT_FOUND"; break;
    case(UNDEFINED):             result = "UNDEFINED"; break;
    case(INITIAL_FAILED):        result = "INITIAL_FAILED"; break;
    case(NOT_IMPLEMENTED):       result = "NOT_IMPLEMENTED"; break;
    case(NULL_POINTER):          result = "NULL_POINTER"; break;
    case(OVER_MAX_LENGTH):       result = "OVER_MAX_LENGTH"; break;
    case(FILE_INVALID):          result = "FILE_INVALID"; break;
    case(UNINSTALL_STEAM):       result = "UNINSTALL_STEAM"; break;
    case(MEMCPY_FAIL):           result = "MEMCPY_FAIL"; break;
    case(NOT_MATCH):             result = "NOT_MATCH"; break;
    case(NODE_NOT_EXIST):        result = "NODE_NOT_EXIST"; break;
    case(UNKONW_MODULE):         result = "UNKONW_MODULE"; break;
    case(MODULE_FULL):           result = "MODULE_FULL"; break;
    case(UNKNOW_TYPE):           result = "UNKNOW_TYPE"; break;
    case(INVALID_MODULE):        result = "INVALID_MODULE"; break;
    case(INVALID_TYPE):          result = "INVALID_TYPE"; break;
    case(MEMORY_NOT_ENOUGH):     result = "MEMORY_NOT_ENOUGH"; break;
    case(BUSY):                  result = "BUSY"; break;
    case(NOT_SUPPORTED):         result = "NOT_SUPPORTED"; break;
    case(INVALID_VALUE):         result = "INVALID_VALUE"; break;
    case(COMING_SOON):           result = "COMING_SOON"; break;
    case(INVALID_CHANGE):        result = "INVALID_CHANGE"; break;
    case(TIMEOUT):               result = "TIMEOUT"; break;
    case(DEVICE_NOT_FOUND):      result = "DEVICE_NOT_FOUND"; break;
    case(INVALID_DEVICE):        result = "INVALID_DEVICE"; break;
    case(NOT_AUTHORIZED):        result = "NOT_AUTHORIZED"; break;
    case(ALREADY):               result = "ALREADY"; break;
    case(INTERNAL):              result = "INTERNAL"; break;
    case(CONNECTION_FAILED):     result = "CONNECTION_FAILED"; break;
    case(ALLOCATION_FAILED):     result = "ALLOCATION_FAILED"; break;
    case(OPERATION_FAILED):      result = "OPERATION_FAILED"; break;
    case(NOT_AVAILABLE):         result = "NOT_AVAILABLE"; break;
    case(CALLBACK_IN_PROGRESS):  result = "CALLBACK_IN_PROGRESS"; break;
    case(SERVICE_NOT_FOUND):     result = "SERVICE_NOT_FOUND"; break;
    case(DISABLED_BY_USER):      result = "DISABLED_BY_USER"; break;
    case(EULA_NOT_ACCEPT):       result = "EULA_NOT_ACCEPT"; break;
    case(RUNTIME_NO_RESPONSE):   result = "RUNTIME_NO_RESPONSE"; break;
    case(OPENCL_NOT_SUPPORT):    result = "OPENCL_NOT_SUPPORT"; break;
    case(NOT_SUPPORT_EYE_TRACKING): result = "NOT_SUPPORT_EYE_TRACKING"; break;
    case(LIP_NOT_SUPPORT):       result = "LIP_NOT_SUPPORT"; break;
    default:
        result = "No such error code";
    }
    return result;
}