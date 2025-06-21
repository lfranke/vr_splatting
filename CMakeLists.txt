cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
#-DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"
#-DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
#-DCMAKE_PREFIX_PATH="C:\Users\Linus\anaconda3\envs\trips\Library;.\External\libtorch"
#-DCONDA_P_PATH="C:\Users\Linus\anaconda3\envs\trips"
#-DCUDA_P_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\"
#list(APPEND CMAKE_PREFIX_PATH "C:/Users/Linus/anaconda3/envs/trips/Library")
#list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/External/libtorch")
set(CMAKE_PREFIX_PATH "C:/Users/Linus/anaconda3/envs/trips/Library;./External/libtorch")
set(CONDA_P_PATH "C:/Users/Linus/anaconda3/envs/trips")
set(CUDA_P_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/")
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/")
#set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
#set(CMAKE_CUDA_FLAGS  "--allow-unsupported-compiler")
#set(CMAKE_GENERATOR_TOOLSET cuda=11.8)
#set(CMAKE_C_FLAGS "_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

##-DCMAKE_GENERATOR_TOOLSET="cuda=11.8"
#add_definitions(/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
project(TRIPS VERSION 1.0.0 LANGUAGES C CXX CUDA)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/saiga/cmake/")

include(helper_macros)
include(ExternalProject)
DefaultBuildType(RelWithDebInfo)
message("Build Options")

OptionsHelper(HEADLESS "Skips the viewer and other apps that require a window." OFF)
OptionsHelper(ASAN "Adds sanitize=address compiler flag" OFF)
OptionsHelper(TINY_EIGEN "Use saiga's tiny eigen library." ON)
OptionsHelper(LIVE_TRAIN_VIEWER "Adds a GL window to training, shows live training progress (only in non-headless mode)." OFF)
OptionsHelper(USE_VET "Adds VET: Visual Error Tomography functionality." OFF)


if (HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)
else ()
endif ()

if (USE_VET)
    add_compile_definitions(COMPILE_WITH_VET)
endif ()

############# Required LIBRARIES ###############

if (UNIX)
    add_subdirectory(External/tensorboard_logger)
    set_property(TARGET tensorboard_logger PROPERTY POSITION_INDEPENDENT_CODE ON)
    PackageHelperTarget(tensorboard_logger TBL_FOUND)
    add_compile_definitions(TBLOGGER)
endif ()

#add_subdirectory(External/openvr)

#set(OPENVR_HEADERS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/openvr/headers/")
if (NOT HEADLESS)

    set(OPENVR_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/openvr/")
    #message(${OPENVR_HEADERS_ROOT_DIR})
    find_package(OpenVR QUIET)
    message("${OPENVR_INCLUDE_DIRS}")
    PackageHelper(OpenVR ${OPENVR_FOUND} "${OPENVR_INCLUDE_DIRS}" "${OPENVR_LIBRARY}")

    message(">>>>>>>>>>>>>>>>>>")
    message(${OPENVR_FOUND} "${OPENVR_INCLUDE_DIRS}" "${OPENVR_LIBRARY}")
    #find_package(OpenVR REQUIRED)
endif ()
# Saiga
set(SAIGA_BUILD_SAMPLES OFF)
set(SAIGA_BUILD_TESTS OFF)
set(SAIGA_MODULE_VULKAN OFF)
set(SAIGA_MODULE_VISION OFF)
if (HEADLESS)
    set(SAIGA_MODULE_OPENGL OFF)
    set(SAIGA_BUILD_GLFW OFF)
    message("HEADLESS MODE")
endif ()
set(SAIGA_NO_INSTALL ON)
set(SAIGA_USE_SUBMODULES ON)
set(SAIGA_WITH_FFMPEG OFF)
set(SAIGA_WITH_FREETYPE OFF)
set(SAIGA_WITH_YAMLCPP OFF)
set(SAIGA_WITH_OPENAL OFF)
set(SAIGA_WITH_MKL ON)
set(SAIGA_WITH_OPENMESH OFF)

set(SAIGA_WITH_OPENVR ON)

set(SAIGA_WITH_G2O OFF)
set(SAIGA_WITH_CHOLMOD OFF)
set(SAIGA_WITH_OPENNI OFF)
set(SAIGA_WITH_K4A OFF)
set(SAIGA_WITH_TINY_EIGEN ON)

if (MSVC)
    #multiprocessor compilation for visual studio
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif ()

add_subdirectory(External/saiga)

PackageHelperTarget(saiga_core SAIGA_FOUND)
if (NOT HEADLESS)
    PackageHelperTarget(saiga_opengl SAIGA_FOUND)
endif ()
PackageHelperTarget(saiga_cuda SAIGA_FOUND)

set(CAFFE2_USE_CUDNN ON)
# Torch
find_package(Torch REQUIRED)
PackageHelperTarget(torch TORCH_FOUND)
message("CMAKE_EXE_LINKER_FLAGS before: " ${CMAKE_EXE_LINKER_FLAGS})
message("TORCH_LIBRARIES: " "${TORCH_LIBRARIES}")
message("TORCH_CXX_FLAGS: " "${TORCH_CXX_FLAGS}")

#replace flags
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")
message("CMAKE_EXE_LINKER_FLAGS after: " ${CMAKE_EXE_LINKER_FLAGS})

##this is broken!!!!
#set(CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")


message("CUDNN_LIBRARY_PATH: " "${CUDNN_LIBRARY_PATH}" "; CUDNN_INCLUDE_PATH: " "${CUDNN_INCLUDE_PATH}")

if (USE_TINYCUDANN)
    add_subdirectory(External/tiny-cuda-nn)
    PackageHelperTarget(tiny-cuda-nn tiny-cuda-nn_FOUND)
endif ()

#add_subdirectory(External)
#PackageHelperTarget(simple-knn simple-knn_FOUND)

add_subdirectory(External/json)
PackageHelperTarget(nlohmann_json nlohmann_json_FOUND)

if (WIN32)
    # set(CMAKE_FIND_DEBUG_MODE TRUE)

    set(SRanipal_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/SRanipal")

    find_package(SRanipal REQUIRED)
    PackageHelper(SRanipal SRanipal_FOUND ${SRanipal_INCLUDE_DIR} ${SRanipal_LIBRARY})
    # set(CMAKE_FIND_DEBUG_MODE FALSE)

    #  message(${SRanipal_INCLUDE_DIR})
    #  message(${SRanipal_LIBRARY})

    #set(SRANI_DIR sranipal)
    #find_library(
    #        SRANI_LIBRARIES
    #        NAMES SRAnipal
    #        PATHS "${SRANI_DIR}/lib"
    #        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    #include_directories("${SRANI_DIR}/include")
endif ()
#openmp
#find_package(OpenMP REQUIRED)
#PackageHelperTarget(OpenMP::OpenMP_CXX OPENMP_FOUND)

include_directories(.)
include_directories(${PACKAGE_INCLUDES})

############# COMPILER FLAGS ###############
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")

if (MSVC)
    #multiprocessor compilation for visual studio
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
endif ()


message(STATUS CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message(STATUS CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})

#did torch forget to also export cupti?
set(LIBS ${LIBS} ${LIB_TARGETS} "${TORCH_LIBRARIES}" ${CUDA_cupti_LIBRARY} ${CUDNN_LIBRARY_PATH})

############# C++ Standard and Filesystem stuff ###############

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    #dll has all symbols, to avoid __declspecs everywhere
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    include(GenerateExportHeader)
endif ()

if (HEADLESS)
else ()
    if (LIVE_TRAIN_VIEWER)
    endif ()
    add_compile_definitions(LIVE_VIEWER)

endif ()

############# SOURCE ###############


add_subdirectory(src)

if (USE_VET)
    if (CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
        set(NEAT_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
        string(REPLACE ";" "|" CMAKE_PREFIX_PATH_ALT_SEP "${NEAT_CMAKE_PREFIX_PATH}")
        if (CMAKE_GENERATOR MATCHES "Ninja")
            ExternalProject_Add(NeAT
                    SOURCE_DIR ../External/NeAT
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/NeAT
                    LIST_SEPARATOR |
                    CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_ALT_SEP}
                    INSTALL_COMMAND echo "Skipping install step.")
        else ()
            ExternalProject_Add(NeAT
                    SOURCE_DIR ../External/NeAT
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/NeAT
                    LIST_SEPARATOR |
                    CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH_ALT_SEP}
                    BUILD_COMMAND make -j20
                    INSTALL_COMMAND echo "Skipping install step.")
        endif ()
    endif ()
endif ()