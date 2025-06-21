function(add_app TARGET_NAME ADD_DLL_POSTBUILD)

    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_include_directories(${TARGET_NAME} PUBLIC ".")
    target_link_libraries(${TARGET_NAME} NeuralPoints)

    if (PR_NO_WINDOW)
        target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
    endif ()
    add_definitions(-DEXECUTABLE_DIR="${PROJECT_SOURCE_DIR}")

    # ----------------------------------------------------------
    # dll copy
    if (WIN32)
        if (ADD_DLL_POSTBUILD)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/src/lib/$<CONFIG>/NeuralPoints.dll" $<TARGET_FILE_DIR:${TARGET_NAME}>)

            message(">>")
            message($<CONFIG>)
            set(copyDest "${PROJECT_BINARY_DIR}/bin/$<CONFIG>")

            function(SANITIZE_DLL_PATHS_AND_COPY PATHS_DLLS PATHS_OUTPUT)
                #Sanitizes paths to remove backslashes
                STRING(REGEX REPLACE "/" "\\\\" copyInput \"${PATHS_DLLS}\")
                STRING(REGEX REPLACE "/" "\\\\" copyDestination \"${PATHS_OUTPUT}\")
                message("dll post build copy: copying ${copyInput} to ${copyDestination}.")

                add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND COMMAND copy ${copyInput} ${copyDestination})
            endfunction()


            function(SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER PATHS_DLLS PATHS_OUTPUT)
                #Sanitizes paths to remove backslashes
                STRING(REGEX REPLACE "/" "\\\\" copyDestination \"${PATHS_OUTPUT}\")

                FILE(GLOB_RECURSE DLLS_FROM_FOLDER ${PATHS_DLLS}*.dll)
                message("--dll post build copy: copying ${PATHS_DLLS} to ${copyDestination}. ${DLLS_FROM_FOLDER}")

                foreach (file_i ${DLLS_FROM_FOLDER})
                    STRING(REGEX REPLACE "/" "\\\\" copyInput \"${file_i}\")

                    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${copyInput} ${copyDestination})
                    message("dll post build copy: copying (ifdifferent) ${copyInput} to ${copyDestination}.")

                endforeach (file_i)

                #add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND COMMAND copy ${copyInput} ${copyDestination})
            endfunction()


            # SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/../External/libtorch/lib/*.dll" ${copyDest})
            SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/../External/libtorch/lib/" ${copyDest})

            #SANITIZE_DLL_PATHS_AND_COPY("${CONDA_P_PATH}/Library/bin/libiomp5md.dll" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/cuda/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/cuda/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/opengl/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/opengl/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/core/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/src/saiga/core/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/submodules/assimp/bin/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/submodules/assimp/bin/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/submodules/glfw/src/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/submodules/glfw/src/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/submodules/glog/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/submodules/glog/$<CONFIG>/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/External/saiga/submodules/glfw/src/$<CONFIG>/*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/External/saiga/submodules/glfw/src/$<CONFIG>/" ${copyDest})

            #SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/../External/openvr/bin/win64/*.dll" ${copyDest})
            SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/../External/openvr/bin/win64/" ${copyDest})

            #SANITIZE_DLL_PATHS_AND_COPY("${PROJECT_BINARY_DIR}/../External/SRanipal/bin/*.dll" ${copyDest})
            SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${PROJECT_BINARY_DIR}/../External/SRanipal/bin/" ${copyDest})

            #SANITIZE_DLL_PATHS_AND_COPY("${CUDA_P_PATH}/bin/*.dll" ${copyDest})
            SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${CUDA_P_PATH}/bin/" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${CONDA_P_PATH}/Library/bin/mkl*.dll" ${copyDest})
            #SANITIZE_DLL_PATHS_AND_COPY_FROM_FOLDER("${CONDA_P_PATH}/Library/bin/mkl" ${copyDest})

            SANITIZE_DLL_PATHS_AND_COPY("${CONDA_P_PATH}/Library/bin/mkl_vml_avx2.2.dll" "${copyDest}/mkl_vml_avx2.1.dll")
            SANITIZE_DLL_PATHS_AND_COPY("${CONDA_P_PATH}/Library/bin/mkl_vml_def.2.dll" "${copyDest}/mkl_vml_def.1.dll")

        endif ()
    endif ()

    message(STATUS "App enabled:       ${TARGET_NAME}")
endfunction()

add_executable(colmap2adop colmap2adop.cpp)
add_app(colmap2adop FALSE)

add_executable(duster_colmap2adop duster_colmap2adop.cpp)
add_app(duster_colmap2adop FALSE)

add_executable(adop2colmap adop2colmap.cpp)
add_app(adop2colmap FALSE)

if (USE_VET)
    add_executable(hdr_to_vox hdr_to_vox.cpp)
    add_app(hdr_to_vox FALSE)
endif ()

add_executable(scannet2adop scannet2adop.cpp)
add_app(scannet2adop FALSE)

add_executable(preprocess_pointcloud preprocess_pointcloud.cpp)
add_app(preprocess_pointcloud FALSE)

add_executable(evaluate evaluate.cpp)
add_app(evaluate FALSE)

#add_executable(renderImages renderImages.cpp)
#add_app(renderImages FALSE)

add_executable(distort_images distort_images.cpp)
add_app(distort_images FALSE)

if (NOT HEADLESS)
    add_executable(preprocess preprocess.cpp)
    add_app(preprocess FALSE)
endif ()

set(TRAIN_SOURCES "train.cpp" "TrainScene.h" "TrainScene.cpp")
if (USE_VET)
    APPEND(TRAIN_SOURCES "../../External/NeAT/src/utils/cimg_wrapper.cpp")
endif ()

add_executable(train ${TRAIN_SOURCES})
if (USE_VET)
    if (UNIX)
        add_dependencies(train NeAT)
    endif ()
endif ()
add_app(train TRUE)

if (TARGET saiga_opengl)
    add_executable(viewer viewer.cpp viewer.h)
    add_app(viewer TRUE)


    # find_package(OpenVR QUIET)
    if (${OPENVR_FOUND})
        add_executable(adop_vr_viewer adop_vr_viewer.cpp UserStudy.cpp)
        add_app(adop_vr_viewer TRUE)

    endif ()
endif ()
