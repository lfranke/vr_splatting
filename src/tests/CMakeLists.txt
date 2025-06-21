function(saiga_test FILE_NAME)
    #cmake_policy(SET CMP0104 OLD)
    string(REGEX REPLACE "\\.[^.]*$" "" NAME ${FILE_NAME})

    set(TARGET_NAME ${NAME})

    add_executable(${TARGET_NAME} ${FILE_NAME})
    message(STATUS "Test enabled:      ${NAME}")

    target_link_libraries(${TARGET_NAME} PUBLIC NeuralPoints gtest gtest_main)
    target_link_libraries(${TARGET_NAME} PUBLIC ${ARGN})
    target_link_libraries(${TARGET_NAME} PUBLIC ${LIBS})

    # target_compile_definitions(${TARGET_NAME} PUBLIC "-g")

    add_test(
            NAME ${TARGET_NAME}
            COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET_NAME}
    )

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER tests/${PREFIX})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests")
endfunction()

saiga_test(test_loss.cpp)
saiga_test(test_grad_oct.cpp)
saiga_test(test_random_sampling.cpp)
saiga_test(test_alpha_blend2.cpp)
saiga_test(test_alpha_list_sort.cu)
saiga_test(test_ocam.cpp)
#saiga_test(test_tcnn.cu)
saiga_test(test_renderer.cpp)
