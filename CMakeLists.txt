cmake_minimum_required(VERSION 3.24)

project(CudaComputeOSS VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CCO_BUILD_TESTS "Build CCO tests" ON)
option(CCO_BUILD_EXAMPLES "Build CCO examples" ON)

include(CTest)

add_subdirectory(kernels)
add_subdirectory(runtime)

add_executable(cco_bench_transform bench/bench_transform.cpp)
target_link_libraries(cco_bench_transform PRIVATE cco_runtime)

add_executable(cco_bench_compute bench/bench_compute.cpp)
target_link_libraries(cco_bench_compute PRIVATE cco_runtime)

add_executable(cco_bench_attention bench/bench_attention.cpp)
target_link_libraries(cco_bench_attention PRIVATE cco_runtime)

add_executable(cco_bench_pipeline bench/bench_pipeline.cpp)
target_link_libraries(cco_bench_pipeline PRIVATE cco_runtime)

if(CCO_BUILD_EXAMPLES)
    add_executable(cco_run_transform_v00 examples/run_transform_v00.cpp)
    target_link_libraries(cco_run_transform_v00 PRIVATE cco_runtime)

    add_executable(cco_compare_versions examples/compare_versions.cpp)
    target_link_libraries(cco_compare_versions PRIVATE cco_runtime)
endif()

if(CCO_BUILD_TESTS)
    add_executable(cco_test_transform tests/test_transform.cpp)
    target_link_libraries(cco_test_transform PRIVATE cco_runtime)
    add_test(NAME cco_test_transform COMMAND cco_test_transform)

    add_executable(cco_test_reconstruct tests/test_reconstruct.cpp)
    target_link_libraries(cco_test_reconstruct PRIVATE cco_runtime)
    add_test(NAME cco_test_reconstruct COMMAND cco_test_reconstruct)

    add_executable(cco_test_compute tests/test_compute.cpp)
    target_link_libraries(cco_test_compute PRIVATE cco_runtime)
    add_test(NAME cco_test_compute COMMAND cco_test_compute)

    add_executable(cco_test_attention tests/test_attention.cpp)
    target_link_libraries(cco_test_attention PRIVATE cco_runtime)
    add_test(NAME cco_test_attention COMMAND cco_test_attention)

    add_executable(cco_test_pipeline tests/test_pipeline.cpp)
    target_link_libraries(cco_test_pipeline PRIVATE cco_runtime)
    add_test(NAME cco_test_pipeline COMMAND cco_test_pipeline)
endif()
