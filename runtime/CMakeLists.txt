add_library(cco_runtime
    src/reference/transform_cpu.cpp
    src/reference/compute_cpu.cpp
    src/reference/reconstruct_cpu.cpp
    src/pipeline/pipeline.cpp
    src/dispatch.cpp
)

target_include_directories(cco_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernels/include
)

target_link_libraries(cco_runtime PUBLIC cco_kernels)
target_compile_features(cco_runtime PUBLIC cxx_std_17)
