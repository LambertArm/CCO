add_library(cco_kernels
    src/common/device.cu
    src/common/memory.cu
    src/common/timing.cu
    src/common/utils.cpp
    src/transform/00_baseline.cu
    src/transform/registry.cpp
    src/compute/gemm.cu
    src/compute/gemv.cu
)

target_include_directories(cco_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(cco_kernels PUBLIC cxx_std_17)
