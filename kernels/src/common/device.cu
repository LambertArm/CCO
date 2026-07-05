#include "cco/kernel_common.h"

#include <string>

#include <cuda_runtime.h>

namespace cco {

bool cuda_runtime_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int cuda_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
}

std::string cuda_device_name(int device_index) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_index) != cudaSuccess) {
        return "unknown";
    }
    return std::string(prop.name);
}

} // namespace cco
