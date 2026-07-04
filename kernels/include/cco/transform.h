#pragma once

#include <string>
#include <vector>

#include "cco/types.h"

namespace cco {

struct TransformResult {
    std::string version;
    MatrixShape original_shape;
    MatrixShape reduced_shape;
    std::vector<float> reduced_data;
};

TransformResult run_transform(const TransformConfig& config);
std::vector<std::string> available_transforms();

} // namespace cco
