#pragma once

#include <vector>

#include "cco/transform.h"

namespace cco {

std::vector<float> reference_reconstruct(const TransformResult& result);
double reference_error(const std::vector<float>& lhs, const std::vector<float>& rhs);

} // namespace cco
