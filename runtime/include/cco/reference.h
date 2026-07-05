#pragma once

#include <vector>

#include "cco/transform.h"
#include "cco/attention.h"

namespace cco {

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       AttentionShape shape);
std::vector<float> reference_reconstruct(const TransformResult& result);
double reference_error(const std::vector<float>& lhs, const std::vector<float>& rhs);

} // namespace cco
