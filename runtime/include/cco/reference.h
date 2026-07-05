#pragma once

#include <vector>

#include "cco/attention.h"
#include "cco/transform.h"

namespace cco {

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       AttentionShape shape);
std::vector<float> reference_reconstruct(const TransformResult& result);
std::vector<float> reference_exact_gemm(const std::vector<float>& a,
                                        const std::vector<float>& b,
                                        MatrixShape a_shape,
                                        MatrixShape b_shape);
double reference_error(const std::vector<float>& lhs, const std::vector<float>& rhs);

} // namespace cco
