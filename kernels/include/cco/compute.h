#pragma once

#include <vector>

#include "cco/types.h"

namespace cco {

std::vector<float> make_zero_matrix(MatrixShape shape);

std::vector<float> run_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape a_shape,
                            MatrixShape b_shape);

} // namespace cco
