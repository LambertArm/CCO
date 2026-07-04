#pragma once

#include <vector>

#include "cco/types.h"

namespace cco {

std::vector<float> run_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape a_shape,
                            MatrixShape b_shape);

std::vector<float> run_gemv(const std::vector<float>& a,
                            const std::vector<float>& x,
                            MatrixShape a_shape);

} // namespace cco
