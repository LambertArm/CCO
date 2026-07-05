#include "cco/compute.h"
#include "cco/reference.h"

namespace cco {

std::vector<float> reference_exact_gemm(const std::vector<float>& a,
                                        const std::vector<float>& b,
                                        MatrixShape a_shape,
                                        MatrixShape b_shape) {
    return run_gemm(a, b, a_shape, b_shape);
}

} // namespace cco
