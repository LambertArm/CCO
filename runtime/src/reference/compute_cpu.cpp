#include "cco/compute.h"

namespace cco {

std::vector<float> make_zero_matrix(MatrixShape shape) {
    return std::vector<float>(shape.rows * shape.cols, 0.0f);
}

std::vector<float> run_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape a_shape,
                            MatrixShape b_shape) {
    const std::size_t m = a_shape.rows;
    const std::size_t n = b_shape.cols;
    const std::size_t k = a_shape.cols;
    std::vector<float> out(m * n, 0.0f);
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    return out;
}

std::vector<float> reference_exact_gemm(const std::vector<float>& a,
                                        const std::vector<float>& b,
                                        MatrixShape a_shape,
                                        MatrixShape b_shape) {
    return run_gemm(a, b, a_shape, b_shape);
}

} // namespace cco
