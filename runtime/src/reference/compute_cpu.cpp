#include "cco/compute.h"

namespace cco {

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

std::vector<float> run_gemv(const std::vector<float>& a,
                            const std::vector<float>& x,
                            MatrixShape a_shape) {
    std::vector<float> out(a_shape.rows, 0.0f);
    for (std::size_t row = 0; row < a_shape.rows; ++row) {
        float acc = 0.0f;
        for (std::size_t col = 0; col < a_shape.cols; ++col) {
            acc += a[row * a_shape.cols + col] * x[col];
        }
        out[row] = acc;
    }
    return out;
}

} // namespace cco
