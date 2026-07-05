#include "cco/compute.h"

#include <stdexcept>

#include "cco/kernel_common.h"

namespace cco {

namespace {

void validate_gemm_shapes(MatrixShape a_shape, MatrixShape b_shape,
                          std::size_t a_size, std::size_t b_size) {
    if (a_shape.cols != b_shape.rows) {
        throw std::invalid_argument("run_gemm requires a_shape.cols == b_shape.rows");
    }
    if (a_size != a_shape.rows * a_shape.cols) {
        throw std::invalid_argument("run_gemm received mismatched lhs storage");
    }
    if (b_size != b_shape.rows * b_shape.cols) {
        throw std::invalid_argument("run_gemm received mismatched rhs storage");
    }
}

} // namespace

std::vector<float> make_zero_matrix(MatrixShape shape) {
    return std::vector<float>(shape.rows * shape.cols, 0.0f);
}

std::vector<float> run_gemm(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape a_shape,
                            MatrixShape b_shape) {
    validate_gemm_shapes(a_shape, b_shape, a.size(), b.size());

    const std::size_t m = a_shape.rows;
    const std::size_t n = b_shape.cols;
    const std::size_t k = a_shape.cols;
    std::vector<float> out(m * n, 0.0f);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t kk = 0; kk < k; ++kk) {
            const float lhs = a[row * k + kk];
            for (std::size_t col = 0; col < n; ++col) {
                out[row * n + col] += lhs * b[kk * n + col];
            }
        }
    }

    return out;
}

} // namespace cco
