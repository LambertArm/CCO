#include "cco/transform.h"

#include <stdexcept>
#include <vector>

#include "cco/kernel_common.h"

namespace cco {

namespace {

constexpr const char* kBaselineVersion = "0.0.0";

void validate_transform_inputs(const std::vector<float>& matrix,
                               MatrixShape shape,
                               const TransformConfig& config) {
    if (!is_square_matrix_shape(shape)) {
        throw std::invalid_argument("0.0.0 transform expects a square matrix");
    }
    if (!is_valid_transform_config(config)) {
        throw std::invalid_argument("invalid transform config for 0.0.0");
    }
    if (shape.rows != config.n || shape.cols != config.n) {
        throw std::invalid_argument("transform config n must match the input matrix shape");
    }
    if (matrix.size() != shape.rows * shape.cols) {
        throw std::invalid_argument("transform received mismatched matrix storage");
    }
}

} // namespace

TransformResult run_transform(const TransformConfig& config) {
    std::vector<float> zero(config.n * config.n, 0.0f);
    return run_transform(zero, {config.n, config.n}, config);
}

TransformResult run_transform(const std::vector<float>& matrix,
                              MatrixShape shape,
                              const TransformConfig& config) {
    validate_transform_inputs(matrix, shape, config);

    TransformResult result;
    result.version = kBaselineVersion;
    result.original_shape = shape;
    result.reduced_shape = {config.m, config.m};
    result.block_ratio = config.n / config.m;
    result.reduced_data.assign(config.m * config.m, 0.0f);

    const std::size_t n = shape.rows;
    const std::size_t m = config.m;
    const std::size_t block = result.block_ratio;

    for (std::size_t row_block = 0; row_block < m; ++row_block) {
        for (std::size_t col_block = 0; col_block < m; ++col_block) {
            double acc = 0.0;
            for (std::size_t row = 0; row < block; ++row) {
                for (std::size_t col = 0; col < block; ++col) {
                    const std::size_t src_row = row_block * block + row;
                    const std::size_t src_col = col_block * block + col;
                    acc += matrix[src_row * n + src_col];
                }
            }
            result.reduced_data[row_block * m + col_block] =
                static_cast<float>(acc / static_cast<double>(block * block));
        }
    }

    return result;
}

std::vector<float> reconstruct_matrix(const TransformResult& result) {
    const std::size_t n = result.original_shape.rows;
    const std::size_t m = result.reduced_shape.rows;
    const std::size_t block = result.block_ratio == 0 ? (n / m) : result.block_ratio;
    std::vector<float> out(n * n, 0.0f);

    for (std::size_t row_block = 0; row_block < m; ++row_block) {
        for (std::size_t col_block = 0; col_block < m; ++col_block) {
            const float value = result.reduced_data[row_block * m + col_block];
            for (std::size_t row = 0; row < block; ++row) {
                for (std::size_t col = 0; col < block; ++col) {
                    const std::size_t dst_row = row_block * block + row;
                    const std::size_t dst_col = col_block * block + col;
                    out[dst_row * n + dst_col] = value;
                }
            }
        }
    }

    return out;
}

} // namespace cco
