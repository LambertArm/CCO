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
    if (!is_valid_transform_config(config)) {
        throw std::invalid_argument("invalid transform config for 0.0.0");
    }
    if (shape.rows != config.original_shape.rows || shape.cols != config.original_shape.cols) {
        throw std::invalid_argument("transform config original_shape must match the input matrix shape");
    }
    if (matrix.size() != shape.rows * shape.cols) {
        throw std::invalid_argument("transform received mismatched matrix storage");
    }
}

} // namespace

TransformResult run_transform(const TransformConfig& config) {
    std::vector<float> zero(config.original_shape.rows * config.original_shape.cols, 0.0f);
    return run_transform(zero, config.original_shape, config);
}

TransformResult run_transform(const std::vector<float>& matrix,
                              MatrixShape shape,
                              const TransformConfig& config) {
    validate_transform_inputs(matrix, shape, config);

    TransformResult result;
    result.version = kBaselineVersion;
    result.original_shape = shape;
    result.transformed_shape = config.transformed_shape;
    result.row_block_ratio = config.original_shape.rows / config.transformed_shape.rows;
    result.col_block_ratio = config.original_shape.cols / config.transformed_shape.cols;
    result.transformed_data.assign(config.transformed_shape.rows * config.transformed_shape.cols, 0.0f);

    const std::size_t transformed_rows = config.transformed_shape.rows;
    const std::size_t transformed_cols = config.transformed_shape.cols;
    const std::size_t row_block = result.row_block_ratio;
    const std::size_t col_block = result.col_block_ratio;

    for (std::size_t row_index = 0; row_index < transformed_rows; ++row_index) {
        for (std::size_t col_index = 0; col_index < transformed_cols; ++col_index) {
            double acc = 0.0;
            for (std::size_t row = 0; row < row_block; ++row) {
                for (std::size_t col = 0; col < col_block; ++col) {
                    const std::size_t src_row = row_index * row_block + row;
                    const std::size_t src_col = col_index * col_block + col;
                    acc += matrix[src_row * shape.cols + src_col];
                }
            }
            result.transformed_data[row_index * transformed_cols + col_index] =
                static_cast<float>(acc / static_cast<double>(row_block * col_block));
        }
    }

    return result;
}

std::vector<float> reconstruct_matrix(const TransformResult& result) {
    const std::size_t rows = result.original_shape.rows;
    const std::size_t cols = result.original_shape.cols;
    const std::size_t transformed_rows = result.transformed_shape.rows;
    const std::size_t transformed_cols = result.transformed_shape.cols;
    const std::size_t row_block =
        result.row_block_ratio == 0 ? (rows / transformed_rows) : result.row_block_ratio;
    const std::size_t col_block =
        result.col_block_ratio == 0 ? (cols / transformed_cols) : result.col_block_ratio;
    std::vector<float> out(rows * cols, 0.0f);

    for (std::size_t row_index = 0; row_index < transformed_rows; ++row_index) {
        for (std::size_t col_index = 0; col_index < transformed_cols; ++col_index) {
            const float value = result.transformed_data[row_index * transformed_cols + col_index];
            for (std::size_t row = 0; row < row_block; ++row) {
                for (std::size_t col = 0; col < col_block; ++col) {
                    const std::size_t dst_row = row_index * row_block + row;
                    const std::size_t dst_col = col_index * col_block + col;
                    out[dst_row * cols + dst_col] = value;
                }
            }
        }
    }

    return out;
}

} // namespace cco
