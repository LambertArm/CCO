#include "cco/kernel_common.h"

namespace cco {

MatrixShape to_matrix_shape(AttentionShape shape) {
    return {shape.seq_len, shape.head_dim};
}

bool is_square_matrix_shape(MatrixShape shape) {
    return shape.rows == shape.cols;
}

bool is_valid_transform_config(const TransformConfig& config) {
    return config.original_shape.rows > 0 &&
           config.original_shape.cols > 0 &&
           config.transformed_shape.rows > 0 &&
           config.transformed_shape.cols > 0 &&
           config.original_shape.rows >= config.transformed_shape.rows &&
           config.original_shape.cols >= config.transformed_shape.cols &&
           (config.original_shape.rows % config.transformed_shape.rows) == 0 &&
           (config.original_shape.cols % config.transformed_shape.cols) == 0;
}

bool is_valid_attention_transform_config(const AttentionTransformConfig& config) {
    return is_valid_transform_config(
        TransformConfig{config.name, to_matrix_shape(config.original_shape), to_matrix_shape(config.transformed_shape)});
}

} // namespace cco
