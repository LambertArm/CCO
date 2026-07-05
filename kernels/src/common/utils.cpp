#include "cco/kernel_common.h"

namespace cco {

bool is_square_matrix_shape(MatrixShape shape) {
    return shape.rows == shape.cols;
}

bool is_valid_transform_config(const TransformConfig& config) {
    return config.n > 0 && config.m > 0 && config.n >= config.m && (config.n % config.m) == 0;
}

} // namespace cco
