#include "cco/kernel_common.h"

namespace cco {

std::size_t matrix_storage_bytes(MatrixShape shape) {
    return shape.rows * shape.cols * sizeof(float);
}

std::size_t attention_storage_bytes(AttentionShape shape) {
    return shape.seq_len * shape.head_dim * sizeof(float) * 3U;
}

} // namespace cco
