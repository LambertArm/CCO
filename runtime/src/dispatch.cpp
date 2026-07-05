#include "cco/metrics.h"
#include "cco/kernel_common.h"

namespace cco {

RunMetrics make_placeholder_metrics() {
    return RunMetrics{0.0, 0.0, 0.0, 0.0};
}

double estimate_attention_memory_mb(AttentionShape full_shape, AttentionShape transformed_shape) {
    const double full_bytes = static_cast<double>(attention_storage_bytes(full_shape)) +
                              static_cast<double>(full_shape.seq_len * full_shape.seq_len * sizeof(float)) +
                              static_cast<double>(full_shape.seq_len * full_shape.head_dim * sizeof(float));
    const double transformed_bytes =
        static_cast<double>(attention_storage_bytes(transformed_shape)) +
        static_cast<double>(transformed_shape.seq_len * transformed_shape.seq_len * sizeof(float)) +
        static_cast<double>(transformed_shape.seq_len * transformed_shape.head_dim * sizeof(float));
    return (full_bytes + transformed_bytes) / (1024.0 * 1024.0);
}

} // namespace cco
