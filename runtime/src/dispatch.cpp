#include "cco/metrics.h"

namespace cco {

RunMetrics make_placeholder_metrics() {
    return RunMetrics{0.0, 0.0, 0.0};
}

double estimate_memory_mb(MatrixShape full_shape, MatrixShape reduced_shape) {
    const double full_bytes =
        static_cast<double>(full_shape.rows * full_shape.cols * sizeof(float)) * 3.0;
    const double reduced_bytes =
        static_cast<double>(reduced_shape.rows * reduced_shape.cols * sizeof(float)) * 3.0;
    return (full_bytes + reduced_bytes) / (1024.0 * 1024.0);
}

} // namespace cco
