#pragma once

#include "cco/types.h"

namespace cco {

RunMetrics make_placeholder_metrics();
double estimate_memory_mb(MatrixShape full_shape, MatrixShape reduced_shape);

} // namespace cco
