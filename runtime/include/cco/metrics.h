#pragma once

#include "cco/types.h"

namespace cco {

RunMetrics make_placeholder_metrics();
double estimate_attention_memory_mb(AttentionShape full_shape, AttentionShape transformed_shape);

} // namespace cco
