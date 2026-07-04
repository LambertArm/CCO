#pragma once

#include "cco/transform.h"

namespace cco {

struct PipelineResult {
    TransformResult transformed;
    RunMetrics metrics;
};

PipelineResult run_pipeline(const TransformConfig& config);

} // namespace cco
