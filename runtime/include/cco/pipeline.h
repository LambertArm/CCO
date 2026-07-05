#pragma once

#include "cco/transform.h"

namespace cco {

struct PipelineResult {
    TransformResult lhs_transform;
    TransformResult rhs_transform;
    std::vector<float> exact_output;
    std::vector<float> transformed_output;
    RunMetrics metrics;
};

PipelineResult run_pipeline(const TransformConfig& config);
PipelineResult run_pipeline(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape shape,
                            const TransformConfig& config);

} // namespace cco
