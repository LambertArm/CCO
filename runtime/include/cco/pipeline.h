#pragma once

#include "cco/transform.h"

namespace cco {

struct PipelineResult {
    TransformResult q_transform;
    TransformResult k_transform;
    TransformResult v_transform;
    std::vector<float> exact_output;
    std::vector<float> transformed_output;
    RunMetrics metrics;
};

PipelineResult run_pipeline(const AttentionTransformConfig& config);
PipelineResult run_pipeline(const std::vector<float>& q,
                            const std::vector<float>& k,
                            const std::vector<float>& v,
                            AttentionShape shape,
                            const AttentionTransformConfig& config);

} // namespace cco
