#include "cco/pipeline.h"
#include "cco/compute.h"
#include "cco/metrics.h"

namespace cco {

PipelineResult run_pipeline(const TransformConfig& config) {
    PipelineResult result;
    result.transformed = run_transform(config);
    (void)run_gemm({}, {}, {0, 0}, {0, 0});
    result.metrics = make_placeholder_metrics();
    return result;
}

} // namespace cco
