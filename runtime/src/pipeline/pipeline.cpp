#include "cco/pipeline.h"

#include <chrono>

#include "cco/compute.h"
#include "cco/metrics.h"
#include "cco/reference.h"

namespace cco {

PipelineResult run_pipeline(const TransformConfig& config) {
    MatrixShape shape{config.n, config.n};
    auto a = make_zero_matrix(shape);
    auto b = make_zero_matrix(shape);
    return run_pipeline(a, b, shape, config);
}

PipelineResult run_pipeline(const std::vector<float>& a,
                            const std::vector<float>& b,
                            MatrixShape shape,
                            const TransformConfig& config) {
    using clock = std::chrono::steady_clock;

    PipelineResult result;
    result.lhs_transform = run_transform(a, shape, config);
    result.rhs_transform = run_transform(b, shape, config);

    const auto exact_start = clock::now();
    result.exact_output = reference_exact_gemm(a, b, shape, shape);
    const auto exact_end = clock::now();

    const auto transformed_start = clock::now();
    const auto reduced_output = run_gemm(
        result.lhs_transform.reduced_data,
        result.rhs_transform.reduced_data,
        result.lhs_transform.reduced_shape,
        result.rhs_transform.reduced_shape);

    TransformResult output_transform{
        config.name,
        shape,
        {config.m, config.m},
        config.n / config.m,
        reduced_output,
    };
    result.transformed_output = reference_reconstruct(output_transform);
    const auto transformed_end = clock::now();

    result.metrics = {
        std::chrono::duration<double, std::milli>(exact_end - exact_start).count(),
        std::chrono::duration<double, std::milli>(transformed_end - transformed_start).count(),
        reference_error(result.exact_output, result.transformed_output),
        estimate_memory_mb(shape, {config.m, config.m}),
    };

    return result;
}

} // namespace cco
