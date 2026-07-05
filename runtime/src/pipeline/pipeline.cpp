#include "cco/pipeline.h"

#include <chrono>
#include <stdexcept>

#include "cco/kernel_common.h"
#include "cco/metrics.h"
#include "cco/reference.h"

namespace cco {

namespace {

AttentionTransformConfig resolve_attention_transform_config(const AttentionTransformConfig& config) {
    AttentionTransformConfig resolved = config;
    if (resolved.transformed_shape.seq_len == 0 || resolved.transformed_shape.head_dim == 0) {
        resolved.transformed_shape = default_transformed_attention_shape(resolved.name, resolved.original_shape);
    }
    return resolved;
}

TransformConfig make_matrix_transform_config(const AttentionTransformConfig& config) {
    return {config.name, to_matrix_shape(config.original_shape), to_matrix_shape(config.transformed_shape)};
}

void validate_attention_inputs(const std::vector<float>& q,
                               const std::vector<float>& k,
                               const std::vector<float>& v,
                               AttentionShape shape) {
    const std::size_t expected = shape.seq_len * shape.head_dim;
    if (q.size() != expected || k.size() != expected || v.size() != expected) {
        throw std::invalid_argument("attention pipeline received mismatched Q/K/V storage");
    }
}

} // namespace

PipelineResult run_pipeline(const AttentionTransformConfig& config) {
    const AttentionTransformConfig resolved = resolve_attention_transform_config(config);
    if (!is_valid_attention_transform_config(resolved)) {
        throw std::invalid_argument("invalid attention transform config");
    }

    const std::size_t count = resolved.original_shape.seq_len * resolved.original_shape.head_dim;
    std::vector<float> q(count, 0.0f);
    std::vector<float> k(count, 0.0f);
    std::vector<float> v(count, 0.0f);
    return run_pipeline(q, k, v, resolved.original_shape, resolved);
}

PipelineResult run_pipeline(const std::vector<float>& q,
                            const std::vector<float>& k,
                            const std::vector<float>& v,
                            AttentionShape shape,
                            const AttentionTransformConfig& config) {
    using clock = std::chrono::steady_clock;
    const AttentionTransformConfig resolved = resolve_attention_transform_config(config);

    if (shape.seq_len != resolved.original_shape.seq_len || shape.head_dim != resolved.original_shape.head_dim) {
        throw std::invalid_argument("attention pipeline shape must match config.original_shape");
    }
    if (!is_valid_attention_transform_config(resolved)) {
        throw std::invalid_argument("invalid attention transform config");
    }
    validate_attention_inputs(q, k, v, shape);

    const TransformConfig matrix_config = make_matrix_transform_config(resolved);
    const MatrixShape matrix_shape = to_matrix_shape(shape);

    PipelineResult result;
    result.q_transform = run_transform(q, matrix_shape, matrix_config);
    result.k_transform = run_transform(k, matrix_shape, matrix_config);
    result.v_transform = run_transform(v, matrix_shape, matrix_config);

    const auto exact_start = clock::now();
    result.exact_output = reference_attention(q, k, v, shape);
    const auto exact_end = clock::now();

    const auto transformed_start = clock::now();
    const AttentionShape transformed_attention_shape{
        resolved.transformed_shape.seq_len,
        resolved.transformed_shape.head_dim,
    };
    const auto transformed_space_output = reference_attention(
        result.q_transform.transformed_data,
        result.k_transform.transformed_data,
        result.v_transform.transformed_data,
        transformed_attention_shape);

    TransformResult output_transform{
        resolved.name,
        matrix_shape,
        to_matrix_shape(transformed_attention_shape),
        result.q_transform.row_block_ratio,
        result.q_transform.col_block_ratio,
        transformed_space_output,
    };
    result.transformed_output = reference_reconstruct(output_transform);
    const auto transformed_end = clock::now();

    result.metrics = {
        std::chrono::duration<double, std::milli>(exact_end - exact_start).count(),
        std::chrono::duration<double, std::milli>(transformed_end - transformed_start).count(),
        reference_error(result.exact_output, result.transformed_output),
        estimate_attention_memory_mb(shape, transformed_attention_shape),
    };

    return result;
}

} // namespace cco
