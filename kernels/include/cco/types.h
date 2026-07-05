#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cco {

inline constexpr std::size_t kDefaultAttentionSeqLen = 4096;
inline constexpr std::size_t kDefaultAttentionHeadDim = 256;
inline constexpr std::size_t kDefaultTransformedAttentionSeqDivisor = 4;
inline constexpr std::size_t kDefaultTransformedAttentionDimDivisor = 4;

struct MatrixShape {
    std::size_t rows{};
    std::size_t cols{};
};

struct DenseMatrix {
    MatrixShape shape{};
    std::vector<float> values;
};

struct AttentionShape {
    std::size_t seq_len{kDefaultAttentionSeqLen};
    std::size_t head_dim{kDefaultAttentionHeadDim};
};

struct TransformConfig {
    std::string name;
    MatrixShape original_shape{};
    MatrixShape transformed_shape{};
};

struct AttentionTransformConfig {
    std::string name;
    AttentionShape original_shape{};
    AttentionShape transformed_shape{0, 0};
};

struct RunMetrics {
    double exact_latency_ms{};
    double transformed_latency_ms{};
    double reconstruction_error{};
    double memory_mb{};
};

} // namespace cco
