#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cco {

struct MatrixShape {
    std::size_t rows{};
    std::size_t cols{};
};

struct DenseMatrix {
    MatrixShape shape{};
    std::vector<float> values;
};

struct TransformConfig {
    std::string name;
    std::size_t n{};
    std::size_t m{};
};

struct RunMetrics {
    double exact_latency_ms{};
    double transformed_latency_ms{};
    double reconstruction_error{};
    double memory_mb{};
};

struct AttentionShape {
    std::size_t seq_len{};
    std::size_t head_dim{};
};

} // namespace cco
