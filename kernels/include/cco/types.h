#pragma once

#include <cstddef>
#include <string>

namespace cco {

struct MatrixShape {
    std::size_t rows{};
    std::size_t cols{};
};

struct TransformConfig {
    std::string name;
    std::size_t n{};
    std::size_t m{};
};

struct RunMetrics {
    double latency_ms{};
    double reconstruction_error{};
    double memory_mb{};
};

} // namespace cco
