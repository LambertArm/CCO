#include <iostream>
#include <vector>

#include "cco/pipeline.h"

int main() {
    std::cout << "CCO attention bench\n";
    const cco::AttentionShape shape{};
    const std::vector<float> q(shape.seq_len * shape.head_dim, 0.25f);
    const std::vector<float> k(shape.seq_len * shape.head_dim, 0.50f);
    const std::vector<float> v(shape.seq_len * shape.head_dim, 1.00f);
    const auto result = cco::run_pipeline(q, k, v, shape, {"0.0.0"});
    std::cout << "exact latency (ms): " << result.metrics.exact_latency_ms << '\n';
    std::cout << "transformed latency (ms): " << result.metrics.transformed_latency_ms << '\n';
    std::cout << "reconstruction error: " << result.metrics.reconstruction_error << '\n';
    return 0;
}
