#include <cassert>
#include <vector>

#include "cco/pipeline.h"

int main() {
    cco::AttentionTransformConfig config{"0.0.0", {32, 16}, {0, 0}};
    const auto result = cco::run_pipeline(config);
    assert(result.q_transform.transformed_shape.rows == 8);
    assert(result.k_transform.transformed_shape.cols == 4);
    assert(result.metrics.memory_mb > 0.0);

    const std::vector<float> q{
        1, 0,
        0, 1,
        1, 1,
        0, 2,
    };
    const std::vector<float> k{
        1, 0,
        0, 1,
        2, 1,
        1, 1,
    };
    const std::vector<float> v{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    const auto rich = cco::run_pipeline(q, k, v, {4, 2}, {"0.0.0", {4, 2}, {2, 1}});
    assert(rich.exact_output.size() == 8);
    assert(rich.transformed_output.size() == 8);
    assert(rich.metrics.reconstruction_error >= 0.0);
    return 0;
}
