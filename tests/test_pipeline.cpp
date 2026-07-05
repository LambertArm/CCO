#include <cassert>
#include <vector>

#include "cco/pipeline.h"

int main() {
    cco::TransformConfig config{"0.0.0", 32, 16};
    const auto result = cco::run_pipeline(config);
    assert(result.lhs_transform.reduced_shape.rows == 16);
    assert(result.rhs_transform.reduced_shape.rows == 16);
    assert(result.metrics.memory_mb > 0.0);

    const std::vector<float> matrix{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    const auto rich = cco::run_pipeline(matrix, matrix, {4, 4}, {"0.0.0", 4, 2});
    assert(rich.exact_output.size() == 16);
    assert(rich.transformed_output.size() == 16);
    assert(rich.metrics.reconstruction_error >= 0.0);
    return 0;
}
