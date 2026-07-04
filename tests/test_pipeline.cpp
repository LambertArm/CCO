#include <cassert>

#include "cco/pipeline.h"

int main() {
    cco::TransformConfig config{"0.0.0", 32, 16};
    const auto result = cco::run_pipeline(config);
    assert(result.transformed.reduced_shape.rows == 16);
    return 0;
}
