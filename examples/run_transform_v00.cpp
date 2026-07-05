#include <iostream>
#include <vector>

#include "cco/transform.h"

int main() {
    cco::TransformConfig config{"0.0.0", 1024, 256};
    std::vector<float> matrix(config.n * config.n, 1.0f);
    const auto result = cco::run_transform(matrix, {config.n, config.n}, config);
    std::cout << "transform=" << result.version
              << " reduced=" << result.reduced_shape.rows
              << "x" << result.reduced_shape.cols << '\n';
    return 0;
}
