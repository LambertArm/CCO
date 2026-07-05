#include <iostream>
#include <vector>

#include "cco/transform.h"

int main() {
    cco::TransformConfig config{"0.0.0", {1024, 128}, {256, 32}};
    std::vector<float> matrix(config.original_shape.rows * config.original_shape.cols, 1.0f);
    const auto result = cco::run_transform(matrix, config.original_shape, config);
    std::cout << "transform=" << result.version
              << " transformed=" << result.transformed_shape.rows
              << "x" << result.transformed_shape.cols << '\n';
    return 0;
}
