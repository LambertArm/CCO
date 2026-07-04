#include <iostream>

#include "cco/transform.h"

int main() {
    cco::TransformConfig config{"0.0.0", 1024, 256};
    const auto result = cco::run_transform(config);
    std::cout << "transform=" << result.version
              << " reduced=" << result.reduced_shape.rows
              << "x" << result.reduced_shape.cols << '\n';
    return 0;
}
