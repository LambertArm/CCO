#include <iostream>
#include <vector>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    std::cout << "CCO transform bench\n";
    std::cout << "registered transforms: " << transforms.size() << '\n';
    if (!transforms.empty()) {
        std::cout << "active transform version: " << transforms.front() << '\n';
    }
    const std::vector<float> matrix(128 * 64, 1.0f);
    cco::TransformConfig config{"0.0.0", {128, 64}, {32, 16}};
    const auto transformed = cco::run_transform(matrix, {128, 64}, config);
    std::cout << "transformed shape: " << transformed.transformed_shape.rows
              << "x" << transformed.transformed_shape.cols << '\n';
    return 0;
}
