#include <cassert>
#include <vector>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    assert(transforms.size() == 1);
    assert(transforms.front() == "0.0.0");
    const std::vector<float> matrix{
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
    };
    cco::TransformConfig config{"0.0.0", {4, 6}, {2, 3}};
    const auto transformed = cco::run_transform(matrix, {4, 6}, config);
    assert(transformed.transformed_data.size() == 6);
    assert(transformed.row_block_ratio == 2);
    assert(transformed.col_block_ratio == 2);
    assert(transformed.transformed_data[0] == 4.5f);
    assert(transformed.transformed_data[1] == 6.5f);
    assert(transformed.transformed_data[2] == 8.5f);
    assert(transformed.transformed_data[3] == 16.5f);
    assert(transformed.transformed_data[4] == 18.5f);
    assert(transformed.transformed_data[5] == 20.5f);
    return 0;
}
