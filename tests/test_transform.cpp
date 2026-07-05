#include <cassert>
#include <vector>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    assert(transforms.size() == 1);
    assert(transforms.front() == "0.0.0");
    const std::vector<float> matrix{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    cco::TransformConfig config{"0.0.0", 4, 2};
    const auto transformed = cco::run_transform(matrix, {4, 4}, config);
    assert(transformed.reduced_data.size() == 4);
    assert(transformed.reduced_data[0] == 3.5f);
    assert(transformed.reduced_data[1] == 5.5f);
    assert(transformed.reduced_data[2] == 11.5f);
    assert(transformed.reduced_data[3] == 13.5f);
    return 0;
}
