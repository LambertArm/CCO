#include <cassert>
#include <vector>

#include "cco/reference.h"
#include "cco/transform.h"

int main() {
    const std::vector<float> matrix{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    cco::TransformConfig config{"0.0.0", 4, 2};
    const auto transformed = cco::run_transform(matrix, {4, 4}, config);
    const auto reconstructed = cco::reference_reconstruct(transformed);
    assert(reconstructed.size() == 16);
    assert(reconstructed[0] == 3.5f);
    assert(reconstructed[1] == 3.5f);
    assert(reconstructed[4] == 3.5f);
    assert(reconstructed[15] == 13.5f);
    return 0;
}
