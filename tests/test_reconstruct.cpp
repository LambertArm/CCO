#include <cassert>
#include <vector>

#include "cco/reference.h"
#include "cco/transform.h"

int main() {
    const std::vector<float> matrix{
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
    };
    cco::TransformConfig config{"0.0.0", {4, 6}, {2, 3}};
    const auto transformed = cco::run_transform(matrix, {4, 6}, config);
    const auto reconstructed = cco::reference_reconstruct(transformed);
    assert(reconstructed.size() == 24);
    assert(reconstructed[0] == 4.5f);
    assert(reconstructed[1] == 4.5f);
    assert(reconstructed[2] == 6.5f);
    assert(reconstructed[6] == 4.5f);
    assert(reconstructed[23] == 20.5f);
    return 0;
}
