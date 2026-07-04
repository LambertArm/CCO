#include <cassert>

#include "cco/reference.h"
#include "cco/transform.h"

int main() {
    cco::TransformConfig config{"0.0.0", 16, 8};
    const auto transformed = cco::run_transform(config);
    const auto reconstructed = cco::reference_reconstruct(transformed);
    assert(reconstructed.size() == 64);
    return 0;
}
