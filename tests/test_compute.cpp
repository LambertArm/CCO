#include <cassert>
#include <vector>

#include "cco/compute.h"

int main() {
    const std::vector<float> a{1, 2, 3, 4};
    const std::vector<float> b{5, 6, 7, 8};
    const auto out = cco::run_gemm(a, b, {2, 2}, {2, 2});
    assert(out.size() == 4);
    assert(out[0] == 19.0f);
    assert(out[1] == 22.0f);
    assert(out[2] == 43.0f);
    assert(out[3] == 50.0f);
    return 0;
}
