#include <cassert>
#include <vector>

#include "cco/attention.h"

int main() {
    const std::vector<float> q{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.5f, -0.5f,
    };
    const std::vector<float> k{
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        1.0f, -1.0f,
    };
    const std::vector<float> v{
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };

    const auto out = cco::run_attention(q, k, v, {4, 2});
    assert(out.size() == 8);
    for (float value : out) {
        assert(value > 0.0f);
    }
    return 0;
}
