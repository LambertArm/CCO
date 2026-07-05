#include <iostream>
#include <vector>

#include "cco/compute.h"

int main() {
    std::cout << "CCO GEMM bench\n";
    const std::vector<float> a(16 * 16, 1.0f);
    const std::vector<float> b(16 * 16, 2.0f);
    const auto out = cco::run_gemm(a, b, {16, 16}, {16, 16});
    std::cout << "output elements: " << out.size() << '\n';
    return 0;
}
