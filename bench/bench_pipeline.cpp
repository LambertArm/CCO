#include <iostream>

#include "cco/pipeline.h"

int main() {
    std::cout << "CCO matrix-transform-plus-GEMM pipeline bench\n";
    const auto result = cco::run_pipeline({"0.0.0", 64, 16});
    std::cout << "reconstruction error: " << result.metrics.reconstruction_error << '\n';
    return 0;
}
