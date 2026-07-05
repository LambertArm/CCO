#include <iostream>

#include "cco/pipeline.h"

int main() {
    std::cout << "CCO transformed-attention pipeline bench\n";
    const auto result = cco::run_pipeline({"0.0.0"});
    std::cout << "reconstruction error: " << result.metrics.reconstruction_error << '\n';
    return 0;
}
