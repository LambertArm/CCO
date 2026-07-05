#include <iostream>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    std::cout << "CCO transform bench\n";
    std::cout << "registered transforms: " << transforms.size() << '\n';
    if (!transforms.empty()) {
        std::cout << "active transform version: " << transforms.front() << '\n';
    }
    return 0;
}
