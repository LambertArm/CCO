#include <iostream>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    std::cout << "CCO transform bench scaffold\n";
    std::cout << "registered transforms: " << transforms.size() << '\n';
    return 0;
}
