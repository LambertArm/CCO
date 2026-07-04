#include <iostream>

#include "cco/transform.h"

int main() {
    for (const auto& name : cco::available_transforms()) {
        std::cout << name << '\n';
    }
    return 0;
}
