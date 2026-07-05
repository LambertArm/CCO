#include <cassert>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    assert(transforms.size() == 1);
    assert(transforms.front() == "0.0.0");
    return 0;
}
