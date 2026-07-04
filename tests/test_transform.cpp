#include <cassert>

#include "cco/transform.h"

int main() {
    const auto transforms = cco::available_transforms();
    assert(!transforms.empty());
    return 0;
}
