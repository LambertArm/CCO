#include "cco/kernel_common.h"

#include <chrono>

namespace cco {

double wall_clock_ms() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

} // namespace cco
