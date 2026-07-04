#include <cmath>

#include "cco/reference.h"
#include "cco/transform.h"

namespace cco {

std::vector<float> reference_reconstruct(const TransformResult& result) {
    return result.reduced_data;
}

double reference_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    const std::size_t count = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
        total += diff * diff;
    }
    return std::sqrt(total);
}

} // namespace cco
