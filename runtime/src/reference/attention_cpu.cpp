#include "cco/reference.h"

namespace cco {

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       AttentionShape shape) {
    return run_attention(q, k, v, shape);
}

} // namespace cco
