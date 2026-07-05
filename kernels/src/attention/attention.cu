#include "cco/attention.h"

#include <cmath>
#include <stdexcept>

namespace cco {

std::vector<float> run_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 AttentionShape shape) {
    if (shape.seq_len == 0 || shape.head_dim == 0) {
        throw std::invalid_argument("run_attention requires non-zero sequence length and head dimension");
    }

    auto scores = run_qk_matmul(q, k, shape);
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    for (auto& value : scores) {
        value *= scale;
    }
    auto probs = run_row_softmax(scores, shape.seq_len, shape.seq_len);
    return run_pv_matmul(probs, v, shape);
}

} // namespace cco
