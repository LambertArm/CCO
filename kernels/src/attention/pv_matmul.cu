#include "cco/attention.h"

#include <stdexcept>

namespace cco {

std::vector<float> run_pv_matmul(const std::vector<float>& probs,
                                 const std::vector<float>& v,
                                 AttentionShape shape) {
    const std::size_t seq_len = shape.seq_len;
    const std::size_t head_dim = shape.head_dim;
    if (probs.size() != seq_len * seq_len || v.size() != seq_len * head_dim) {
        throw std::invalid_argument("run_pv_matmul received mismatched tensor storage");
    }

    std::vector<float> out(seq_len * head_dim, 0.0f);
    for (std::size_t row = 0; row < seq_len; ++row) {
        for (std::size_t col = 0; col < seq_len; ++col) {
            const float weight = probs[row * seq_len + col];
            for (std::size_t d = 0; d < head_dim; ++d) {
                out[row * head_dim + d] += weight * v[col * head_dim + d];
            }
        }
    }
    return out;
}

} // namespace cco
