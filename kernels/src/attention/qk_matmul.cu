#include "cco/attention.h"

#include <stdexcept>

namespace cco {

std::vector<float> run_qk_matmul(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 AttentionShape shape) {
    const std::size_t seq_len = shape.seq_len;
    const std::size_t head_dim = shape.head_dim;
    if (q.size() != seq_len * head_dim || k.size() != seq_len * head_dim) {
        throw std::invalid_argument("run_qk_matmul received mismatched tensor storage");
    }

    std::vector<float> scores(seq_len * seq_len, 0.0f);
    for (std::size_t i = 0; i < seq_len; ++i) {
        for (std::size_t j = 0; j < seq_len; ++j) {
            float acc = 0.0f;
            for (std::size_t d = 0; d < head_dim; ++d) {
                acc += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = acc;
        }
    }
    return scores;
}

} // namespace cco
