#include <algorithm>
#include <cmath>
#include <vector>

#include "cco/reference.h"

namespace cco {

namespace {

float dot_row(const std::vector<float>& lhs,
              const std::vector<float>& rhs,
              std::size_t lhs_row,
              std::size_t rhs_row,
              std::size_t width) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < width; ++i) {
        acc += lhs[lhs_row * width + i] * rhs[rhs_row * width + i];
    }
    return acc;
}

} // namespace

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       AttentionShape shape) {
    const std::size_t seq_len = shape.seq_len;
    const std::size_t head_dim = shape.head_dim;
    std::vector<float> scores(seq_len * seq_len, 0.0f);
    std::vector<float> output(seq_len * head_dim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t i = 0; i < seq_len; ++i) {
        float max_score = -INFINITY;
        for (std::size_t j = 0; j < seq_len; ++j) {
            const float score = dot_row(q, k, i, j, head_dim) * scale;
            scores[i * seq_len + j] = score;
            max_score = std::max(max_score, score);
        }

        float denom = 0.0f;
        for (std::size_t j = 0; j < seq_len; ++j) {
            const float exp_score = std::exp(scores[i * seq_len + j] - max_score);
            scores[i * seq_len + j] = exp_score;
            denom += exp_score;
        }

        for (std::size_t j = 0; j < seq_len; ++j) {
            const float weight = scores[i * seq_len + j] / denom;
            for (std::size_t d = 0; d < head_dim; ++d) {
                output[i * head_dim + d] += weight * v[j * head_dim + d];
            }
        }
    }

    return output;
}

std::vector<float> run_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 AttentionShape shape) {
    return reference_attention(q, k, v, shape);
}

} // namespace cco
