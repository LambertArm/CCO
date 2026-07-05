#include "cco/attention.h"

#include <cmath>
#include <stdexcept>

namespace cco {

std::vector<float> run_row_softmax(const std::vector<float>& scores,
                                   std::size_t rows,
                                   std::size_t cols) {
    if (scores.size() != rows * cols) {
        throw std::invalid_argument("run_row_softmax received mismatched matrix storage");
    }

    std::vector<float> probs(scores.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        float max_score = scores[row * cols];
        for (std::size_t col = 1; col < cols; ++col) {
            const float value = scores[row * cols + col];
            if (value > max_score) {
                max_score = value;
            }
        }

        float denom = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            const float exp_value = std::exp(scores[row * cols + col] - max_score);
            probs[row * cols + col] = exp_value;
            denom += exp_value;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            probs[row * cols + col] /= denom;
        }
    }
    return probs;
}

} // namespace cco
