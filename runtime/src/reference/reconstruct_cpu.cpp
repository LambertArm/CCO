#include <cmath>

#include "cco/reference.h"
#include "cco/transform.h"

namespace cco {

std::vector<float> reference_reconstruct(const TransformResult& result) {
    return reconstruct_matrix(result);
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

std::vector<float> reconstruct_matrix(const TransformResult& result) {
    const std::size_t n = result.original_shape.rows;
    const std::size_t m = result.reduced_shape.rows;
    const std::size_t block = result.block_ratio == 0 ? (n / m) : result.block_ratio;
    std::vector<float> out(n * n, 0.0f);

    for (std::size_t row_block = 0; row_block < m; ++row_block) {
        for (std::size_t col_block = 0; col_block < m; ++col_block) {
            const float value = result.reduced_data[row_block * m + col_block];
            for (std::size_t row = 0; row < block; ++row) {
                for (std::size_t col = 0; col < block; ++col) {
                    const std::size_t dst_row = row_block * block + row;
                    const std::size_t dst_col = col_block * block + col;
                    out[dst_row * n + dst_col] = value;
                }
            }
        }
    }

    return out;
}

} // namespace cco
