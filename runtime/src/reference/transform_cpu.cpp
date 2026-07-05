#include "cco/transform.h"

namespace cco {

TransformResult run_transform(const TransformConfig& config) {
    std::vector<float> zero(config.n * config.n, 0.0f);
    return run_transform(zero, {config.n, config.n}, config);
}

TransformResult run_transform(const std::vector<float>& matrix,
                              MatrixShape shape,
                              const TransformConfig& config) {
    TransformResult result;
    result.version = config.name;
    result.original_shape = shape;
    result.reduced_shape = {config.m, config.m};
    result.block_ratio = config.n / config.m;
    result.reduced_data.assign(config.m * config.m, 0.0f);

    const std::size_t n = shape.rows;
    const std::size_t m = config.m;
    const std::size_t block = n / m;

    for (std::size_t row_block = 0; row_block < m; ++row_block) {
        for (std::size_t col_block = 0; col_block < m; ++col_block) {
            double acc = 0.0;
            for (std::size_t row = 0; row < block; ++row) {
                for (std::size_t col = 0; col < block; ++col) {
                    const std::size_t src_row = row_block * block + row;
                    const std::size_t src_col = col_block * block + col;
                    acc += matrix[src_row * n + src_col];
                }
            }
            result.reduced_data[row_block * m + col_block] =
                static_cast<float>(acc / static_cast<double>(block * block));
        }
    }
    return result;
}

} // namespace cco
