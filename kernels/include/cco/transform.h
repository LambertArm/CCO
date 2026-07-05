#pragma once

#include <string>
#include <vector>

#include "cco/types.h"

namespace cco {

struct TransformResult {
    std::string version;
    MatrixShape original_shape;
    MatrixShape transformed_shape;
    std::size_t row_block_ratio{};
    std::size_t col_block_ratio{};
    std::vector<float> transformed_data;
};

TransformResult run_transform(const TransformConfig& config);
TransformResult run_transform(const std::vector<float>& matrix,
                              MatrixShape shape,
                              const TransformConfig& config);
MatrixShape default_transformed_shape(const std::string& version, MatrixShape original_shape);
AttentionShape default_transformed_attention_shape(const std::string& version, AttentionShape original_shape);
std::vector<float> reconstruct_matrix(const TransformResult& result);
std::vector<std::string> available_transforms();

} // namespace cco
