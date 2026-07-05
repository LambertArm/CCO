#pragma once

#include <cstddef>
#include <string>

#include "cco/types.h"

namespace cco {

bool cuda_runtime_available();
int cuda_device_count();
std::string cuda_device_name(int device_index = 0);

std::size_t matrix_storage_bytes(MatrixShape shape);
std::size_t attention_storage_bytes(AttentionShape shape);

MatrixShape to_matrix_shape(AttentionShape shape);
bool is_square_matrix_shape(MatrixShape shape);
bool is_valid_transform_config(const TransformConfig& config);
bool is_valid_attention_transform_config(const AttentionTransformConfig& config);
double wall_clock_ms();

} // namespace cco
