#pragma once

#include <vector>

#include "cco/types.h"

namespace cco {

std::vector<float> run_qk_matmul(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 AttentionShape shape);
std::vector<float> run_row_softmax(const std::vector<float>& scores,
                                   std::size_t rows,
                                   std::size_t cols);
std::vector<float> run_pv_matmul(const std::vector<float>& probs,
                                 const std::vector<float>& v,
                                 AttentionShape shape);
std::vector<float> run_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 AttentionShape shape);

} // namespace cco
