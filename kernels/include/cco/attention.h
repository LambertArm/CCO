#pragma once

#include <vector>

#include "cco/types.h"

namespace cco {

std::vector<float> run_attention(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 AttentionShape shape);

} // namespace cco
