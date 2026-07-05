#include "cco/transform.h"

#include <stdexcept>

namespace cco {

namespace {

constexpr const char* kBaselineVersion = "0.0.0";

std::size_t choose_transformed_extent(std::size_t original_extent, std::size_t divisor) {
    if (original_extent == 0 || divisor == 0) {
        return 0;
    }
    const std::size_t candidate = original_extent / divisor;
    if (candidate == 0) {
        return 1;
    }
    if ((original_extent % candidate) == 0) {
        return candidate;
    }
    for (std::size_t probe = candidate; probe > 0; --probe) {
        if ((original_extent % probe) == 0) {
            return probe;
        }
    }
    return 1;
}

} // namespace

MatrixShape default_transformed_shape(const std::string& version, MatrixShape original_shape) {
    if (version != kBaselineVersion) {
        throw std::invalid_argument("no default transformed shape policy registered for version: " + version);
    }
    return {
        choose_transformed_extent(original_shape.rows, kDefaultTransformedAttentionSeqDivisor),
        choose_transformed_extent(original_shape.cols, kDefaultTransformedAttentionDimDivisor),
    };
}

AttentionShape default_transformed_attention_shape(const std::string& version, AttentionShape original_shape) {
    const auto matrix_shape = default_transformed_shape(version, {original_shape.seq_len, original_shape.head_dim});
    return {matrix_shape.rows, matrix_shape.cols};
}

std::vector<std::string> available_transforms() {
    return {
        "0.0.0",
    };
}

} // namespace cco
