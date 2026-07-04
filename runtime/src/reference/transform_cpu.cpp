#include "cco/transform.h"

namespace cco {

TransformResult run_transform(const TransformConfig& config) {
    TransformResult result;
    result.version = config.name;
    result.original_shape = {config.n, config.n};
    result.reduced_shape = {config.m, config.m};
    result.reduced_data.assign(config.m * config.m, 0.0f);
    return result;
}

} // namespace cco
