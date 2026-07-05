# Transform Ladder

CCO starts from one explicit transform version for flexible `(n, d) -> (n', d')` attention preprocessing.

The versioned transform files live in `kernels/src/transform/`.

## Rules

- keep one stable public interface
- improve only the transform internals unless a broader design change is necessary
- benchmark every new version against the same shapes
- do not merge a new version without eval output

## Current Version

- `0.0.0`: simple rectangular block-mean transform implemented in `00_baseline.cu`

## Shape Contract

- `Q`, `K`, and `V` start in shape `(n, d)`
- the transform maps them into `(n', d')`
- attention runs fully in the transformed space
- the output is reconstructed back to `(n, d)`
- the current baseline accepts any `(n', d')` that evenly divide `(n, d)`
- if `(n', d')` are omitted in runtime or eval config, the active transform version supplies its own default policy
