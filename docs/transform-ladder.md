# Transform Ladder

CCO starts from one explicit transform version.

The versioned transform files live in `kernels/src/transform/`.

## Rules

- keep one stable public interface
- improve only the transform internals unless a broader design change is necessary
- benchmark every new version against the same shapes
- do not merge a new version without eval output

## Current Version

- `0.0.0`: simple reference reduction path implemented in `00_baseline.cu`
