#!/usr/bin/env bash
set -euo pipefail

ctest --test-dir build --output-on-failure
