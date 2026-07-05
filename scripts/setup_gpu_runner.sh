#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Cuda-Compute-OSS}"
VENV_DIR="${VENV_DIR:-$HOME/cco-venv}"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "CCO GPU runner setup"
echo "repo dir:   $REPO_DIR"
echo "venv dir:   $VENV_DIR"
echo "runner dir: $RUNNER_DIR"
echo

sudo apt update
sudo apt install -y \
  git \
  curl \
  build-essential \
  cmake \
  "$PYTHON_BIN" \
  python3-venv \
  python3-pip

echo
echo "Checking NVIDIA driver availability..."
nvidia-smi

echo
echo "Creating Python virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo
echo "Installing PyTorch with CUDA wheels from:"
echo "  $PYTORCH_INDEX_URL"
pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
pip install pyyaml

echo
echo "Verifying PyTorch CUDA..."
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in torch")
print("gpu:", torch.cuda.get_device_name(0))
print("cuda runtime:", torch.version.cuda)
PY

if [ -d "$REPO_DIR" ]; then
  echo
  echo "Running repo smoke checks..."
  cd "$REPO_DIR"
  python3 eval/run_eval.py
  python3 eval/run_eval_gpu.py --seed 1234
else
  echo
  echo "Repo directory not found at $REPO_DIR."
  echo "Clone the repo there if you want the script to run repo smoke checks."
fi

mkdir -p "$RUNNER_DIR"

cat <<EOF

Machine prep completed.

Next steps in GitHub:
1. Open the repo in GitHub
2. Go to Settings -> Actions -> Runners
3. Click "New self-hosted runner"
4. Choose the correct OS and architecture
5. Run the commands GitHub shows inside:
   $RUNNER_DIR

Important runner labels:
- self-hosted
- gpu

Your workflows target:
  runs-on: [self-hosted, gpu]

After registration, start the runner with:
  ./run.sh

Or install it as a service:
  sudo ./svc.sh install
  sudo ./svc.sh start

Then use the GitHub Actions UI to run:
  GPU Queue

EOF
