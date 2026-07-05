# GPU Runner Setup

This guide connects the repo's PR bot flow to a rented GPU machine.

## Why This Exists

CCO uses a two-stage PR bot flow:

1. CPU-only precheck runs automatically on PR open/update
2. accepted PRs enter the ordered GPU eval queue

The GPU queue worker needs a self-hosted GitHub Actions runner with a working CUDA stack.

## One-Time Machine Prep

Run:

```bash
bash scripts/setup_gpu_runner.sh
```

That script:

- installs basic Ubuntu build tools
- checks `nvidia-smi`
- creates a Python virtual environment
- installs PyTorch with CUDA wheels
- installs repo Python dependencies
- runs:
  - `python3 eval/run_eval.py`
  - `python3 eval/run_eval_gpu.py --seed 1234`

Environment variables you can override:

```bash
REPO_DIR=/path/to/Cuda-Compute-OSS
VENV_DIR=$HOME/cco-venv
RUNNER_DIR=$HOME/actions-runner
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
```

Example:

```bash
REPO_DIR=$HOME/work/Cuda-Compute-OSS bash scripts/setup_gpu_runner.sh
```

## Register The Self-Hosted Runner

In GitHub:

1. open the repository
2. go to `Settings` -> `Actions` -> `Runners`
3. click `New self-hosted runner`
4. choose the machine OS and architecture
5. run the shown commands on the rented GPU machine

The runner must include the `gpu` label.

Your workflows target:

```yaml
runs-on: [self-hosted, gpu]
```

## Start The Runner

Foreground mode:

```bash
cd ~/actions-runner
./run.sh
```

Service mode:

```bash
cd ~/actions-runner
sudo ./svc.sh install
sudo ./svc.sh start
```

## How The Queue Works

After the runner is online:

- PR updates trigger `.github/workflows/pr-precheck.yml`
- passing PRs get `status:gpu-queued`
- failing PRs get `status:precheck-failed` and are auto-closed

When you want to spend GPU time:

1. open GitHub `Actions`
2. choose `GPU Queue`
3. click `Run workflow`

That worker:

- finds queued PRs
- sorts them by arrival order
- runs real GPU eval one by one
- comments the result back on each PR
- labels each PR as:
  - `status:gpu-passed`
  - or `status:gpu-failed`

## Safety Note

Self-hosted runners are safest with private repositories or trusted contributors because workflow code runs on your machine.
