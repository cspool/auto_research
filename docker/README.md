# AutoResearch GPU Docker Environment

## Current Docker Storage

Docker daemon data-root is already:

```text
/data3/docker
```

That means locally built images and Docker-managed volumes are stored under
`/data3/docker` by default. Existing Docker volumes currently resolve to:

```text
/data3/docker/volumes/chipyard_sync_vol/_data
/data3/docker/volumes/vscode/_data
```

No local image named `nv-ubuntu` was found in `docker image ls`. The currently
available NVIDIA Ubuntu base image is:

```text
nvidia/cuda:12.8.0-base-ubuntu22.04
```

This project uses `nvidia/cuda:12.8.0-devel-ubuntu22.04` as the Dockerfile base
so CUDA developer tools are available inside the container.

The image installs Node.js 20 for Codex CLI compatibility and pins PyTorch to
the CUDA 12.8 wheel channel:

```text
node v20.20.2
codex-cli 0.134.0
torch 2.11.0+cu128
triton 3.6.0
```

## Build

Build the image from the project root:

```bash
docker build \
  --network host \
  -f docker/Dockerfile.gpu \
  -t auto-research-gpu:cu128 \
  .
```

The resulting image layers will be stored under Docker's data-root:
`/data3/docker`.

The current local build is:

```text
auto-research-gpu:cu128
image id: 292c3482f072
size: 17.6GB
```

## Run

Use the helper:

```bash
bash docker/run_auto_research_gpu.sh
```

For the long-lived container used by outer-agent `docker exec` experiments:

```bash
docker run -d \
  --name auto-research-gpu-dev \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/data3/auto_research_docker_cache/huggingface \
  -e TORCH_HOME=/data3/auto_research_docker_cache/torch \
  -e TRITON_CACHE_DIR=/data3/auto_research_docker_cache/triton \
  -e CODEX_HOME=/data3/auto_research_codex_home \
  -v /data3/auto_research:/workspace/auto_research \
  -v /data3/auto_research_docker_cache:/data3/auto_research_docker_cache \
  -v /data3/auto_research_codex_home:/data3/auto_research_codex_home \
  -w /workspace/auto_research \
  auto-research-gpu:cu128 \
  sleep infinity
```

Enter it with:

```bash
docker exec -it auto-research-gpu-dev bash
```

Equivalent explicit command:

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/data3/auto_research_docker_cache/huggingface \
  -e TORCH_HOME=/data3/auto_research_docker_cache/torch \
  -e TRITON_CACHE_DIR=/data3/auto_research_docker_cache/triton \
  -e CODEX_HOME=/data3/auto_research_codex_home \
  -v /data3/auto_research:/workspace/auto_research \
  -v /data3/auto_research_docker_cache:/data3/auto_research_docker_cache \
  -v /data3/auto_research_codex_home:/data3/auto_research_codex_home \
  -w /workspace/auto_research \
  auto-research-gpu:cu128 \
  /bin/bash
```

The helper uses host networking via `--network host`. Dockerfile cannot enable
host networking for runtime containers; it must be a `docker run` or
`docker compose` setting.

## Configure Codex Inside The Container

Inside the container:

```bash
mkdir -p "${CODEX_HOME:-/data3/auto_research_codex_home}"
cat >> "${CODEX_HOME:-/data3/auto_research_codex_home}/config.toml" <<'EOF'
sandbox_mode = "danger-full-access"
approval_policy = "on-request"
EOF
```

Then run:

```bash
codex -C /workspace/auto_research
```

Docker is now the outer safety boundary. Codex can use
`danger-full-access` inside the container while still being limited to the
container filesystem and the explicitly mounted `/data3` project/cache paths.

## Verify GPU

Inside the container:

```bash
ls -l /dev/nvidia*
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Smoke test from the completed build:

```text
torch 2.11.0+cu128
cuda_available True
device_count 2
device0 NVIDIA GeForce RTX 4090
```

## Optional Profiling Mode

For Nsight Compute/System profiling, a stricter default container is preferred
for ordinary runs. If profiling counters fail, start a dedicated profiling
container with:

```bash
--cap-add SYS_ADMIN --security-opt seccomp=unconfined
```

Only use that mode for H5 profiling runs that need hardware counters.
