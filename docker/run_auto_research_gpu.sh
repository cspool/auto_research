#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-auto-research-gpu:cu128}"
CONTAINER_NAME="${CONTAINER_NAME:-auto_research_dev}"
PROJECT_DIR="${PROJECT_DIR:-/data3/auto_research}"
CACHE_DIR="${CACHE_DIR:-/data3/auto_research_docker_cache}"
HOST_HOME="${HOST_HOME:-${HOME}}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
HOST_USER="${HOST_USER:-$(id -un)}"
HOST_GROUP="${HOST_GROUP:-$(id -gn)}"
CONTAINER_HOME="${CONTAINER_HOME:-${HOST_HOME}}"
CONTAINER_CODEX_HOME="${CONTAINER_CODEX_HOME:-${CONTAINER_HOME}/.codex}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/data3/auto_research_docker_cache}"
CONTAINER_HOME_CACHE_DIR="${CONTAINER_HOME_CACHE_DIR:-${CACHE_DIR}/home/${HOST_USER}}"
NPM_GLOBAL_DIR="${NPM_GLOBAL_DIR:-${CACHE_DIR}/npm-global}"
CONTAINER_NPM_GLOBAL_DIR="${CONTAINER_NPM_GLOBAL_DIR:-${CONTAINER_CACHE_DIR}/npm-global}"
RECREATE="${RECREATE:-0}"

mkdir -p "${CACHE_DIR}"

if [ ! -w "${CACHE_DIR}" ]; then
  echo "Cache dir is not writable by ${HOST_USER}:${HOST_GROUP} (${HOST_UID}:${HOST_GID}): ${CACHE_DIR}" >&2
  echo "Please chown it or set CACHE_DIR to a writable path." >&2
  exit 1
fi

mkdir -p "${CONTAINER_HOME_CACHE_DIR}" "${NPM_GLOBAL_DIR}" "${CACHE_DIR}/xdg-cache"

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

if container_exists; then
  if [ "${RECREATE}" = "1" ]; then
    docker rm -f "${CONTAINER_NAME}"
  else
    if container_running; then
      echo "Container ${CONTAINER_NAME} is already running."
      echo "Attach with: docker exec -it ${CONTAINER_NAME} /bin/bash"
    else
      echo "Container ${CONTAINER_NAME} already exists but is stopped."
      echo "Start it with: docker start -ai ${CONTAINER_NAME}"
    fi
    echo "To recreate it with the mounts in this script, run:"
    echo "  RECREATE=1 $0"
    exit 1
  fi
fi

docker_args=(
  run
  -it
  --name "${CONTAINER_NAME}"
  --user "${HOST_UID}:${HOST_GID}"
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e USER="${HOST_USER}"
  -e LOGNAME="${HOST_USER}"
  -e HOME="${CONTAINER_HOME}"
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  -e HF_HOME="${CONTAINER_CACHE_DIR}/huggingface"
  -e TORCH_HOME="${CONTAINER_CACHE_DIR}/torch"
  -e TRITON_CACHE_DIR="${CONTAINER_CACHE_DIR}/triton"
  -e XDG_CACHE_HOME="${CONTAINER_CACHE_DIR}/xdg-cache"
  -e NPM_CONFIG_PREFIX="${CONTAINER_NPM_GLOBAL_DIR}"
  -e PATH="${CONTAINER_NPM_GLOBAL_DIR}/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin"
  -e CODEX_HOME="${CONTAINER_CODEX_HOME}"
  -v "${PROJECT_DIR}:/workspace/auto_research"
  -v "${CACHE_DIR}:${CONTAINER_CACHE_DIR}"
  -v "${CONTAINER_HOME_CACHE_DIR}:${CONTAINER_HOME}"
  -v "${NPM_GLOBAL_DIR}:${CONTAINER_NPM_GLOBAL_DIR}"
  -v /etc/passwd:/etc/passwd:ro
  -v /etc/group:/etc/group:ro
  -w /workspace/auto_research
)

for group_id in $(id -G); do
  if [ "${group_id}" != "${HOST_GID}" ]; then
    docker_args+=(--group-add "${group_id}")
  fi
done

add_mount_if_exists() {
  local source_path="$1"
  local target_path="$2"
  local mode="${3:-}"

  if [ -e "${source_path}" ]; then
    if [ -n "${mode}" ]; then
      docker_args+=(-v "${source_path}:${target_path}:${mode}")
    else
      docker_args+=(-v "${source_path}:${target_path}")
    fi
  else
    echo "Skip missing mount source: ${source_path}"
  fi
}

add_mount_if_exists "${HOST_HOME}/.codex" "${CONTAINER_CODEX_HOME}"
add_mount_if_exists "${HOST_HOME}/.orchestra" "${CONTAINER_HOME}/.orchestra" "ro"
add_mount_if_exists "${HOST_HOME}/.claude" "${CONTAINER_HOME}/.claude"
add_mount_if_exists "${HOST_HOME}/.claude.json" "${CONTAINER_HOME}/.claude.json"
add_mount_if_exists "${HOST_HOME}/.bashrc" "${CONTAINER_HOME}/.bashrc" "ro"
add_mount_if_exists "${HOST_HOME}/.gitconfig" "${CONTAINER_HOME}/.gitconfig" "ro"
add_mount_if_exists "${HOST_HOME}/.git-credentials" "${CONTAINER_HOME}/.git-credentials" "ro"
add_mount_if_exists "${HOST_HOME}/.ssh" "${CONTAINER_HOME}/.ssh" "ro"
add_mount_if_exists "${HOST_HOME}/.config/gh" "${CONTAINER_HOME}/.config/gh"

docker "${docker_args[@]}" \
  "${IMAGE_NAME}" \
  bash -lc 'mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${NPM_CONFIG_PREFIX}"; command -v codex >/dev/null 2>&1 || npm install -g @openai/codex; command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code; exec /bin/bash'
