#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cleans up GPU processes from the current container to prevent "VRAM not clean" errors.

set -o pipefail

echo "[*] ==== Starting cleanup_processes.sh ===="

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
CLEANUP_GPU_RESET="${CLEANUP_GPU_RESET:-false}"
BUILD_DIR="${WORKSPACE}/build"
EXIT_CODE=0

get_container_id() {
    # Extract container ID from cgroup (works for Docker and Podman)
    local cgroup_file="/proc/self/cgroup"
    local container_id=""

    if [[ -f "$cgroup_file" ]]; then
        # Try cgroup v2 format (e.g., /docker/<id> or /libpod-<id>)
        container_id=$(grep -oP '(?<=/docker/)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=/libpod-)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
        # Try cgroup v1 format
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=:cpuset:/docker/)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=:cpuset:/libpod-)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
    fi

    echo "$container_id"
}

is_same_container() {
    local pid="$1"
    local our_container_id="$2"

    [[ -z "$our_container_id" ]] && return 1

    local pid_cgroup="/proc/${pid}/cgroup"
    [[ ! -f "$pid_cgroup" ]] && return 1

    grep -q "$our_container_id" "$pid_cgroup" 2>/dev/null
}

find_gpu_processes() {
    local container_id="$1"
    local pids=()

    if [[ -d /proc ]]; then
        for pid_dir in /proc/[0-9]*; do
            local pid="${pid_dir##*/}"
            [[ ! -d "${pid_dir}/fd" ]] && continue

            # Check if process is in our container
            if [[ -n "$container_id" ]] && ! is_same_container "$pid" "$container_id"; then
                continue
            fi

            # Check if process has GPU device file descriptors open
            local has_gpu_fd=false
            for fd in "${pid_dir}"/fd/*; do
                local target
                target=$(readlink "$fd" 2>/dev/null) || continue
                if [[ "$target" == /dev/kfd || "$target" == /dev/dri/* ]]; then
                    has_gpu_fd=true
                    break
                fi
            done

            if [[ "$has_gpu_fd" == "true" ]]; then
                # Additional filter: only processes from our build directory
                local exe
                exe=$(readlink "${pid_dir}/exe" 2>/dev/null) || continue
                if [[ "$exe" == "${BUILD_DIR}"/* ]]; then
                    pids+=("$pid")
                fi
            fi
        done
    fi

    echo "${pids[@]}"
}

get_process_info() {
    local pid="$1"
    local exe name state
    exe=$(readlink "/proc/${pid}/exe" 2>/dev/null) || exe="<unknown>"
    name=$(cat "/proc/${pid}/comm" 2>/dev/null) || name="<unknown>"
    state=$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null) || state="?"
    echo "[pid:${pid}][state:${state}] ${name} (${exe})"
}

is_uninterruptible() {
    local pid="$1"
    local state
    state=$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null) || return 1
    [[ "$state" == "D" ]]
}

wait_for_termination() {
    local -a pids=("$@")
    local max_wait=10

    for ((i = 0; i < max_wait; i++)); do
        local remaining=()
        for pid in "${pids[@]}"; do
            if [[ -d "/proc/${pid}" ]]; then
                remaining+=("$pid")
            fi
        done

        if [[ ${#remaining[@]} -eq 0 ]]; then
            echo "[+] All processes terminated after ${i} second(s)"
            return 0
        fi

        echo "    > Waiting for ${#remaining[@]} process(es)..."
        sleep 1
        pids=("${remaining[@]}")
    done

    echo "[-] ${#pids[@]} process(es) still running after ${max_wait} seconds"
    return 1
}

CONTAINER_ID=$(get_container_id)
if [[ -n "$CONTAINER_ID" ]]; then
    echo "[*] Container ID: ${CONTAINER_ID:0:12}"
else
    echo "[*] Not running in a container (or container ID not detected)"
fi
echo "[*] Build directory: ${BUILD_DIR}"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "[*] Build directory does not exist, nothing to clean up"
    exit 0
fi

echo "[*] Searching for GPU processes from this container's build directory..."
read -ra GPU_PIDS <<< "$(find_gpu_processes "$CONTAINER_ID")"

if [[ ${#GPU_PIDS[@]} -eq 0 ]]; then
    echo "[+] No GPU processes found"
    exit 0
fi

echo "[*] Found ${#GPU_PIDS[@]} GPU process(es) to clean up:"
for pid in "${GPU_PIDS[@]}"; do
    echo "    > $(get_process_info "$pid")"
done

KILLABLE_PIDS=()
STUCK_PIDS=()
for pid in "${GPU_PIDS[@]}"; do
    if is_uninterruptible "$pid"; then
        STUCK_PIDS+=("$pid")
    else
        KILLABLE_PIDS+=("$pid")
    fi
done

if [[ ${#STUCK_PIDS[@]} -gt 0 ]]; then
    echo "[!] WARNING: ${#STUCK_PIDS[@]} process(es) in uninterruptible sleep (D state):"
    for pid in "${STUCK_PIDS[@]}"; do
        echo "    > $(get_process_info "$pid")"
    done
    echo "[!] These processes cannot be killed and may require GPU reset or node reboot"
    EXIT_CODE=1
fi

if [[ ${#KILLABLE_PIDS[@]} -gt 0 ]]; then
    echo "[*] Sending SIGTERM to ${#KILLABLE_PIDS[@]} process(es)..."
    for pid in "${KILLABLE_PIDS[@]}"; do
        echo "    > Terminating $(get_process_info "$pid")"
        kill -TERM "$pid" 2>/dev/null || true
    done

    if ! wait_for_termination "${KILLABLE_PIDS[@]}"; then
        echo "[*] Some processes did not terminate, sending SIGKILL..."
        for pid in "${KILLABLE_PIDS[@]}"; do
            if [[ -d "/proc/${pid}" ]]; then
                echo "    > Force killing $(get_process_info "$pid")"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done

        sleep 2
        for pid in "${KILLABLE_PIDS[@]}"; do
            if [[ -d "/proc/${pid}" ]]; then
                echo "[-] Failed to kill process: $(get_process_info "$pid")"
                EXIT_CODE=1
            fi
        done
    fi
fi

if [[ "$CLEANUP_GPU_RESET" == "true" && ${#STUCK_PIDS[@]} -gt 0 ]]; then
    echo "[*] Attempting GPU reset via rocm-smi..."
    if command -v rocm-smi &>/dev/null; then
        if rocm-smi --gpureset 2>&1; then
            echo "[+] GPU reset completed"
            sleep 2
            for pid in "${STUCK_PIDS[@]}"; do
                if [[ ! -d "/proc/${pid}" ]]; then
                    echo "[+] Process ${pid} terminated after GPU reset"
                fi
            done
        else
            echo "[-] GPU reset failed (may require elevated permissions)"
        fi
    else
        echo "[-] rocm-smi not found, cannot perform GPU reset"
    fi
fi

REMAINING_PIDS=()
for pid in "${GPU_PIDS[@]}"; do
    if [[ -d "/proc/${pid}" ]]; then
        REMAINING_PIDS+=("$pid")
    fi
done

if [[ ${#REMAINING_PIDS[@]} -eq 0 ]]; then
    echo "[+] ==== Cleanup completed successfully ===="
else
    echo "[-] ==== Cleanup completed with ${#REMAINING_PIDS[@]} process(es) still running ===="
    for pid in "${REMAINING_PIDS[@]}"; do
        echo "    > $(get_process_info "$pid")"
    done
fi

exit $EXIT_CODE
