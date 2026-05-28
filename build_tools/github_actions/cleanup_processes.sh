#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cleans up GPU processes to prevent resource contention and ensure clean test environments.
#
# This script handles:
#   - Container-local cleanup: kills GPU processes from the current container's build directory
#   - Machine-wide orphan cleanup: kills orphaned GPU processes from dead/crashed containers
#
# Orphan detection criteria (any match triggers cleanup):
#   - Process running longer than MAX_PROCESS_AGE_MINUTES (default: 360 min)
#   - Process container's cgroup no longer exists
#   - Process parent is PID 1 (adopted by init) and was in a container

set -o pipefail

echo "[*] ==== Starting cleanup_processes.sh ===="

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
CLEANUP_GPU_RESET="${CLEANUP_GPU_RESET:-false}"
CLEANUP_ORPHANS="${CLEANUP_ORPHANS:-true}"
MAX_PROCESS_AGE_MINUTES="${MAX_PROCESS_AGE_MINUTES:-360}"
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

get_container_id_for_pid() {
    local pid="$1"
    local cgroup_file="/proc/${pid}/cgroup"
    local container_id=""

    if [[ -f "$cgroup_file" ]]; then
        container_id=$(grep -oP '(?<=/docker/)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=/libpod-)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=:cpuset:/docker/)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
        if [[ -z "$container_id" ]]; then
            container_id=$(grep -oP '(?<=:cpuset:/libpod-)[a-f0-9]{12,64}' "$cgroup_file" 2>/dev/null | head -1)
        fi
    fi

    echo "$container_id"
}

is_process_too_old() {
    local pid="$1"
    local max_age_seconds=$((MAX_PROCESS_AGE_MINUTES * 60))
    local start_time
    start_time=$(stat -c %Y "/proc/${pid}" 2>/dev/null) || return 1
    local current_time
    current_time=$(date +%s)
    local age=$((current_time - start_time))

    [[ $age -gt $max_age_seconds ]]
}

is_cgroup_orphaned() {
    local pid="$1"
    local container_id
    container_id=$(get_container_id_for_pid "$pid")

    [[ -z "$container_id" ]] && return 1

    # Check if container's cgroup still exists (cgroup v2)
    if [[ -d "/sys/fs/cgroup/system.slice/docker-${container_id}.scope" ]] || \
       [[ -d "/sys/fs/cgroup/machine.slice/libpod-${container_id}.scope" ]]; then
        return 1
    fi

    # Check cgroup v1 paths
    if [[ -d "/sys/fs/cgroup/cpu/docker/${container_id}" ]] || \
       [[ -d "/sys/fs/cgroup/cpu/libpod-${container_id}" ]]; then
        return 1
    fi

    return 0
}

is_parent_dead() {
    local pid="$1"
    local ppid
    ppid=$(awk '{print $4}' "/proc/${pid}/stat" 2>/dev/null) || return 1

    # If parent is PID 1 and process was in a container, it's likely orphaned
    if [[ "$ppid" == "1" ]]; then
        local container_id
        container_id=$(get_container_id_for_pid "$pid")
        [[ -n "$container_id" ]]
    else
        return 1
    fi
}

is_orphaned_process() {
    local pid="$1"

    if is_process_too_old "$pid"; then
        echo "too_old"
        return 0
    fi

    if is_cgroup_orphaned "$pid"; then
        echo "cgroup_orphaned"
        return 0
    fi

    if is_parent_dead "$pid"; then
        echo "parent_dead"
        return 0
    fi

    return 1
}

find_orphaned_gpu_processes() {
    local pids=()

    if [[ ! -d /proc ]]; then
        echo ""
        return
    fi

    for pid_dir in /proc/[0-9]*; do
        local pid="${pid_dir##*/}"
        [[ ! -d "${pid_dir}/fd" ]] && continue

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
            # Skip processes in our own container
            if [[ -n "$CONTAINER_ID" ]] && is_same_container "$pid" "$CONTAINER_ID"; then
                continue
            fi

            # Check if process is orphaned
            if is_orphaned_process "$pid" >/dev/null; then
                pids+=("$pid")
            fi
        fi
    done

    echo "${pids[@]}"
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
    echo "[+] ==== Container cleanup completed successfully ===="
else
    echo "[-] ==== Container cleanup completed with ${#REMAINING_PIDS[@]} process(es) still running ===="
    for pid in "${REMAINING_PIDS[@]}"; do
        echo "    > $(get_process_info "$pid")"
    done
fi

# Machine-wide orphan cleanup (optional)
if [[ "$CLEANUP_ORPHANS" == "true" ]]; then
    echo ""
    echo "[*] ==== Starting machine-wide orphan cleanup ===="
    echo "[*] Max process age: ${MAX_PROCESS_AGE_MINUTES} minutes"

    read -ra ORPHAN_PIDS <<< "$(find_orphaned_gpu_processes)"

    if [[ ${#ORPHAN_PIDS[@]} -eq 0 ]]; then
        echo "[+] No orphaned GPU processes found"
    else
        echo "[*] Found ${#ORPHAN_PIDS[@]} orphaned GPU process(es):"
        for pid in "${ORPHAN_PIDS[@]}"; do
            reason=$(is_orphaned_process "$pid")
            echo "    > $(get_process_info "$pid") [reason: ${reason}]"
        done

        ORPHAN_KILLABLE=()
        ORPHAN_STUCK=()
        for pid in "${ORPHAN_PIDS[@]}"; do
            if is_uninterruptible "$pid"; then
                ORPHAN_STUCK+=("$pid")
            else
                ORPHAN_KILLABLE+=("$pid")
            fi
        done

        if [[ ${#ORPHAN_STUCK[@]} -gt 0 ]]; then
            echo "[!] WARNING: ${#ORPHAN_STUCK[@]} orphaned process(es) in D state"
            EXIT_CODE=1
        fi

        if [[ ${#ORPHAN_KILLABLE[@]} -gt 0 ]]; then
            echo "[*] Sending SIGTERM to ${#ORPHAN_KILLABLE[@]} orphaned process(es)..."
            for pid in "${ORPHAN_KILLABLE[@]}"; do
                echo "    > Terminating $(get_process_info "$pid")"
                kill -TERM "$pid" 2>/dev/null || true
            done

            if ! wait_for_termination "${ORPHAN_KILLABLE[@]}"; then
                echo "[*] Some orphans did not terminate, sending SIGKILL..."
                for pid in "${ORPHAN_KILLABLE[@]}"; do
                    if [[ -d "/proc/${pid}" ]]; then
                        echo "    > Force killing $(get_process_info "$pid")"
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                done

                sleep 2
                for pid in "${ORPHAN_KILLABLE[@]}"; do
                    if [[ -d "/proc/${pid}" ]]; then
                        echo "[-] Failed to kill orphan: $(get_process_info "$pid")"
                        EXIT_CODE=1
                    fi
                done
            fi
        fi

        echo "[+] ==== Orphan cleanup completed ===="
    fi
fi

exit $EXIT_CODE
