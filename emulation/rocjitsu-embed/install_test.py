# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the standalone rocjitsu embedding library and its memory helper."""

import argparse
import os
from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_void_p, cast
from pathlib import Path
from tempfile import TemporaryDirectory


def check_status(operation: str, status: int) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} returned rocjitsu status {status}")


def validate_install(library: Path) -> None:
    rocjitsu = CDLL(library.resolve(strict=True), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    libc = CDLL("libc.so.6")
    if cast(rocjitsu.open, c_void_p).value != cast(libc.open, c_void_p).value:
        raise RuntimeError(f"{library} exports a libc open interposer")

    rocjitsu.rj_vm_create_from_string.argtypes = [c_char_p, c_int, POINTER(c_void_p)]
    rocjitsu.rj_vm_topology_path.argtypes = [c_void_p, POINTER(c_char_p)]
    rocjitsu.rj_vm_destroy_checked.argtypes = [c_void_p]

    config = b"""{
      "num_threads": 1, "cpu_dispatch_threads": 1,
      "vm": {"arch": "cdna4", "gpu": {"device": {
        "gpu_id": 42, "gfx_target_version": 90500, "num_sdma_engines": 0
      }}},
      "topology": {"root": {"name": "soc", "type": "soc", "children": [
        {"name": "vram", "type": "gpu_memory"}
      ]}}
    }"""
    vm = c_void_p()
    local_mode = 1  # RJ_VM_MODE_LOCAL from rocjitsu/vm/rj_vm.h
    check_status(
        "rj_vm_create_from_string",
        rocjitsu.rj_vm_create_from_string(config, local_mode, byref(vm)),
    )
    try:
        topology = c_char_p()
        check_status(
            "rj_vm_topology_path", rocjitsu.rj_vm_topology_path(vm, byref(topology))
        )
        if not topology.value:
            raise RuntimeError("rj_vm_topology_path returned an empty path")
        # The VM can return a directory even if its memory helper failed to
        # populate the topology files. Check the data an embedding host reads.
        gpu_id_file = Path(os.fsdecode(topology.value)) / "nodes/1/gpu_id"
        gpu_id = gpu_id_file.read_text().strip()
        if gpu_id != "42":
            raise RuntimeError(f"{gpu_id_file}: expected GPU ID 42, read {gpu_id!r}")
    finally:
        check_status("rj_vm_destroy_checked", rocjitsu.rj_vm_destroy_checked(vm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, help="Path to librocjitsu_embed.so")
    args = parser.parse_args()
    with TemporaryDirectory() as runtime_dir:
        os.environ["ROCJITSU_RUNTIME_DIR"] = runtime_dir
        validate_install(args.library)
