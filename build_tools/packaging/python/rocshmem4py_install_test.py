#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke test an installed rocshmem4py wheel on the local GPU.

This is an installed-package integration test: the caller creates the Python
environment and selects a GPU before running this script with that environment's
interpreter.
"""

import logging

import rocshmem4py


def main() -> None:
    logging.info(
        "Testing rocshmem4py %s linked with rocSHMEM %s",
        rocshmem4py.__version__,
        rocshmem4py.__rocshmem_version__,
    )
    logging.info("Selecting the HIP device from the runner environment")
    rocshmem4py.set_hip_device_from_env()

    logging.info("Creating a single-PE rocSHMEM context")
    unique_id = rocshmem4py.rocshmem_get_uniqueid()
    rocshmem4py.rocshmem_init_attr(0, 1, unique_id)
    try:
        logging.info("Allocating a 4096-byte symmetric GPU buffer")
        buffer = rocshmem4py.rocshmem_create_buffer(4096)
        if buffer.ptr <= 0:
            raise RuntimeError("rocshmem4py returned an invalid GPU buffer")
        logging.info("Freeing the symmetric GPU buffer")
        buffer.free()
    finally:
        logging.info("Synchronizing the HIP device")
        try:
            rocshmem4py.hip_device_synchronize()
        finally:
            logging.info("Running the single-PE rocSHMEM barrier")
            try:
                rocshmem4py.rocshmem_barrier_all()
            finally:
                logging.info("Finalizing rocSHMEM")
                rocshmem4py.rocshmem_finalize()

    logging.info("rocshmem4py installed-wheel GPU smoke test passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
