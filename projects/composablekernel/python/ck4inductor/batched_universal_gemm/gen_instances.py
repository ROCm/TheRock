# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
from dataclasses import replace
from functools import lru_cache
from typing import List

from ..util import library_path

from .op import CKBatchedGemmOperation

log = logging.getLogger(__name__)


def _ck_library_dir():
    gemm_instances_path = os.path.join(
        library_path(),
        "src",
        "tensor_operation_instance",
        "gpu",
        "gemm_universal_batched",
    )
    if not os.path.exists(gemm_instances_path):
        log.error("CK library path %s does not exist", gemm_instances_path)
        return None
    return gemm_instances_path


def parse_instances(str_instances: List[str]) -> List[CKBatchedGemmOperation]:
    """
    Parse the lines containing Universal Gemm template instances into `CKBatchedGemmOperation` instances
    """

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    op_instances = []
    for line in str_instances:
        s_template_args = line.split("DeviceBatchedGemmMultiD_Xdl_CShuffle_V3")[
            -1
        ].strip("<>, ")
        template_args = []
        i_current = 0
        while i_current < len(s_template_args):
            if s_template_args[i_current] == " ":
                # skip whitespace
                i_current += 1
                continue
            elif s_template_args[i_current : i_current + 2] == "S<":
                # parse template S<Index...>
                i_next = s_template_args.find(">", i_current)
                template_args.append(
                    tuple(map(int, s_template_args[i_current + 2 : i_next].split(",")))
                )
                i_current = i_next + 2
            else:
                # all string attributes must be either type aliases or global constants in C++
                i_next = s_template_args.find(",", i_current)
                template_args.append(
                    maybe_int(
                        s_template_args[i_current : i_next if i_next != -1 else None]
                    )
                )
                if i_next != -1:
                    i_current = i_next + 1
            if i_next == -1:
                break

        # ds layout and dtype are parsed as placeholder; reset value
        template_args[2] = tuple()  # ds layout
        template_args[6] = tuple()  # ds dtype

        new_instance = CKBatchedGemmOperation(
            *template_args,  # type: ignore[arg-type]
        )

        op_instances.append(new_instance)
    return op_instances


@lru_cache(None)
def gen_ops_library() -> List[CKBatchedGemmOperation]:
    """
    Parse the Universal Gemm instances defined in the composable kernel library folder.
    """
    ck_library_dir = _ck_library_dir()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3",
            _ck_library_dir(),
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(grep_result.stdout.strip().split("\n"))

    log.debug("ck instances from library: %d", len(op_instances))

    schedulers = [
        "BlockGemmPipelineScheduler::Intrawave",
        "BlockGemmPipelineScheduler::Interwave",
    ]
    gemm_specs = [
        "GemmSpecialization::Default",
        "GemmSpecialization::MPadding",
        "GemmSpecialization::NPadding",
        "GemmSpecialization::KPadding",
        "GemmSpecialization::MNPadding",
        "GemmSpecialization::MKPadding",
        "GemmSpecialization::NKPadding",
        "GemmSpecialization::MNKPadding",
    ]

    # substitute templated args by looping through their domains
    substitute_instances = []
    for instance in op_instances:
        sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
        sub_spec = instance.gemm_specialization == "GemmSpec"
        schedulers_range = (
            schedulers if sub_scheduler else [instance.block_gemm_pipeline_scheduler]
        )
        spec_range = gemm_specs if sub_spec else [instance.gemm_specialization]
        for scheduler in schedulers_range:
            for spec in spec_range:
                substitute_instances.append(
                    replace(
                        instance,
                        block_gemm_pipeline_scheduler=scheduler,
                        gemm_specialization=spec,
                    )
                )

    return substitute_instances


if __name__ == "__main__":
    print(gen_ops_library())
