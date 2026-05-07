# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
from dataclasses import replace
from functools import lru_cache
from typing import List

from ..util import library_path

from .op import CKGroupedConvFwdOp

log = logging.getLogger(__name__)


def _ck_conv_instances_path():
    conv_instances_path = os.path.join(  # noqa: F821
        library_path(),
        "include",
        "ck",
        "library",
        "tensor_operation_instance",
        "gpu",
        "grouped_conv_fwd",
    )
    if not os.path.exists(conv_instances_path):
        log.error(
            "CK library conv instances path %s does not exist", conv_instances_path
        )
        return None
    return conv_instances_path


def parse_instances(str_instances: List[str]) -> List[CKGroupedConvFwdOp]:
    """
    Parse the lines containing Grouped Convolution Forward template instances
    into `CKGroupedConvFwdOp` instances
    """

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    op_instances = []
    # TODO: maybe use libclang for parsing C++ code in the future
    # to avoid this hacky parsing logic below ? :) - copilot
    for line in str_instances:
        s_template_args = line.split("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3")[
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

        template_args[0] = -1  # n_dim_spatial
        template_args[3] = tuple()  # ds_layout
        template_args[9] = tuple()  # ds_element_dtype

        new_instance = CKGroupedConvFwdOp(
            *template_args,  # type: ignore[arg-type]
        )

        op_instances.append(new_instance)
    return op_instances


@lru_cache(None)
def gen_conv_ops_library() -> List[CKGroupedConvFwdOp]:
    """
    Parse the Grouped Convolution Forward instances
    defined in the Composable Kernel library folder.
    """
    ck_library_dir = _ck_conv_instances_path()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
            ck_library_dir,
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
    conv_specs = [
        "ConvolutionForwardSpecialization::Default",
        "ConvolutionForwardSpecialization::Filter1x1Pad0",
        "ConvolutionForwardSpecialization::Filter1x1Stride1Pad0",
        "ConvolutionForwardSpecialization::OddC",
    ]

    # substitute templated args by looping through their domains
    substitute_instances = []
    for instance in op_instances:
        sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
        sub_spec = instance.conv_forward_specialization == "ConvSpec"
        schedulers_range = (
            schedulers if sub_scheduler else [instance.block_gemm_pipeline_scheduler]
        )
        spec_range = conv_specs if sub_spec else [instance.conv_forward_specialization]
        for scheduler in schedulers_range:
            for spec in spec_range:
                for channels_last in [True, False]:
                    if channels_last:
                        a_layout = "NHWGC"
                        e_layout = "NHWGK"
                    else:
                        a_layout = "NGCHW"
                        e_layout = "NGKHW"
                    substitute_instances.append(
                        replace(
                            instance,
                            block_gemm_pipeline_scheduler=scheduler,
                            conv_forward_specialization=spec,
                            gemm_specialization="GemmSpecialization::MNKPadding",
                            n_dim_spatial=2,
                            a_layout=a_layout,
                            b_layout="GKYXC",
                            e_layout=e_layout,
                        )
                    )

    return substitute_instances


if __name__ == "__main__":
    print(gen_conv_ops_library())
