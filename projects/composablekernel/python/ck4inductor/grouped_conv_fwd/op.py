# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class CKGroupedConvFwdOp:
    n_dim_spatial: int
    a_layout: str
    b_layout: str
    ds_layout: Tuple[str]
    e_layout: str
    a_element_dtype: str
    b_element_dtype: str
    acc_dtype: str
    c_shuffle_dtype: str
    ds_element_dtype: Tuple[str]
    e_element_dtype: str
    a_elementwise_op: str
    b_elementwise_op: str
    cde_elementwise_op: str
    conv_forward_specialization: str
    gemm_specialization: str

    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    ak1: int
    bk1: int
    m_per_xdl: int
    n_per_xdl: int
    m_xdl_per_wave: int
    n_xdl_per_wave: int
    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[int, int, int]
    a_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    a_block_transfer_src_access_order: Tuple[int, int, int]
    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool

    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[int, int, int]
    b_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    b_block_transfer_src_access_order: Tuple[int, int, int]

    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool

    c_shuffle_m_xdl_per_wave_per_shuffle: int
    c_shuffle_n_xdl_per_wave_per_shuffle: int
    cde_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block: Tuple[  # noqa
        int,
        int,
        int,
        int,
    ]
    cde_block_transfer_scalar_per_vector_n_per_block: int
    block_gemm_pipeline_scheduler: str
    block_gemm_pipeline_version: str

    a_compute_dtype: Optional[str] = None
    b_compute_dtype: Optional[str] = None

    direct_load: Optional[bool] = None

    def name(self):
        # cpp alias for template instance
        return (
            f"ck_device_grouped_convolution_fwd_multiple_abd_xdl_c_shuffle_v3_"
            f"{self.key_name()}"
        )

    def key_name(self):
        # TBD; must be unique per instance. Intended to use as dict key
        return "_".join(
            [
                "K"
                + field_name.replace("_", "").lower()
                + "V"
                + (
                    "x".join(map(str, iter(field_value)))
                    if isinstance(field_value, tuple)
                    else str(field_value).replace(":", "")
                )
                for field_name, field_value in self.dict_items()
            ]
        )

    def dict_items(self):
        return asdict(self).items()
