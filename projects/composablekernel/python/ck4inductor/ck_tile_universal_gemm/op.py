# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from dataclasses import asdict, dataclass


@dataclass
class CKTileGemmOperation:
    layout_a: str
    layout_b: str
    layout_c: str

    datatype_a: str
    datatype_b: str
    datatype_c: str

    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    m_is_padded: str
    n_is_padded: str
    k_is_padded: str

    pipeline: str
    scheduler: str
    epilogue: str

    def layout_repr(self):
        return f"{self.layout_a[0]}{self.layout_b[0]}{self.layout_c[0]}"

    def dtype_repr(self):
        return f"{self.datatype_a}{self.datatype_b}{self.datatype_c}"

    def tile_sizes(self):
        return "_".join(
            [
                f"{self.tile_m}{self.tile_n}{self.tile_k}",
                f"{self.warp_m}{self.warp_n}{self.warp_k}",
                f"{self.warp_tile_m}{self.warp_tile_n}{self.warp_tile_k}",
            ]
        )

    def name(self):
        return "ck_tile_gemm_universal_" + "_".join(
            [
                f"{self.layout_repr()}",
                f"{self.dtype_repr()}",
                f"{self.tile_sizes()}",
                f"{self.pipeline}",
                f"{self.scheduler}",
                f"{self.epilogue}",
            ]
        )

    def dict_items(self):
        return asdict(self).items()
