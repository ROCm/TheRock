# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
import copy
import fnmatch
import itertools
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from codegen.arch import ArchTrait, get_factories_for_targets
from codegen.cmake_config import GEN_DIR
from codegen.cpp_symbol_map import (
    FWD_DTYPE_MAP,
    BOOL_MAP,
    ROPE_MAP,
    LAYOUT_MAP,
    ROPE_CHECK_MAP,
)
from codegen.utils import check_duplicates_and_paddings, if_, indent, update_file

from codegen.ops.fmha_fwd import (
    FMHA_FWD_KERNEL_HEADER,
    FMHA_FWD_API_PER_ARCH,
    FMHA_FWD_API_PER_DTYPE,
    FMHA_FWD_API_PER_HDIM_CASE,
)


FMHA_FWD_APPENDKV_KERNEL_BODY = """
#include <iostream>

#if !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})

using fmha_dtype_{F_idx} = {F_dtype};

using fmha_trait_{F_idx} = ck_tile::TileFmhaFwdAppendKVTraits<{F_spad},
                                                    {F_skpad},
                                                    {F_dpad},
                                                    {F_dvpad},
                                                    {F_occupancy}>;

using fmha_pipeline_problem_{F_idx} = ck_tile::BlockFmhaFwdAppendKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    {F_bs},
    {F_bsk},
    {F_bd},
    {F_bdv},
    {F_vlayout},
    {F_rope},
    {F_pagedkv},
    fmha_trait_{F_idx}>;

using fmha_pipeline_{F_idx} = ck_tile::BlockFmhaFwdAppendKVPipeline<
    fmha_pipeline_problem_{F_idx}>;

using fmha_kernel_{F_idx} = ck_tile::FmhaFwdAppendKVKernel<fmha_pipeline_{F_idx}>;

using trait_{F_idx} = fmha_fwd_appendkv_traits_<{F_hdim}, {F_dtype}, {F_bs}, {F_bsk}, {F_bd}, {F_bdv}, {F_vlayout},
                        {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_rope}, {F_pagedkv}>;

template<>
float fmha_fwd_appendkv_<trait_{F_idx}, {F_arch.tag}>(const ck_tile::stream_config& s, fmha_fwd_appendkv_args a)
{{
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids] = fmha_fwd_appendkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu, {F_arch.tag}>(k_{{}}, grids, blocks, 0, kargs));
}}

#endif // !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})
"""

FMHA_FWD_APPENDKV_API_FILENAME = "fmha_fwd_appendkv_api.cpp"
FMHA_FWD_APPENDKV_API = """
float fmha_fwd_appendkv(fmha_fwd_appendkv_traits t, fmha_fwd_appendkv_args a, const ck_tile::stream_config& s) {{
    float r = -1;

    [[maybe_unused]] const std::string device_name = ck_tile::get_device_name();

{F_dispatch}
    return r;
}}
"""

FMHA_FWD_APPENDKV_API_INNER_DISPATCH = """{F_if}((t.is_v_rowmajor == {F_vlayout}) &&
        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck}) && (t.rope_type == {F_rope_check}) &&
        ((a.block_table_ptr != nullptr) == {F_pagedkv})) {{
    using trait_ = fmha_fwd_appendkv_traits_<{F_hdim}, {F_dtype}, {F_bs}, {F_bsk}, {F_bd}, {F_bdv}, {F_vlayout}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_rope}, {F_pagedkv}>;
    return fmha_fwd_appendkv_<trait_, {F_arch.tag}>(s, a);
}}
"""


@dataclass
class FmhaFwdAppendKVApiTrait:
    arch: ArchTrait
    # sync with fmha_fwd_appendkv_traits, to generate fallback calls
    hdim: str
    dtype: str  # data type
    bs: int  # tile size along q seqlen
    bsk: int  # tile size along k seqlen
    bd: int  # tile size along qk gemm unroll
    bdv: int  # tile size along kv gemm unroll
    vlayout: str
    spad: str
    skpad: str
    dpad: str
    dvpad: str
    rope: str  # key from ROPE_MAP
    pagedkv: str

    @property
    def name(self) -> str:
        return (
            f"{self.hdim}-{self.dtype}-{self.bs}-{self.bsk}-{self.bd}-{self.bdv}-{self.vlayout}-"
            + f"{self.spad}-{self.skpad}-{self.dpad}-{self.dvpad}-{self.rope}-{self.pagedkv}"
        )

    @property
    def scheck(self) -> str:
        if self.spad == "t":
            return f"true /*a.seqlen_q % {self.bs} != 0*/"
        else:
            return f"a.seqlen_q % {self.bs} == 0"

    @property
    def skcheck(self) -> str:
        # we do not check all the values in a.seqlen_k_ptr
        return "true"

    @property
    def dcheck(self) -> str:
        if self.dpad == "t":
            return f"true /*a.hdim_q % {self.bd} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
        else:
            return f"a.hdim_q % {self.bd} == 0"

    @property
    def dvcheck(self) -> str:
        if self.dvpad == "t":
            return f"true /*a.hdim_v % {self.bdv} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
        else:
            return f"a.hdim_v % {self.bdv} == 0"


@dataclass
class FmhaFwdAppendKVPipeline:
    F_vlayout: str  # row/col
    F_spad: str  # true/false
    F_skpad: str  #
    F_dpad: str  #
    F_dvpad: str  #
    F_rope: str  # key from ROPE_MAP
    F_pagedkv: str  # t/f

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ""
            if self.F_spad == "t":
                n += "s"
            if self.F_skpad == "t":
                n += "sk"
            if self.F_dpad == "t":
                n += "d"
            if self.F_dvpad == "t":
                n += "dv"
            if n != "":
                n = "p" + n
            return n

        pn = pad_name()
        n = f"v{self.F_vlayout[0]}"
        if pn != "":
            n += f"_{pn}"
        if self.F_rope != "no":
            n += f"_{self.F_rope}"
        if self.F_pagedkv == "t":
            n += "_pagedkv"
        return n


class FmhaFwdAppendKVApiPool:
    def __init__(self, mask_impl):
        self.pool = OrderedDict()
        self.mask_impl = mask_impl

    def register_traits(self, trait: FmhaFwdAppendKVApiTrait) -> None:
        hdim = trait.hdim
        ts = (
            self.pool.setdefault(trait.arch, OrderedDict())
            .setdefault(trait.dtype, OrderedDict())
            .setdefault(hdim, [])
        )
        check_duplicates_and_paddings(ts, trait)
        ts.append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_arch = str()
        for i_arch, (arch, pool_by_arch) in enumerate(self.pool.items()):
            per_dtypes = str()
            for i_dtype, (dtype, pool_by_dtype) in enumerate(pool_by_arch.items()):
                per_hdim_case = str()
                for i_hdim, (hdim, pool_by_hdim) in enumerate(pool_by_dtype.items()):
                    inners = str()
                    for i_trait, trait in enumerate(pool_by_hdim):
                        inners += FMHA_FWD_APPENDKV_API_INNER_DISPATCH.format(
                            F_if=if_(i_trait),
                            F_arch=arch,
                            F_vlayout=LAYOUT_MAP[trait.vlayout],
                            F_scheck=trait.scheck,
                            F_skcheck=trait.skcheck,
                            F_dcheck=trait.dcheck,
                            F_dvcheck=trait.dvcheck,
                            F_rope_check=ROPE_CHECK_MAP[trait.rope],
                            F_pagedkv=BOOL_MAP[trait.pagedkv],
                            F_spad=BOOL_MAP[trait.spad],
                            F_skpad=BOOL_MAP[trait.skpad],
                            F_dpad=BOOL_MAP[trait.dpad],
                            F_dvpad=BOOL_MAP[trait.dvpad],
                            F_rope=ROPE_MAP[trait.rope],
                            F_bs=trait.bs,
                            F_bsk=trait.bsk,
                            F_bd=trait.bd,
                            F_bdv=trait.bdv,
                            F_hdim=hdim,
                            F_dtype=FWD_DTYPE_MAP[dtype],
                        )
                    per_hdim_case += FMHA_FWD_API_PER_HDIM_CASE.format(
                        F_if=if_(i_hdim),
                        F_hdim=hdim,
                        F_hdim_v=hdim,
                        F_inner_dispatch=indent(inners),
                    )
                per_dtypes += FMHA_FWD_API_PER_DTYPE.format(
                    F_if=if_(i_dtype), F_dtype=dtype, F_hdim_case=indent(per_hdim_case)
                )
            per_arch += FMHA_FWD_API_PER_ARCH.format(
                F_if=if_(i_arch),
                F_arch=arch,
                F_dtype_case=indent(per_dtypes),
            )
        if not per_arch:
            # empty string we add some ignore to suppress warning in api
            per_arch = "(void)t; (void)s; (void)a;"
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_APPENDKV_API.format(
            F_dispatch=indent(per_arch)
        )


@dataclass
class FmhaFwdAppendKVTileSize:
    F_bs: int  # tile size along q seqlen
    F_bsk: int  # tile size along k seqlen
    F_bd: int  # tile size along qk gemm unroll
    F_bdv: int  # tile size along kv gemm unroll
    F_occupancy: int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy

    @property
    def name(self) -> str:
        return f"b{self.F_bs}x{self.F_bsk}x{self.F_bd}x{self.F_bdv}" + (
            "" if self.F_occupancy == -1 else f"_o{self.F_occupancy}"
        )


@dataclass
class FmhaFwdAppendKVKernel:
    F_arch: ArchTrait
    F_idx: int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim: int  # hdim
    F_dtype: str  # data type
    F_tile: FmhaFwdAppendKVTileSize
    F_pipeline: FmhaFwdAppendKVPipeline
    mask_impl: str

    @property
    def template(self) -> str:
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_APPENDKV_KERNEL_BODY.format(
            F_idx=self.F_idx,
            F_arch=self.F_arch,
            F_hdim=self.F_hdim,
            F_dtype=FWD_DTYPE_MAP[self.F_dtype],
            F_bs=self.F_tile.F_bs,
            F_bsk=self.F_tile.F_bsk,
            F_bd=self.F_tile.F_bd,
            F_bdv=self.F_tile.F_bdv,
            F_vlayout=LAYOUT_MAP[self.F_pipeline.F_vlayout],
            F_spad=BOOL_MAP[self.F_pipeline.F_spad],
            F_skpad=BOOL_MAP[self.F_pipeline.F_skpad],
            F_dpad=BOOL_MAP[self.F_pipeline.F_dpad],
            F_dvpad=BOOL_MAP[self.F_pipeline.F_dvpad],
            F_rope=ROPE_MAP[self.F_pipeline.F_rope],
            F_pagedkv=BOOL_MAP[self.F_pipeline.F_pagedkv],
            F_occupancy=self.F_tile.F_occupancy,
        )

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return (
            f"fmha_fwd_appendkv_d{self.F_hdim}_{self.F_dtype}_"
            + self.F_tile.name
            + "_"
            + self.F_pipeline.name
        )

    @property
    def filename(self) -> str:
        return f"{self.name}{self.F_arch.filename_suffix}.cpp"

    def api_trait(self) -> FmhaFwdAppendKVApiTrait:
        return FmhaFwdAppendKVApiTrait(
            arch=self.F_arch,
            hdim=str(self.F_hdim),
            dtype=self.F_dtype,
            bs=self.F_tile.F_bs,
            bsk=self.F_tile.F_bsk,
            bd=self.F_tile.F_bd,
            bdv=self.F_tile.F_bdv,
            vlayout=self.F_pipeline.F_vlayout,
            spad=self.F_pipeline.F_spad,
            skpad=self.F_pipeline.F_skpad,
            dpad=self.F_pipeline.F_dpad,
            dvpad=self.F_pipeline.F_dvpad,
            rope=self.F_pipeline.F_rope,
            pagedkv=self.F_pipeline.F_pagedkv,
        )


class KernelComponentFactoryBase:
    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                "32": FmhaFwdAppendKVTileSize(64, 64, 32, 32, -1),
                "64": FmhaFwdAppendKVTileSize(64, 64, 64, 64, -1),
                "128": FmhaFwdAppendKVTileSize(64, 64, 128, 128, -1),
                "256": FmhaFwdAppendKVTileSize(64, 64, 256, 256, -1),
            }
        elif dtype in ["fp8", "bf8"]:
            return {
                "64": FmhaFwdAppendKVTileSize(64, 64, 64, 64, -1),
                "128": FmhaFwdAppendKVTileSize(64, 64, 128, 128, -1),
                "256": FmhaFwdAppendKVTileSize(64, 64, 256, 256, -1),
            }
        else:
            return None

    @staticmethod
    def get_pipelines(dtype, hdim) -> List[FmhaFwdAppendKVPipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr pipeline, let 't' padding to appear later!!
        # TODO: how to design this more generic?
        pipelines = []
        if dtype in ["fp16", "bf16"]:
            # NOTICE: it will be very complicated if we consider all the hdim_q padding cases while
            #         applying rotary embedding, so I just use 't' in inter/half pipelines
            for vlayout, pagedkv in itertools.product(["row"], ["t", "f"]):
                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "f", "t", "f", "f", "no", pagedkv))  # fmt: skip
                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "t", "t", "t", "t", "no", pagedkv))  # fmt: skip

                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "f", "t", "t", "f", "inter", pagedkv))  # fmt: skip
                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "t", "t", "t", "t", "inter", pagedkv))  # fmt: skip

                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "f", "t", "t", "f", "half", pagedkv))  # fmt: skip
                pipelines.append(FmhaFwdAppendKVPipeline(vlayout, "t", "t", "t", "t", "half", pagedkv))  # fmt: skip
        elif dtype in ["fp8", "bf8"]:
            # rope/paged-kv is not supported
            pipelines.append(FmhaFwdAppendKVPipeline("row", "t", "t", "t", "t", "no", "f"))  # fmt: skip
        elif dtype in ["fp8fp16", "fp8bf16"]:
            # TODO
            None
        else:
            assert False
        return pipelines


class KernelComponentFactoryGfx9(KernelComponentFactoryBase):
    arch = ArchTrait("gfx9")


class KernelComponentFactoryGfx12(KernelComponentFactoryBase):
    arch = ArchTrait("gfx12")


def get_factory(target: str):
    # Place more specific architectures first

    if target.startswith("gfx9"):
        return KernelComponentFactoryGfx9

    if target.startswith("gfx12"):
        return KernelComponentFactoryGfx12

    raise Exception(f"Unsupported device target {target}")


def get_fwd_appendkv_blobs(
    targets: List[str], kernel_filter: Optional[str], receipt, mask_impl, optdim_list
) -> Tuple[FmhaFwdAppendKVApiPool, List[FmhaFwdAppendKVKernel]]:
    gen = list()
    api_pool = FmhaFwdAppendKVApiPool(mask_impl)

    factories = get_factories_for_targets(targets, get_factory)

    for factory, dtype in itertools.product(factories, FWD_DTYPE_MAP.keys()):
        d = factory.get_hdim_tile_size_dict(dtype)
        if d is None:
            continue
        for hdim_str in d.keys():
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in factory.get_pipelines(dtype, hdim):
                k = FmhaFwdAppendKVKernel(
                    F_arch=factory.arch,
                    F_idx=0,
                    F_hdim=hdim,
                    F_dtype=dtype,
                    F_tile=tile,
                    F_pipeline=pipeline,
                    mask_impl=mask_impl,
                )
                if kernel_filter != "":
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                if optdim_list != [-1]:
                    if hdim not in optdim_list:
                        continue
                # 2 - Flash attention integration
                if receipt == 2:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    if not cond:
                        continue
                # PyTorch integration
                elif receipt == 4:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    if not cond:
                        continue

                # fp32 only
                if receipt == 800 or receipt == 801:
                    cond = dtype == "fp32"
                    if not cond:
                        continue

                api_pool.register_traits(k.api_trait())
                gen.append(k)

    return (api_pool, gen)


def write_single_kernel(kernel: FmhaFwdAppendKVKernel, autogen_dir: Path) -> None:
    update_file(autogen_dir / kernel.filename, kernel.template)


def write_fwd_appendkv_api(api_pool: FmhaFwdAppendKVApiPool, autogen_dir: Path) -> None:
    update_file(autogen_dir / FMHA_FWD_APPENDKV_API_FILENAME, api_pool.api)


def write_blobs(
    targets: List[str],
    output_dir: Path,
    kernel_filter: Optional[str],
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    api_pool, kernels = get_fwd_appendkv_blobs(
        targets, kernel_filter, receipt, mask_impl, optdim_list
    )
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    write_fwd_appendkv_api(api_pool, output_dir)


def list_blobs(
    targets: List[str],
    file_path: Path,
    kernel_filter: Optional[str],
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    with file_path.open("a") as f:
        _, kernels = get_fwd_appendkv_blobs(
            targets, kernel_filter, receipt, mask_impl, optdim_list
        )
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_APPENDKV_API_FILENAME) + "\n")
