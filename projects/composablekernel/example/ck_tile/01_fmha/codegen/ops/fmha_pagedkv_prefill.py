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
    LAYOUT_MAP,
    BIAS_CHECK_MAP,
    get_mask_check_map,
    MODE_MAP,
    get_mask_map,
    BIAS_MAP,
    FWD_DTYPE_MAP,
    BOOL_MAP,
    PIPELINE_ENUM_MAP,
)
from codegen.utils import check_duplicates_and_paddings, if_, indent, update_file

from codegen.ops.fmha_fwd import (
    DTYPE_BITS,
    K0_MAX_SUBMAX_MAP,
    FMHA_FWD_KERNEL_HEADER,
    FMHA_FWD_API_PER_ARCH,
    FMHA_FWD_API_PER_DTYPE,
    FMHA_FWD_API_PER_HDIM_CASE,
)


FMHA_FWD_PAGEDKV_PIPELINE_MAP = {
    "qr_pagedkv": "ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVS"
}

FMHA_FWD_KERNEL_BODY = """
#include <iostream>

#if !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})

using fmha_dtype_{F_idx} = {F_dtype};

using fmha_block_tile_{F_idx} = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}>;

using fmha_shape_{F_idx} = ck_tile::TileFmhaShape<fmha_block_tile_{F_idx},
                                      ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>,
                                      ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>,
                                      ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>,
                                      ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>,
                                      {F_vlayout}>;

using fmha_trait_{F_idx} = ck_tile::TileFmhaFwdPagedKVTraits<{F_spad},
                                                             {F_skpad},
                                                             {F_dpad},
                                                             {F_dvpad},
                                                             {F_logits},
                                                             {F_bias},
                                                             false,
                                                             {F_lse},      //lse
                                                             {F_pagedkv},  //pagedkv
                                                             {F_squant},
                                                             {F_occupancy},
                                                             {F_skip},
                                                             {F_sink}>;

using fmha_variant_{F_idx} = ck_tile::ComposedAttention<{F_logits} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;

using fmha_mask_{F_idx} = {F_mask};

using fmha_pipeline_problem_{F_idx} = ck_tile::BlockFmhaFwdPagedKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    fmha_shape_{F_idx},
    {F_mode},
    fmha_variant_{F_idx},
    fmha_mask_{F_idx},
    fmha_trait_{F_idx}>;

using fmha_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}>;

using fmha_epilogue_{F_idx} =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::ODataType,
                                           {F_spad}, {F_dvpad}>>;

using fmha_kernel_{F_idx} =
    ck_tile::FmhaFwdPagedKVKernel<fmha_pipeline_{F_idx}, fmha_epilogue_{F_idx}>;

using trait_{F_idx} = fmha_fwd_pagedkv_traits_<{F_hdim}, {F_dtype}, {F_mode},{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, {F_vlayout},
                        {F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_pagedkv}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_skip}, {F_sink}>;

template<>
float fmha_fwd_pagedkv_<trait_{F_idx}, {F_arch.tag}>(const ck_tile::stream_config& s, fmha_fwd_pagedkv_args a)
{{
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids] = fmha_fwd_pagedkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu, {F_arch.tag}>(k_{{}}, grids, blocks, 0, kargs));
}}

#endif // !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})
"""

FMHA_FWD_API_FILENAME = "fmha_fwd_pagedkv_api.cpp"
FMHA_FWD_API = """
float fmha_fwd_pagedkv(fmha_fwd_pagedkv_traits& t, fmha_fwd_pagedkv_args& a, const ck_tile::stream_config& s) {{
    float r = -1;

    [[maybe_unused]] const std::string device_name = ck_tile::get_device_name();

{F_dispatch}
    return r;
}}
"""

FMHA_FWD_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logits_soft_cap == {F_logits}) && ({F_mask_check}) && (t.bias_type == {F_bias_check}) && (t.has_lse == {F_lse})  && (t.use_pagedkv == {F_pagedkv}) && (t.do_fp8_static_quant == {F_squant}) && (t.skip_min_seqlen_q == {F_skip}) && (t.has_sink == {F_sink}) &&
        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck})) {{
    using trait_ = fmha_fwd_pagedkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, {F_vlayout}, {F_pipeline_enum}, {F_logits}, {F_mask}, {F_bias}, {F_lse}, {F_pagedkv}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_skip},{F_sink}>;
    return fmha_fwd_pagedkv_<trait_, {F_arch.tag}>(s, a);
}}
"""


@dataclass
class FmhaFwdApiTrait:
    arch: ArchTrait
    pipeline_tag: str
    # sync with fmha_fwd_traits<>, to generate fallback calls
    hdim: str
    dtype: str  # data type
    mode: str  # value from MODE_MAP
    bm0: int  # tile size along q seqlen (block size)
    bn0: int  # tile size along qk seqlen
    bk0: int  # tile size along qk gemm unroll
    bn1: int  # tile size along v head_dim
    bk1: int  # tile size along kv gemm unroll
    bk0max: int
    vlayout: str
    logits: str
    mask: str
    bias: str  #
    lse: str  #
    pagedkv: str
    squant: str  #
    spad: str
    skpad: str
    dpad: str
    dvpad: str
    skip: str
    sink: str

    @property
    def name(self) -> str:
        return (
            f"{self.hdim}-{self.dtype}-{self.mode}-{self.bm0}-{self.bn0}-{self.bk0}-{self.bn0}-{self.bk1}-{self.bk0max}-"
            + f"{self.vlayout}-{self.logits}-{self.mask}-{self.bias}-{self.lse}-{self.pagedkv}-{self.squant}-{self.spad}-{self.skpad}-{self.dpad}-{self.dvpad}-{self.skip}-{self.sink}"
        )

    @property
    def scheck(self) -> str:
        if self.mode == "group":
            return "true/*group mode spad always true*/"  # group mode only generate spad/skpad == true
        if self.pipeline_tag == "qr_async":
            if self.spad == "t":
                return "true"  # always support
            else:
                return "true"
        elif self.pipeline_tag in ["qr_pagedkv", "qs"]:
            if self.spad == "t":
                return f"true /*a.seqlen_q % {self.bm0} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.seqlen_q % {self.bm0} == 0"
        else:
            assert False

    @property
    def skcheck(self) -> str:
        if self.mode == "group":
            return "true/*group mode skpad always true*/"  # group mode only generate spad/skpad == true
        if self.pipeline_tag == "qr_async":
            if self.skpad == "t":
                return f"a.seqlen_k == 0 || a.seqlen_k % {self.bn0} != 0"
            else:
                return f"a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0"
        elif self.pipeline_tag in ["qr_pagedkv", "qs"]:
            if self.skpad == "t":
                return f"true /*a.seqlen_k_ptr != nullptr || a.seqlen_k % {self.bn0} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.seqlen_k_ptr == nullptr && a.seqlen_k % {self.bn0} == 0"
        else:
            assert False

    @property
    def dcheck(self) -> str:
        if self.pipeline_tag == "qr_async":
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dpad == "t":
                return f"a.hdim_q % {vec} == 0"
            else:
                assert False
        elif self.pipeline_tag in ["qr_pagedkv", "qs"]:
            bk0submax = K0_MAX_SUBMAX_MAP[self.bk0max]
            if self.dpad == "t":
                return f"true /*a.hdim_q % {bk0submax} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.hdim_q % {bk0submax} == 0"
        else:
            assert False

    @property
    def dvcheck(self) -> str:
        if self.pipeline_tag == "qr_async":
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dvpad == "t":
                return f"a.hdim_v % {vec} == 0"
            else:
                assert False
        elif self.pipeline_tag in ["qr_pagedkv", "qs"]:
            bk0submax = K0_MAX_SUBMAX_MAP[self.bk0max]
            if self.dvpad == "t":
                return f"true /*a.hdim_v % {bk0submax} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.hdim_v % {bk0submax} == 0"
        else:
            assert False


@dataclass
class FmhaFwdPipeline:
    tag: str

    F_vlayout: str  # row/col
    F_spad: str  # true/false
    F_skpad: str  #
    F_dpad: str  #
    F_dvpad: str  #
    F_logits: str  # t/f
    F_bias: str  # true/false
    F_lse: str  #
    F_pagedkv: str  #
    F_squant: str  #
    F_mask: str  # value from MASK_MAP
    F_skip: str  # true/false
    F_sink: str  # true/false

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
        n = f"{self.tag}_v{self.F_vlayout[0]}"
        if pn != "":
            n += f"_{pn}"
        else:
            n += "_npad"

        if self.F_logits == "t":
            n += "_logits"
        else:
            n += "_nlogits"

        if self.F_bias != "no":
            n += f"_{self.F_bias}"
        else:
            n += "_nbias"

        if self.F_mask[0:2] == "s_":
            if self.F_mask == "s_mask":
                n += "_mask"
            else:
                n += "_nmask"
        else:
            if self.F_mask != "no":
                n += f"_m{self.F_mask[0]}"
            else:
                n += "_nmask"

        if self.F_lse == "t":
            n += "_lse"
        else:
            n += "_nlse"

        if self.F_skip == "t":
            n += "_skip"
        else:
            n += "_nskip"

        if self.F_squant == "t":
            n += "_squant"
        else:
            n += "_nsquant"

        if self.F_pagedkv == "t":
            n += "_pagedkv"
        else:
            n += "_npagedkv"
        if self.F_sink == "t":
            n += "_sink"
        else:
            n += "_nsink"

        return n


class FmhaFwdApiPool:
    def __init__(self, mask_impl):
        self.pool = OrderedDict()
        self.mask_impl = mask_impl

    def register_traits(self, trait: FmhaFwdApiTrait) -> None:
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
                        inners += FMHA_FWD_API_INNER_DISPATCH.format(
                            F_if=if_(i_trait),
                            F_arch=arch,
                            F_mode=MODE_MAP[trait.mode],
                            F_vlayout=LAYOUT_MAP[trait.vlayout],
                            F_pipeline_enum=PIPELINE_ENUM_MAP[trait.pipeline_tag],
                            F_logits=BOOL_MAP[trait.logits],
                            F_mask=get_mask_map(self.mask_impl)[trait.mask],
                            F_mask_check=get_mask_check_map(self.mask_impl)[trait.mask],
                            F_bias_check=BIAS_CHECK_MAP[trait.bias],
                            F_bias=BIAS_MAP[trait.bias],
                            F_lse=BOOL_MAP[trait.lse],
                            F_pagedkv=BOOL_MAP[trait.pagedkv],
                            F_skip=BOOL_MAP[trait.skip],
                            F_sink=BOOL_MAP[trait.sink],
                            F_squant=BOOL_MAP[trait.squant],
                            F_scheck=trait.scheck,
                            F_skcheck=trait.skcheck,
                            F_dcheck=trait.dcheck,
                            F_dvcheck=trait.dvcheck,
                            F_spad=BOOL_MAP[trait.spad],
                            F_skpad=BOOL_MAP[trait.skpad],
                            F_dpad=BOOL_MAP[trait.dpad],
                            F_dvpad=BOOL_MAP[trait.dvpad],
                            F_bm0=trait.bm0,
                            F_bn0=trait.bn0,
                            F_bk0=trait.bk0,
                            F_bn1=trait.bn1,
                            F_bk1=trait.bk1,
                            F_bk0max=trait.bk0max,
                            F_hdim=hdim,
                            F_dtype=FWD_DTYPE_MAP[dtype],
                        )
                    per_hdim_case += FMHA_FWD_API_PER_HDIM_CASE.format(
                        F_if=if_(i_hdim),
                        F_hdim=hdim,
                        F_hdim_v=trait.bn1,
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
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_API.format(F_dispatch=indent(per_arch))


@dataclass
class FmhaFwdTileSize:
    F_bm0: int  # tile size along q seqlen (block size)
    F_bn0: int  # tile size along k seqlen
    F_bk0: int  # tile size along qk gemm unroll
    F_bn1: int  # tile size along v head_dim
    F_bk1: int  # tile size along kv gemm unroll
    F_bk0max: int  # total length of K0, used for pipeline that need load Q at once (or repeately load Q as a whole tile)
    F_rm0: int  # number of warps for gemm0 along q seqlen
    F_rn0: int  # number of warps for gemm0 along k seqlen
    F_rk0: int  # number of warps for gemm0 along head dim q (not used)
    F_rm1: int  # number of warps for gemm1 along q seqlen
    F_rn1: int  # number of warps for gemm1 along head dim v
    F_rk1: int  # number of warps for gemm1 along k seqlen (not used)
    F_wm0: int  # gemm0 warp size along m
    F_wn0: int  # gemm0 warp size along n
    F_wk0: int  # gemm0 warp size along k
    F_wm1: int  # gemm1 warp size along m
    F_wn1: int  # gemm1 warp size along n
    F_wk1: int  # gemm1 warp size along k
    F_occupancy: int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy

    @property
    def name(self) -> str:
        return (
            f"b{self.F_bm0}x{self.F_bn0}x{self.F_bk0}x{self.F_bn1}x{self.F_bk1}x{self.F_bk0max}"
            + f"_r{self.F_rm0}x{self.F_rn0}x{self.F_rk0}_r{self.F_rm1}x{self.F_rn1}x{self.F_rk1}"
            + f"_w{self.F_wm0}x{self.F_wn0}x{self.F_wk0}_w{self.F_wm1}x{self.F_wn1}x{self.F_wk1}"
            + ("" if self.F_occupancy == -1 else f"_o{self.F_occupancy}")
        )


@dataclass
class FmhaFwdKernel:
    F_arch: ArchTrait
    F_idx: int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim: int  # hdim
    F_dtype: str  # data type
    F_mode: str  # value from MODE_MAP
    F_tile: FmhaFwdTileSize
    F_pipeline: FmhaFwdPipeline
    mask_impl: str

    @property
    def template(self) -> str:
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_KERNEL_BODY.format(
            F_idx=self.F_idx,
            F_arch=self.F_arch,
            F_hdim=self.F_hdim,
            F_dtype=FWD_DTYPE_MAP[self.F_dtype],
            F_bm0=self.F_tile.F_bm0,
            F_bn0=self.F_tile.F_bn0,
            F_bk0=self.F_tile.F_bk0,
            F_bn1=self.F_tile.F_bn1,
            F_bk1=self.F_tile.F_bk1,
            F_bk0max=self.F_tile.F_bk0max,
            F_rm0=self.F_tile.F_rm0,
            F_rn0=self.F_tile.F_rn0,
            F_rk0=self.F_tile.F_rk0,
            F_rm1=self.F_tile.F_rm1,
            F_rn1=self.F_tile.F_rn1,
            F_rk1=self.F_tile.F_rk1,
            F_wm0=self.F_tile.F_wm0,
            F_wn0=self.F_tile.F_wn0,
            F_wk0=self.F_tile.F_wk0,
            F_wm1=self.F_tile.F_wm1,
            F_wn1=self.F_tile.F_wn1,
            F_wk1=self.F_tile.F_wk1,
            F_vlayout=LAYOUT_MAP[self.F_pipeline.F_vlayout],
            F_spad=BOOL_MAP[self.F_pipeline.F_spad],
            F_skpad=BOOL_MAP[self.F_pipeline.F_skpad],
            F_dpad=BOOL_MAP[self.F_pipeline.F_dpad],
            F_dvpad=BOOL_MAP[self.F_pipeline.F_dvpad],
            F_logits=BOOL_MAP[self.F_pipeline.F_logits],
            F_bias=BIAS_MAP[self.F_pipeline.F_bias],
            F_lse=BOOL_MAP[self.F_pipeline.F_lse],
            F_pagedkv=BOOL_MAP[self.F_pipeline.F_pagedkv],
            F_squant=BOOL_MAP[self.F_pipeline.F_squant],
            F_skip=BOOL_MAP[self.F_pipeline.F_skip],
            F_sink=BOOL_MAP[self.F_pipeline.F_sink],
            F_occupancy=self.F_tile.F_occupancy,
            F_pipeline_enum=PIPELINE_ENUM_MAP[self.F_pipeline.tag],
            F_mask=get_mask_map(self.mask_impl)[self.F_pipeline.F_mask],
            F_mode=MODE_MAP[self.F_mode],
            F_pipeline=FMHA_FWD_PAGEDKV_PIPELINE_MAP[self.F_pipeline.tag],
        )

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return (
            f"fmha_fwd_pagedkv_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_"
            + self.F_tile.name
            + "_"
            + self.F_pipeline.name
        )

    @property
    def filename(self) -> str:
        return f"{self.name}{self.F_arch.filename_suffix}.cpp"

    def api_trait(self) -> FmhaFwdApiTrait:
        return FmhaFwdApiTrait(
            arch=self.F_arch,
            pipeline_tag=self.F_pipeline.tag,
            hdim=str(self.F_hdim),
            dtype=self.F_dtype,
            mode=self.F_mode,
            bm0=self.F_tile.F_bm0,
            bn0=self.F_tile.F_bn0,
            bk0=self.F_tile.F_bk0,
            bn1=self.F_tile.F_bn1,
            bk1=self.F_tile.F_bk1,
            bk0max=self.F_tile.F_bk0max,
            vlayout=self.F_pipeline.F_vlayout,
            mask=self.F_pipeline.F_mask,
            logits=self.F_pipeline.F_logits,
            bias=self.F_pipeline.F_bias,
            lse=self.F_pipeline.F_lse,
            pagedkv=self.F_pipeline.F_pagedkv,
            squant=self.F_pipeline.F_squant,
            spad=self.F_pipeline.F_spad,
            skpad=self.F_pipeline.F_skpad,
            dpad=self.F_pipeline.F_dpad,
            dvpad=self.F_pipeline.F_dvpad,
            skip=self.F_pipeline.F_skip,
            sink=self.F_pipeline.F_sink,
        )


class KernelComponentFactoryBase:
    @staticmethod
    def get_pipelines(dtype, hdim, mask_impl) -> List[FmhaFwdPipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr_pagedkv pipeline, let "t" padding to appear later!!
        # TODO: how to design this more generic?
        squant = "t" if dtype == "fp8" else "f"
        pipelines = []
        if dtype in ["fp16", "bf16"]:
            for logits, mask, bias, pagedkv, skip, sink in itertools.product(
                ["t", "f"],
                get_mask_map(mask_impl).keys(),
                BIAS_MAP.keys(),
                ["t"],
                ["f"],
                ["t", "f"],
            ):
                pipelines.append(FmhaFwdPipeline("qr_pagedkv", "row", "t", "f", "f", "f", logits, bias, "f", pagedkv, squant, mask, skip, sink))  # fmt: skip
                pipelines.append(FmhaFwdPipeline("qr_pagedkv", "row", "t", "t", "f", "f", logits, bias, "f", pagedkv, squant, mask, skip, sink))  # fmt: skip
        elif dtype in ["fp8", "bf8"]:
            # no need lse/dropout kernels
            for logits, mask, bias in itertools.product(
                ["t", "f"], get_mask_map(mask_impl).keys(), BIAS_MAP.keys()
            ):
                pipelines.append(FmhaFwdPipeline("qr_pagedkv", "row", "f", "f", "f", "f", logits, bias, "f", "t", squant, mask, "f", "f"))  # fmt: skip
                pipelines.append(FmhaFwdPipeline("qr_pagedkv", "row", "t", "t", "f", "f", logits, bias, "f", "t", squant, mask, "f", "f"))  # fmt: skip
        elif dtype in ["fp8fp16", "fp8bf16"]:
            pass  # TODO
        else:
            assert False
        return pipelines


class KernelComponentFactoryGfx9(KernelComponentFactoryBase):
    arch = ArchTrait("gfx9")

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                # "32":  FmhaFwdTileSize(128,  64, 16,  32, 32,  32,  2, 1, 1,  2, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
                # "64":  FmhaFwdTileSize(128,  64, 32,  64, 32,  64,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
                # "96":  FmhaFwdTileSize(128, 128, 32, 128, 32,  96,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
                "128": FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
                # "192": FmhaFwdTileSize(128, 128, 32, 128, 32, 192,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
                # "256": FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,  -1),
            }  # fmt: skip
        elif dtype in ["fp8", "bf8"]:
            return {
                "64":  FmhaFwdTileSize(128,  64, 32, 64,  32,  64,  2, 1, 1,  2, 1, 1,  32, 32, 32,  32, 32, 32,  -1),
                "128": FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1,  4, 1, 1,  32, 32, 32,  32, 32, 32,  -1),
                "256": FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1,  4, 1, 1,  32, 32, 32,  32, 32, 32,  -1),
            }  # fmt: skip
        else:
            return None


class KernelComponentFactoryGfx12(KernelComponentFactoryBase):
    arch = ArchTrait("gfx12")

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                #                      bm0, bn0, bk0, bn1, bk1,
              # "32":  FmhaFwdTileSize( 64,  64,  16,  32,  32,   32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
              # "64":  FmhaFwdTileSize( 64,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
                "128": FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
              # "192": FmhaFwdTileSize( 64,  64,  32, 128,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
              # "256": FmhaFwdTileSize( 64,  64,  32, 256,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
            }  # fmt: skip
        elif dtype in ["fp8", "bf8"]:
            return {
                #                      bm0, bn0, bk0, bn1, bk1,
                "64":  FmhaFwdTileSize(128,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
                "128": FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
                "256": FmhaFwdTileSize( 64,  32,  32, 256,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
            }  # fmt: skip
        else:
            return None


def get_factory(target: str):
    # Place more specific architectures first

    if target.startswith("gfx9"):
        return KernelComponentFactoryGfx9

    if target.startswith("gfx12"):
        return KernelComponentFactoryGfx12

    raise Exception(f"Unsupported device target {target}")


def get_fwd_blobs(
    targets: List[str], kernel_filter: Optional[str], receipt, optdim_list, mask_impl
) -> Tuple[FmhaFwdApiPool, List[FmhaFwdKernel]]:
    gen = list()
    api_pool = FmhaFwdApiPool(mask_impl)

    factories = get_factories_for_targets(targets, get_factory)

    for factory, dtype in itertools.product(factories, FWD_DTYPE_MAP.keys()):
        d = factory.get_hdim_tile_size_dict(dtype)
        if d is None:
            continue
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in factory.get_pipelines(dtype, hdim, mask_impl):
                # if pipeline.F_pagedkv == "f":
                #     continue
                if mode == "group":
                    if pipeline.F_spad != "t" or pipeline.F_skpad != "t":
                        # in group mode, spad/skpad must be true, since we can't predict if seqlen of current batch need pad or not
                        continue
                if hdim == 192 and tile.F_bn1 == 128:
                    # NOTE: this is used to speedup deepseek prefill case, we don't gen training
                    if pipeline.F_bias != "no" or pipeline.F_lse == "t":
                        continue
                # logits_soft_cap is only allowed if no bias
                if not (
                    (pipeline.F_logits == "t" and pipeline.F_bias == "no")
                    or pipeline.F_logits == "f"
                ):
                    continue
                k = FmhaFwdKernel(
                    F_arch=factory.arch,
                    F_idx=0,
                    F_hdim=hdim,
                    F_dtype=dtype,
                    F_mode=mode,
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
                if receipt in (2, 3):
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_bias in ["no", "alibi"]
                    cond &= pipeline.F_squant == "f"
                    cond &= pipeline.F_skip == "f"
                    cond &= pipeline.F_sink == "f"
                    if not cond:
                        continue
                # PyTorch integration
                elif receipt == 4:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_bias in ["no", "bias"]
                    cond &= pipeline.F_squant == "f"
                    cond &= pipeline.F_skip == "f"
                    cond &= pipeline.F_sink == "f"
                    if not cond:
                        continue
                # Aiter(mha_fwd) integration
                elif receipt == 100:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= mode == "batch"
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_squant == "f"
                    if not cond:
                        continue
                # Aiter(mha_varlen_fwd) integration
                elif receipt == 200:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= mode == "group"
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_squant == "f"
                    if not cond:
                        continue
                # aiter::mha_fwd C++ api integration
                elif receipt == 600:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_squant == "f"
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


def write_single_fwd_kernel(kernel: FmhaFwdKernel, autogen_dir: Path) -> None:
    update_file(autogen_dir / kernel.filename, kernel.template)


def write_fwd_api(api_pool: FmhaFwdApiPool, autogen_dir: Path) -> None:
    update_file(autogen_dir / FMHA_FWD_API_FILENAME, api_pool.api)


def write_blobs(
    targets: List[str],
    output_dir: Path,
    kernel_filter: str,
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    api_pool, kernels = get_fwd_blobs(
        targets, kernel_filter, receipt, optdim_list, mask_impl
    )
    for kernel in kernels:
        write_single_fwd_kernel(kernel, output_dir)
    write_fwd_api(api_pool, output_dir)


def list_blobs(
    targets: List[str],
    file_path: Path,
    kernel_filter: str,
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    with file_path.open("a") as f:
        _, kernels = get_fwd_blobs(
            targets, kernel_filter, receipt, optdim_list, mask_impl
        )
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_API_FILENAME) + "\n")
