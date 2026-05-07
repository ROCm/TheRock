# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
import argparse
from enum import IntEnum
from pathlib import Path
import sys
from typing import List, Any
import functools
import itertools
import copy
from dataclasses import dataclass


def get_if_str(idx, total, lase_else=True):
    if idx == 0:
        return "if"
    elif idx < total - 1:
        return "else if"
    else:
        if lase_else:
            return "else"
        else:
            return "else if"


XBIAS_ENUM_STR_MAP = ["no", "xbias"]  # pre-norm add bias

FUSED_ADD_ENUM_STR_MAP = [
    "no",
    "pras",  # pre-norm
    "pra",
]  # post-norm

FUSED_FUSED_SWEEP_STR_MAP = ["no", "dquant"]

DATA_TYPE_MAP = {
    "fp32": "float",
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
    "int8": "ck_tile::int8_t",
    "fp8": "ck_tile::fp8_t",
}


def BOOL_MAP(b_) -> str:
    if b_:
        return "true"
    else:
        return "false"


class layernorm_fwd_codegen:
    API_TRAITS_DEFINE = """
// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kWelford_,
          bool kTwoPass_,
          ck_tile::index_t kXbias_ = 0,
          ck_tile::index_t kFusedAdd_ = 0,
          ck_tile::index_t kFusedQuant_ = 0>
struct layernorm2d_fwd_traits_
{
    using XDataType = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType = ck_tile::remove_cvref_t<YDataType_>;
    using SmoothScaleDataType = ck_tile::remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType = ck_tile::remove_cvref_t<YScaleDataType_>;

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;
    using ThreadPerBlock = ck_tile::sequence<ThreadPerBlock_M_, ThreadPerBlock_N_>;
    
    using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kFastFDiv       = kFastFDiv_;
    static constexpr bool kWelford        = kWelford_;
    static constexpr bool kTwoPass        = kTwoPass_;
    static constexpr ck_tile::index_t kXbias = kXbias_;
    static constexpr ck_tile::index_t kFusedAdd = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant = kFusedQuant_;
};

template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kWelford_,
          bool kTwoPass_,
          int  kXbias_,
          int  kFusedAdd_,
          int  kFusedQuant_>
using traits_ = layernorm2d_fwd_traits_<XDataType_,
                                       YDataType_,
                                       SmoothScaleDataType_,
                                       YScaleDataType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_N_,
                                       kPadN_,
                                       kSaveMeanInvStd_,
                                       kFastFDiv_,
                                       kWelford_,
                                       kTwoPass_,
                                       kXbias_,
                                       kFusedAdd_,
                                       kFusedQuant_>;
"""
    API_COMMON_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = layernorm2d_fwd_args;

{F_traits_define}

template <typename Traits_>
float layernorm2d_fwd_(const S& s, A a)
{{
    using XDataType = typename Traits_::XDataType;
    using YDataType = typename Traits_::YDataType;
    using SmoothScaleDataType = typename Traits_::SmoothScaleDataType;
    using YScaleDataType = typename Traits_::YScaleDataType;
    using ComputeDataType = typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType;

    using PipelineTraits = ck_tile::Layernorm2dFwdTraits<Traits_::kPadN,
        Traits_::kSaveMeanInvStd,
        Traits_::kFastFDiv,
        Traits_::kWelford,
        Traits_::kTwoPass,
        static_cast<ck_tile::Layernorm2dXBiasEnum>(Traits_::kXbias),
        static_cast<ck_tile::Layernorm2dFusedAddEnum>(Traits_::kFusedAdd),
        static_cast<ck_tile::Layernorm2dFusedQuantEnum>(Traits_::kFusedQuant)>;
    using PipelineProblem = ck_tile::Layernorm2dFwdPipelineProblem<
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::XDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::XBiasDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::GammaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::BetaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::MeanDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::InvStdDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::SmoothScaleDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YScaleDataType,
        typename Traits_::Shape,
        PipelineTraits>;

    using OnePassPipeline = ck_tile::Layernorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using Default2DEpilogueProblem = ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, Traits_::kPadN, true>;
    using Default2DEpilogue = ck_tile::Default2DEpilogue<Default2DEpilogueProblem>;

    static constexpr bool UseSmoothInputScale = Traits_::kFusedQuant == 1;
    static constexpr bool UseRawStore = sizeof(YDataType) == 4;
    using DynamicQuantEpilogueProblem = ck_tile::DynamicQuantEpilogueProblem<ComputeDataType, SmoothScaleDataType, YScaleDataType, YDataType, typename Traits_::Shape,
            ck_tile::DynamicQuantEpilogueTraits<false, Traits_::kPadN, UseSmoothInputScale, UseRawStore,  true/*max3*/>>;

    using DynamicQuantEpilogue = ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem>;

    using Epilogue = std::conditional_t<Traits_::kFusedQuant == 1, DynamicQuantEpilogue,  Default2DEpilogue>;

    using Kernel = ck_tile::Layernorm2dFwd<Pipeline, Epilogue>;

    const dim3 grids                       = Kernel::GridSize(a);
    const dim3 blocks                      = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
}}

"""

    API_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"

{F_traits_define}

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename Traits_>
float layernorm2d_fwd_(const ck_tile::stream_config& s, layernorm2d_fwd_args a);

float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{{
    float r = -1;
{F_dispatch}
    return r;
}}

"""

    API_PER_DTYPE = """    {F_if}(t.prec_i == \"{F_i_type}\" && t.prec_o == \"{F_o_type}\"){{
{F_per_n_case}
    }}
"""
    API_PER_N_CASE = """        {F_if} {F_N_COND} {{
{F_inner_dispatch}
        }}
"""
    API_INNER_CASE = """            {F_if} {F_VEC_COND}
                r={F_instance_func}(s, a);
"""

    INSTANCE_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_api_common.hpp"

// clang-format off
//                                      prec_i           prec_o           prec_sy           rm  rn  tm    tn  vn  pd     mv    rpcf   welford   2p   xbias   add  sweep
{F_instance_def}
// clang-format on

"""

    def __init__(self, working_path, kernel_filter):
        self.working_path = working_path
        self.kernel_filter = kernel_filter

    class k_xbias_enum(IntEnum):
        F_NO_XBIAS = 0
        F_ADD_XBIAS = 1

    class k_fuesd_add_enum(IntEnum):
        F_NO_ADD = 0
        F_PRE_ADD = 1
        F_PRE_ADD_STORE_RESIDUAL = 2

    class k_fused_sweep_enum(IntEnum):
        F_NO_SWEEP = 0
        F_RENORM = 1
        F_DYNAMIC_QUANT = 2

    @dataclass
    class k_traits:
        F_kPadN: bool
        F_kSaveMeanInvStd: bool
        F_kTwoPass: bool
        F_kXbias: Any  #: layernorm_fwd_codegen.k_bias_enum
        F_kFusedAdd: Any  #: layernorm_fwd_codegen.k_fuesd_add_enum
        F_kFusedQuant: Any  #: layernorm_fwd_codegen.k_fused_sweep_enum

    @dataclass
    class k_shape:
        F_BlockTile: List[int]
        F_WarpPerBlock: List[int]
        F_WarpTile: List[int]
        F_Vector_: List[int]

        @property
        def F_BlockSize(self) -> int:
            return functools.reduce(lambda a, b: a * b, self.F_WarpTile)

    @dataclass
    class k_problem:
        F_XDataType: str
        F_XBiasDataType: str
        F_GammaDataType: str
        F_BetaDataType: str
        F_ComputeDataType: str
        F_YDataType: str
        F_MeanDataType: str
        F_InvStdDataType: str
        F_BlockShape: str
        F_Traits: Any  # k_traits

    @dataclass
    class k_pipeline_one_pass:
        F_Problem: Any  # k_problem

    @dataclass
    class k_pipeline_two_pass:
        F_Problem: Any  # k_problem

    @dataclass
    class default_2d_epilogue_problem:
        F_AccDataType: str
        F_ODataType: str
        F_kPadM: bool
        F_kPadN: bool

    @dataclass
    class default_2d_epilogue:
        F_problem: Any

    @dataclass
    class k_kernel:
        F_pipeline: Any
        F_epilogue: Any

    @dataclass
    class h_traits:
        F_XDataType: str
        F_YDataType: str
        F_SmoothScaleDataType: str
        F_YScaleDataType: str
        F_Repeat_M: int
        F_Repeat_N: int
        F_ThreadPerBlock_M: int
        F_ThreadPerBlock_N: int
        F_Vector_N: int
        F_kPadN: bool
        F_kSaveMeanInvStd_: bool
        F_kFastFDiv_: bool
        F_kWelford_: bool
        F_kTwoPass_: bool
        F_kXbias_: int
        F_kFusedAdd: int
        F_kFusedQuant: int

        @property
        def trait_name(self) -> str:
            t_ = f"{DATA_TYPE_MAP[self.F_XDataType]}, {DATA_TYPE_MAP[self.F_YDataType]}, {DATA_TYPE_MAP[self.F_SmoothScaleDataType]}, {DATA_TYPE_MAP[self.F_YScaleDataType]}, {self.F_Repeat_M:2}, {self.F_Repeat_N:2}, {self.F_ThreadPerBlock_M:2}, {self.F_ThreadPerBlock_N:4}"
            t_ += f", {self.F_Vector_N:2}, {BOOL_MAP(self.F_kPadN):5}, {BOOL_MAP(self.F_kSaveMeanInvStd_):5}, {BOOL_MAP(self.F_kFastFDiv_):5}, {BOOL_MAP(self.F_kWelford_):5}"
            t_ += f", {BOOL_MAP(self.F_kTwoPass_):5}, {self.F_kXbias:4}, {self.F_kFusedAdd:4}, {self.F_kFusedQuant:4}"
            return t_

        # string when calling this kernel
        @property
        def call_name(self) -> str:
            return f"layernorm2d_fwd_<traits_<{self.trait_name}>>"

        # string when define this kernel
        @property
        def def_name(self) -> str:
            return f"template float layernorm2d_fwd_<traits_<{self.trait_name}>>(const S&, A);"

    # this class hold kernel under same source file
    @dataclass
    class h_instance:
        F_DataTypePair: str
        F_N: str
        F_xbias: int
        F_add: int
        F_sweep: int
        instance_list: List[Any]  # List[h_traits]

        @property
        def name(self) -> str:
            prec_i, prec_o = self.F_DataTypePair.split(",")
            dtype_str = f"{prec_i}" if prec_i == prec_o else f"{prec_i}_{prec_o}"
            nnn = f"layernorm2d_fwd_{dtype_str}_n{self.F_N}"
            if self.F_xbias != 0:
                nnn = nnn + "_" + XBIAS_ENUM_STR_MAP[self.F_xbias]
            if self.F_add != 0:
                nnn = nnn + "_" + FUSED_ADD_ENUM_STR_MAP[self.F_add]
            if self.F_sweep != 0:
                nnn = nnn + "_" + FUSED_FUSED_SWEEP_STR_MAP[self.F_sweep]
            return nnn

        @property
        def instance_name(self) -> str:
            return self.name

        @property
        def content(self) -> str:
            instance_defs = ""
            for ins in self.instance_list:
                instance_defs += ins.def_name + "\n"
            return layernorm_fwd_codegen.INSTANCE_BASE.format(
                F_instance_def=instance_defs
            )

    @property
    def name_api(self) -> str:
        return "layernorm2d_fwd_api"

    @property
    def name_common_header(self) -> str:
        return "layernorm2d_fwd_api_common"

    def content_api(self, args) -> str:
        # 1 sort based on dtype
        t_dtype_dict = dict()
        blobs = self.get_blobs(args)
        for blob in blobs:
            if blob.F_DataTypePair not in t_dtype_dict:
                t_dtype_dict[blob.F_DataTypePair] = {}
            if blob.F_N not in t_dtype_dict[blob.F_DataTypePair]:
                t_dtype_dict[blob.F_DataTypePair][blob.F_N] = []
            t_dtype_dict[blob.F_DataTypePair][blob.F_N].append(blob)

        d_str = ""
        for i_d, dtype_ in enumerate(t_dtype_dict):
            blob_per_t = t_dtype_dict[dtype_]
            n_str = ""
            for i_n, n_ in enumerate(blob_per_t):
                blob_per_n = blob_per_t[n_]
                inner_str = ""
                for i_b, b_ in enumerate(blob_per_n):
                    # generate single kernel instance file
                    # vec_str = ""
                    for i_ins, ins in enumerate(b_.instance_list):
                        idx_in_n = i_b * len(b_.instance_list) + i_ins
                        len_in_n = len(blob_per_n) * len(b_.instance_list)
                        # _if = 'if' if i_ins == 0 else 'else if'
                        if ins.F_kFusedQuant == 0:
                            _sweep_cond = "t.fused_quant == {f_fused_sweep}".format(
                                f_fused_sweep=ins.F_kFusedQuant
                            )
                        elif ins.F_kFusedQuant == 1:
                            _sweep_cond = 't.fused_quant == {f_fused_sweep} && (t.prec_sm == "{f_sx_type}" && t.prec_sy == "{f_sy_type}")'.format(
                                f_fused_sweep=ins.F_kFusedQuant,
                                f_sx_type=ins.F_SmoothScaleDataType,
                                f_sy_type=ins.F_YScaleDataType,
                            )
                        elif ins.F_kFusedQuant == 2:
                            _sweep_cond = 't.fused_quant == {f_fused_sweep} && (t.prec_sy == "{f_sy_type}")'.format(
                                f_fused_sweep=ins.F_kFusedQuant,
                                f_sy_type=ins.F_YScaleDataType,
                            )
                        _cond = "((a.n % {f_vec_n} == 0) && (t.xbias == {f_xbias}) && (t.fused_add == {f_fused_add}) && ({f_sweep_cond}))".format(
                            f_vec_n=ins.F_Vector_N,
                            f_xbias=ins.F_kXbias,
                            f_fused_add=ins.F_kFusedAdd,
                            f_sweep_cond=_sweep_cond,
                        )
                        inner_str += self.API_INNER_CASE.format(
                            F_if=get_if_str(idx_in_n, len_in_n, False),
                            F_VEC_COND=_cond,
                            F_instance_func=ins.call_name,
                        )
                    # inner_str = inner_str + vec_str
                n_cnd = f"(a.n <= {n_})" if isinstance(n_, int) else ""
                n_str += self.API_PER_N_CASE.format(
                    F_if=get_if_str(i_n, len(blob_per_t), not isinstance(n_, int)),
                    F_N_COND=n_cnd,
                    F_inner_dispatch=inner_str,
                )
            prec_i, prec_o = dtype_.split(",")
            d_str += self.API_PER_DTYPE.format(
                F_if=get_if_str(i_d, len(t_dtype_dict), False),
                F_i_type=prec_i,
                F_o_type=prec_o,
                F_per_n_case=n_str,
            )

        api_base = self.API_BASE.format(
            F_traits_define=self.API_TRAITS_DEFINE, F_dispatch=d_str
        )
        return api_base

    @property
    def content_common_header(self) -> str:
        return self.API_COMMON_HEADER.format(F_traits_define=self.API_TRAITS_DEFINE)

    def get_blobs(self, args):
        h_traits = layernorm_fwd_codegen.h_traits
        h_instance = layernorm_fwd_codegen.h_instance

        dynamic_quant_out_dtype = ["int8", "fp8"]
        # some predefined support range
        # (prec_i,prec_o) for simplicity this string will be used as key for dict
        scale_list = [("fp32,fp32")]
        dtype_list = [
            ("fp16,fp16"),
            ("bf16,bf16"),
            ("fp16,int8"),
            ("bf16,int8"),
            ("fp16,fp8"),
            ("bf16,fp8"),
        ]  # NOTE: only fused-dynamic-quant use int8 or fp8 out
        types_8bit = ("int8", "fp8")
        types_16bit = ("int16", "fp16", "bf16")
        # fused_add_list = [0, 1, 2]
        # fused_sweep_list = [0, 1, 2] # NOTE: only single pass can use fused dynamic quant
        xbias_list = [0, 1]
        fused_add_list = [0, 1]
        fused_sweep_list = [0, 1]  # NOTE: only single pass can use fused dynamic quant
        #                                                       rm  rn  tm   tn  vn  pd     mv     fdiv  welford   2p     xbias    add   sweep
        h_trait_dict = {
            "64": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    8,
                    8,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    16,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    64,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "128": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    16,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    64,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    4,
                    64,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "256": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    64,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    4,
                    64,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    4,
                    64,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "512": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    4,
                    64,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    4,
                    64,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    4,
                    64,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    8,
                    4,
                    64,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "768": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    4,
                    64,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    6,
                    4,
                    64,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    12,
                    4,
                    64,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "1024": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    2,
                    128,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    2,
                    128,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    2,
                    128,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    256,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "1536": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    4,
                    64,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    2,
                    128,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    256,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    6,
                    1,
                    256,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "2048": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    1,
                    256,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    1,
                    256,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    256,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    8,
                    1,
                    256,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "3072": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    128,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    256,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    6,
                    1,
                    256,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    1024,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "4096": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    1,
                    256,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    256,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    2,
                    1,
                    1024,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    1024,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "6144": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    256,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    512,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    3,
                    1,
                    1024,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    6,
                    1,
                    1024,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "8192": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    256,
                    8,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    512,
                    4,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    1024,
                    2,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    8,
                    1,
                    1024,
                    1,
                    True,
                    False,
                    True,
                    True,
                    False,
                    0,
                    0,
                    0,
                ),
            ],
            "big": [
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    1,
                    1,
                    1024,
                    8,
                    True,
                    False,
                    True,
                    True,
                    True,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    256,
                    4,
                    True,
                    False,
                    True,
                    True,
                    True,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    12,
                    1,
                    256,
                    2,
                    True,
                    False,
                    True,
                    True,
                    True,
                    0,
                    0,
                    0,
                ),
                h_traits(
                    "x",
                    "y",
                    "xs",
                    "ys",
                    1,
                    4,
                    1,
                    1024,
                    1,
                    True,
                    False,
                    True,
                    True,
                    True,
                    0,
                    0,
                    0,
                ),
            ],
        }
        total_blob = list()
        for hs_key in h_trait_dict:
            hs = h_trait_dict[hs_key]
            current_n = hs[0].F_Repeat_N * hs[0].F_ThreadPerBlock_N * hs[0].F_Vector_N
            for dtype, scale_type, xbias, fused_add, fused_quant in itertools.product(
                dtype_list, scale_list, xbias_list, fused_add_list, fused_sweep_list
            ):
                prec_i, prec_o = dtype.split(",")
                scale_sm, scale_y = scale_type.split(",")
                if prec_o in dynamic_quant_out_dtype and fused_quant != 1:
                    continue  # skip non dynamic quant case
                if fused_quant == 1 and hs_key == "big":
                    continue
                current_hs = list()
                for chs_ in hs:
                    h_ = copy.copy(chs_)  # copy the base instance out
                    h_.F_XDataType = prec_i
                    h_.F_YDataType = prec_o
                    h_.F_SmoothScaleDataType = scale_sm
                    h_.F_YScaleDataType = scale_y
                    h_.F_kXbias = xbias
                    h_.F_kFusedAdd = fused_add
                    h_.F_kFusedQuant = fused_quant
                    # disable welford update for 8bit and 16 bit smallN
                    if not h_.F_kTwoPass_:
                        # disable 16 bit when set args disable_16b_welford
                        if args.disable_16b_welford and prec_i in types_16bit:
                            h_.F_kWelford_ = False
                        # disable 8bit by default
                        elif prec_i in types_8bit or prec_o in types_8bit:
                            h_.F_kWelford_ = False
                        # disable 16bit small N
                        elif prec_i in types_16bit and hs_key == "64":
                            h_.F_kWelford_ = False
                    current_hs.append(h_)  # + "\n"
                # f.write(str(f.parent / GEN_DIR / (blobs.api_common_header_
                current_n_str = "big" if hs_key == "big" else current_n
                total_blob.append(
                    h_instance(
                        dtype, current_n_str, xbias, fused_add, fused_quant, current_hs
                    )
                )
        return total_blob

    def list_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        list_p = w_p / "layernorm2d_fwd_blobs.txt"
        blobs = self.get_blobs(args)
        with list_p.open("w") as list_f:
            # api related file
            list_f.write(str(w_p / (self.name_api + ".cpp")) + "\n")
            list_f.write(str(w_p / (self.name_common_header + ".hpp")) + "\n")
            # kernel instance file
            for b in blobs:
                list_f.write(str(w_p / (b.name + ".cpp")) + "\n")

    def gen_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        w_str = self.content_api(args)
        (w_p / (self.name_api + ".cpp")).write_text(w_str)
        (w_p / (self.name_common_header + ".hpp")).write_text(
            self.content_common_header
        )
        blobs = self.get_blobs(args)
        for b in blobs:
            (w_p / (b.name + ".cpp")).write_text(b.content)


def list_blobs(args):
    api_list = args.api.split(",")
    for api in api_list:
        if api == "fwd":
            layernorm_fwd_codegen(args.working_path, args.filter).list_blobs(args)


def gen_blobs(args):
    api_list = args.api.split(",")
    for api in api_list:
        if api == "fwd":
            layernorm_fwd_codegen(args.working_path, args.filter).gen_blobs(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK layernorm kernel",
    )
    parser.add_argument(
        "-a",
        "--api",
        default="fwd[all]",
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma.",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    # this script have 2 modes
    # 1) list_blobs mode, will generate a txt file with all the files going to be generated.
    #    this is useful in build system like cmake to construct source code dependency, by
    #    reading the content out of this file
    # 2) gen_blobs mode, will generate the actuall kernel instance and api. If in framework
    #    like FA, only need to use this mode
    parser.add_argument(
        "-l",
        "--list_blobs",
        action="store_true",
        help="list all the kernels to a file, ",
    )

    parser.add_argument(
        "-g",
        "--gen_blobs",
        action="store_true",
        help="generate all kernels into different tile",
    )

    # TODO: if using filter, must apply same value to output_dir and list_blobs
    parser.add_argument(
        "-f",
        "--filter",
        required=False,
        help="filter out kernels that need to generate, using fnmatch module",
    )

    parser.add_argument(
        "-t",
        "--traits",
        default="all",
        required=False,
        help="enable/disable some feature. default generate all",
    )

    parser.add_argument(
        "-r", "--receipt", default=0, required=False, help="codegen receipt."
    )

    parser.add_argument(
        "--disable_16b_welford",
        default=False,
        required=False,
        help="enable/disable welford for 16bit datatype n > 64",
    )

    args = parser.parse_args()

    # print(f'{args.list_blobs}-{args.gen_blobs}')
    if (args.gen_blobs and args.list_blobs) or (
        (not args.gen_blobs) and (not args.list_blobs)
    ):
        print("gen_blobs/list_blobs must specify only one option")
        sys.exit()

    p = Path(args.working_path)
    if not p.exists():
        p.mkdir()

    if args.list_blobs:
        list_blobs(args)
    else:
        gen_blobs(args)
