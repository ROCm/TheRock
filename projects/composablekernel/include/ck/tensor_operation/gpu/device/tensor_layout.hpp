// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_layout {

struct BaseTensorLayout
{
    static constexpr const char* name = "BaseTensorLayout";
};

struct BypassLayoutVerification : public BaseTensorLayout
{
    static constexpr const char* name = "BypassLayoutVerification";
};

namespace gemm {

struct BaseGemmLayout : public BaseTensorLayout
{
    static constexpr const char* name = "BaseConvolutionLayout";
};
struct RowMajor : public BaseGemmLayout
{
    static constexpr const char* name = "RowMajor";
};

struct ColumnMajor : public BaseGemmLayout
{
    static constexpr const char* name = "ColumnMajor";
};

struct MFMA : public BaseGemmLayout
{
    static constexpr const char* name = "MFMA";
};

} // namespace gemm

namespace convolution {

struct BaseConvolutionLayout : public BaseTensorLayout
{
    static constexpr const char* name = "BaseConvolutionLayout";
};

// input tensor
// packed NCW/NCHW/NCDHW
struct NCW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NCW";
};

struct NCHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NCHW";
};

struct NCDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NCDHW";
};

// packed GNCW/GNCHW/GNCDHW
struct GNCW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNCW";
};

struct GNCHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNCHW";
};

struct GNCDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNCDHW";
};

// input tensor
// packed NWC/NHWC/NDHWC
struct NWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NWC";
};

struct NHWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NHWC";
};

struct NDHWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NDHWC";
};

// input tensor
// packed GNWC/GNHWC/GNDHWC
struct GNWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNWC";
};

struct GNHWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNHWC";
};

struct GNDHWC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNDHWC";
};

// for input bias
struct GC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GC";
};

// input tensor
// packed NWGC/NHWGC/NDHWGC
struct NWGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NWGC";
};

struct NHWGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NHWGC";
};

struct NDHWGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "NDHWGC";
};

// input tensor
// packed NGCW/NGCHW/NGCDHW
struct NGCW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGCW";
};

struct NGCHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGCHW";
};

struct NGCDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGCDHW";
};

// input tensor
// strided layout
struct G_NW_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NW_C";
};

struct G_NHW_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NHW_C";
};

struct G_NDHW_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NDHW_C";
};

// for input bias
struct G_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_C";
};

// weight tensor
// packed KCX/KCYX/KCZYX
struct KCX : public BaseConvolutionLayout
{
    static constexpr const char* name = "KCX";
};

struct KCYX : public BaseConvolutionLayout
{
    static constexpr const char* name = "KCYX";
};

struct KCZYX : public BaseConvolutionLayout
{
    static constexpr const char* name = "KCZYX";
};

// weight tensor
// packed KCX/KCYX/KCZYX
struct GKCX : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKCX";
};

struct GKCYX : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKCYX";
};

struct GKCZYX : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKCZYX";
};

// weight tensor
// packed KXC/KYXC/KZYXC
struct KXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KXC";
};

struct KYXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KYXC";
};

struct KZYXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KZYXC";
};

// weight tensor
// packed GKXC/GKYXC/GKZYXC
struct GKXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKXC";
};

struct GKYXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKYXC";
};

struct GKZYXC : public BaseConvolutionLayout
{
    static constexpr const char* name = "GKZYXC";
};

// weight tensor
// packed KXGC/KYXGC/KZYXGC
struct KXGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KXGC";
};

struct KYXGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KYXGC";
};

struct KZYXGC : public BaseConvolutionLayout
{
    static constexpr const char* name = "KZYXGC";
};

// weight tensor
// strided
struct G_K_X_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_K_X_C";
};

struct G_K_YX_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_K_YX_C";
};

struct G_K_ZYX_C : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_K_ZYX_C";
};

// output tensor
// packed NKW/NKHW/NKDHW
struct NKW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NKW";
};

struct NKHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NKHW";
};

struct NKDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NKDHW";
};

// output tensor
// packed GNKW/GNKHW/GNKDHW
struct GNKW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNKW";
};

struct GNKHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNKHW";
};

struct GNKDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNKDHW";
};

// output tensor
// packed NWK/NHWK/NDHWK
struct NWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NWK";
};

struct NHWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NHWK";
};

struct NDHWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NDHWK";
};

// output tensor
// packed GNWK/GNHWK/GNDHWK
struct GNWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNWK";
};

struct GNHWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNHWK";
};

struct GNDHWK : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNDHWK";
};

// output tensor
// packed NWGK/NHWGK/NDHWGK
struct NWGK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NWGK";
};

struct NHWGK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NHWGK";
};

struct NDHWGK : public BaseConvolutionLayout
{
    static constexpr const char* name = "NDHWGK";
};

struct NGKW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGKW";
};

struct NGKHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGKHW";
};

struct NGKDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "NGKDHW";
};

// output tensor
// strided layout
struct G_NW_K : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NW_K";
};

struct G_NHW_K : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NHW_K";
};

struct G_NDHW_K : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NDHW_K";
};

// for output bias
struct G_K : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_K";
};

// K-reduced output tensor (packed)
struct GNW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNW";
};

struct GNHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNHW";
};

struct GNDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "GNDHW";
};

// K-reduced output tensor (packed)
struct NWG : public BaseConvolutionLayout
{
    static constexpr const char* name = "NWG";
};

struct NHWG : public BaseConvolutionLayout
{
    static constexpr const char* name = "NHWG";
};

struct NDHWG : public BaseConvolutionLayout
{
    static constexpr const char* name = "NDHWG";
};

// K-reduced output tensor (strided)
struct G_NW : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NW";
};

struct G_NHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NHW";
};

struct G_NDHW : public BaseConvolutionLayout
{
    static constexpr const char* name = "G_NDHW";
};

} // namespace convolution

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
template <
    typename Layout,
    typename std::enable_if<std::is_base_of<BaseTensorLayout, Layout>::value, bool>::type = false>
std::ostream& operator<<(std::ostream& os, const Layout&)
{
    os << Layout::name;
    return os;
}
#endif

} // namespace tensor_layout
} // namespace ck
