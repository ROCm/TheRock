// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

// keep sync with BlockAttentionQuantScaleEnum
enum class quant_scale_enum
{
    no_scale  = 0,
    pertensor = 1,
};

struct quant_scale_info
{
    quant_scale_enum type;

    void serialize(std::ostream& os) const
    {
        if(type == quant_scale_enum::no_scale)
            os << "n";
        else if(type == quant_scale_enum::pertensor)
            os << "pt";
    }

    static quant_scale_info decode(std::string str)
    {
        quant_scale_info info{quant_scale_enum::no_scale};
        if(str == "n" || str == "0")
        {
            info.type = quant_scale_enum::no_scale;
        }
        else if(str == "pt" || str == "1")
        {
            info.type = quant_scale_enum::pertensor;
        }
        else
        {
            throw std::invalid_argument("invalid quant scale value: " + str);
        }
        return info;
    }

    friend std::ostream& operator<<(std::ostream& os, const quant_scale_info& qsi)
    {
        qsi.serialize(os);
        return os;
    }
};
