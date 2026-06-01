/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "argument_model.hpp"
#include "efficiency_monitor.hpp"

// FLOPS/CLOCK/CU values for gfx942
auto hipblaslt_get_flops_per_clock_per_cu_gfx942(hipDataType          inputType,
                                                 hipblasComputeType_t computeType)
{
    if(inputType == HIP_R_32F || inputType == HIP_R_64F)
        return 256;
    else if(computeType == HIPBLAS_COMPUTE_32F_FAST_TF32)
        return 1024;
    else if(inputType == HIP_R_16F || inputType == HIP_R_16BF)
        return 2048;
    else if(inputType == HIP_R_8F_E4M3_FNUZ || inputType == HIP_R_8F_E5M2_FNUZ
            || inputType == HIP_R_8F_E4M3 || inputType == HIP_R_8F_E5M2 || inputType == HIP_R_8I)
        return 4096;
    else
        return 0;
}

// this should have been a member variable but due to the complex variadic template this singleton allows global control
static bool log_function_name = false;

void ArgumentModel_set_log_function_name(bool f)
{
    log_function_name = f;
}

bool ArgumentModel_get_log_function_name()
{
    return log_function_name;
}

void ArgumentModel_log_efficiency(hipblaslt_internal_ostream& name_line,
                                  hipblaslt_internal_ostream& val_line,
                                  const Arguments&            arg,
                                  double                      hipblaslt_gflops)
{
    EfficiencyMonitor& efficiency_monitor = getEfficiencyMonitor();
    if(!efficiency_monitor.enabled())
        return;

    if(efficiency_monitor.getDeviceString() == "gfx942"
       && hipblaslt_get_flops_per_clock_per_cu_gfx942(arg.a_type, arg.compute_type) != 0)
    {
        double theoretical_gflops
            = hipblaslt_get_flops_per_clock_per_cu_gfx942(arg.a_type, arg.compute_type)
              * efficiency_monitor.getCuCount() * efficiency_monitor.getLowestAverageSYSCLK()
              * 0.001;
        name_line << ",efficiency";
        val_line << "," << (hipblaslt_gflops / theoretical_gflops) * 100;
    }
}

void ArgumentModel_log_performance(hipblaslt_internal_ostream& name_line,
                                   hipblaslt_internal_ostream& val_line)
{

    EfficiencyMonitor& efficiency_monitor = getEfficiencyMonitor();
    if(!efficiency_monitor.enabled())
        return;

    if(getenv("HIPBLASLT_BENCH_EFF") != nullptr)
    {
        name_line << ",total_gran";
        val_line << "," << efficiency_monitor.getTotalGranularityValue();

        name_line << ",tiles_per_cu";
        val_line << "," << efficiency_monitor.getTilesPerCuValue();

        name_line << ",num_cu's";
        val_line << "," << efficiency_monitor.getCUs();

        name_line << ",tile_0_granularity";
        val_line << "," << efficiency_monitor.getTile0Granularity();

        name_line << ",tile_1_granularity";
        val_line << "," << efficiency_monitor.getTile1Granularity();

        name_line << ",cu_gran";
        val_line << "," << efficiency_monitor.getCuGranularity();

        name_line << ",wave_gran";
        val_line << "," << efficiency_monitor.getWaveGranularity();

        name_line << ",mem_read_bytes";
        val_line << "," << efficiency_monitor.getMemReadBytes();

        name_line << ",mem_write_bytes";
        val_line << "," << efficiency_monitor.getMemWriteBytesD();
    }

    if(!efficiency_monitor.detailedReport())
    {
        name_line << ",lowest_avg_freq";
        val_line << "," << efficiency_monitor.getLowestAverageSYSCLK();

        name_line << ",lowest_median_freq";
        val_line << "," << efficiency_monitor.getLowestMedianSYSCLK();
    }
    else
    {
        auto allAvgSYSCLK = efficiency_monitor.getAllAverageSYSCLK();
        for(int i = 0; i < allAvgSYSCLK.size(); i++)
        {
            name_line << ",avg_freq" << i;
            val_line << "," << allAvgSYSCLK[i];
        }

        auto allMedianSYSCLK = efficiency_monitor.getAllMedianSYSCLK();
        for(int i = 0; i < allMedianSYSCLK.size(); i++)
        {
            name_line << ",median_freq" << i;
            val_line << "," << allMedianSYSCLK[i];
        }
    }

    name_line << ",avg_MCLK";
    val_line << "," << efficiency_monitor.getAverageMEMCLK();

    name_line << ",median_MCLK";
    val_line << "," << efficiency_monitor.getMedianMEMCLK();
}
