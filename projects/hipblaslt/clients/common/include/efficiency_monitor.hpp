
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************/
#pragma once

#include <vector>
class EfficiencyMonitor
{
public:
    virtual bool enabled()        = 0;
    virtual bool detailedReport() = 0;

    virtual void set_device_id(int deviceId) = 0;

    virtual void start() = 0;
    virtual void stop()  = 0;

    virtual double              getLowestAverageSYSCLK()   = 0;
    virtual double              getLowestMedianSYSCLK()    = 0;
    virtual std::vector<double> getAllAverageSYSCLK()      = 0;
    virtual std::vector<double> getAllMedianSYSCLK()       = 0;
    virtual double              getAverageMEMCLK()         = 0;
    virtual double              getMedianMEMCLK()          = 0;
    virtual double              getTotalGranularityValue() = 0;
    virtual double              getTilesPerCuValue()       = 0;
    virtual double              getTile0Granularity()      = 0;
    virtual double              getTile1Granularity()      = 0;
    virtual double              getCuGranularity()         = 0;
    virtual double              getWaveGranularity()       = 0;
    virtual int                 getCUs()                   = 0;
    virtual size_t              getMemWriteBytesD()        = 0;
    virtual size_t              getMemReadBytes()          = 0;
    virtual uint16_t            getCuCount()               = 0;
    virtual std::string         getDeviceString()          = 0;
};

EfficiencyMonitor& getEfficiencyMonitor();
void                freeEfficiencyMonitor();
