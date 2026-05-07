# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import statistics
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class MeasurementKey:
    name: str = None
    samples: int = 0
    mean: float = 0
    median: float = 0
    values: List[float] = field(init=False)

    def __init__(self, name):
        self.name = name
        self.values = []

    def getMeanName(self):
        assert(self.name != None)
        return 'mean_' + self.name

    def getMedianName(self):
        assert(self.name != None)
        return 'median_' + self.name

    def addSample(self, value):
        self.samples += 1
        self.values.append(float(value))

    def calcMean(self):
        assert(len(self.values) == self.samples)
        assert(self.name != None)
        self.mean = statistics.mean(self.values)
        return self.mean

    def calcMedian(self):
        assert(len(self.values) == self.samples)
        assert(self.name != None)
        self.median = statistics.median(self.values)
        return self.median

    def getValuesStr(self):
        assert(len(self.values) == self.samples)
        assert(self.name != None)
        return ','.join([str(v) for v in self.values])

@dataclass
class BenchSample:
    probKey: str = None
    measurements: Dict[str, MeasurementKey] = field(default_factory=dict)

    def addSampleOfKey(self, key, value):
        if key not in self.measurements:
            self.measurements[key] = MeasurementKey(name=key)
        self.measurements[key].addSample(value)

    def getMeasurementMeanMedianPair(self, key):
        mean = self.measurements[key].calcMean()
        median = self.measurements[key].calcMedian()
        return str(mean) + ',' + str(median)

    def getMeasurementValues(self, key):
        return self.measurements[key].getValuesStr()

    def finalize(self, num_samples, keysOrder):
        assert(self.probKey != None)
        content = self.probKey + ',' + str(num_samples)

        for key in keysOrder:
            assert(self.measurements[key].samples == num_samples)
            content += ',' + self.getMeasurementMeanMedianPair(key)

        for key in keysOrder:
            content += ',' + self.getMeasurementValues(key)

        return content
