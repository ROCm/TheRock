#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
echo $DRIVER
OP=$1
DATATYPE=$2
VERIFY=$3
INIT=$4
LOG=$5
TIME=$6


# 1D
########  op  datatype  verify  init  log  time  dims     in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  67108864 0                0

# # 2D
# ########  op  datatype  verify  init  log  time  dims      in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8192 8192   0 1              1 0
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8192 8192   1 0              0 1

# 3D
########  op  datatype  verify  init  log  time  dims        in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 1024 8192 0 1 2            2 1 0
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 1024 8192 2 1 0            0 1 2

# 4D
########  op  datatype  verify  init  log  time  dims          in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 512 8192  0 1 2 3          3 2 1 0
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 512 8192  3 2 1 0          0 1 2 3

# 5D
########  op  datatype  verify  init  log  time  dims            in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 2 256 8192  0 1 2 3 4        4 3 2 1 0
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 2 256 8192  4 3 2 1 0        0 1 2 3 4

 # 6D
########  op  datatype  verify  init  log  time  dims             in_strides_order out_strides_order
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 2 2 128 8192 0 1 2 3 4 5      5 4 3 2 1 0
 $DRIVER $OP $DATATYPE $VERIFY $INIT $LOG $TIME  8 2 2 2 128 8192 5 4 3 2 1 0      0 1 2 3 4 5

