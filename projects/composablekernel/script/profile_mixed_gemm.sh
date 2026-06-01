#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
echo $DRIVER
OP=$1
DATATYPE=$2
LAYOUT=$3
VERIFY=$4
INIT=$5
LOG=$6
TIME=$7
KBatch=$8

########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC  KBatch_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16    16 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16    16 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16    16 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  2048 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  2048 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  2048 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  8192 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  8192 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME    16  8192 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048    16 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048    16 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048    16 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  2048 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  2048 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  2048 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  8192 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  8192 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  2048  8192 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192    16 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192    16 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192    16 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  2048 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  2048 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  2048 65536      -1     -1      -1   $KBatch

 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  8192 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  8192 8192       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  8192  8192 65536      -1     -1      -1   $KBatch
 