#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"

OP=$1
DATATYPE=$2
OUTELEMENTOP=$3
LAYOUT=$4
VERIFY=$5
INIT=$6
LOG=$7
TIME=$8

N=$9

#######  op    datatype  OUTELEMENTOP  layout   verify   init   log   time  Ndims  G    N    K     C   Z   Y   X   Di  Hi   Wi  Sz  Sy  Sx  Dz  Dy  Dx  Left Pz LeftPy  LeftPx  RightPz RightPy  RightPx
$DRIVER $OP   $DATATYPE $OUTELEMENTOP $LAYOUT  $VERIFY  $INIT  $LOG  $TIME      3 32   $N   96    96   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
$DRIVER $OP   $DATATYPE $OUTELEMENTOP $LAYOUT  $VERIFY  $INIT  $LOG  $TIME      3 32   $N  192   192   3   3   3   28  28   28   1   1   1   1   1   1        1      1       1        1       1        1
