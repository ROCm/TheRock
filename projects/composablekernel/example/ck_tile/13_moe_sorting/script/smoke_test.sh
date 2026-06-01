# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# #!/bin/sh

EXE=./build/bin/tile_example_moe_sorting
MOE_BUF="12"

if [ "x$MOE_BUF" = "x1" ] ; then
$EXE -t=80 -e=17 -moe_buf_size=16
$EXE -t=111 -e=117 -moe_buf_size=4
$EXE -t=1000 -e=55 -moe_buf_size=1024
$EXE -t=99 -e=120  -moe_buf_size=10244
$EXE -t=175 -e=64 -k=8
$EXE -t=65 -e=8 -k=2
$EXE -t=1 -e=25
$EXE -t=31 -e=19 -k=15
$EXE -t=81 -e=37 -k=7
$EXE -t=23 -e=1 -k=1
$EXE -t=127 -e=99 -k=19
$EXE -t=71 -e=11 -k=11
$EXE -t=1 -e=1 -k=1
$EXE -t=99 -e=2 -k=1
$EXE -t=333 -e=99 -k=13
$EXE -t=11 -e=256 -k=5
$EXE -t=64 -e=455 -k=8
$EXE -t=777 -e=802 -k=99
$EXE -t=4097 -e=906 -k=51
$EXE -t=128 -e=32 -k=5 -moe_buf_size=262144
$EXE -t=13 -e=64 -k=3 -local_eid=4,5,6,7,8,9,10,11
$EXE -t=99 -e=33 -k=9 -local_eid=6,10,11,15,19
$EXE -t=80 -e=99 -k=10 -local_eid=0,8,12,33
$EXE -t=11 -e=256 -k=5 -local_eid=99,110,129
$EXE -t=128 -e=128 -k=6 -moe_buf_size=163840
$EXE -t=8192 -e=32 -k=5 -moe_buf_size=163840
$EXE -t=8192 -e=32 -k=8 -moe_buf_size=163840
$EXE -t=8192 -e=256 -k=5 -moe_buf_size=163840
$EXE -t=8192 -e=256 -k=8 -moe_buf_size=163840
$EXE -t=163840 -e=256 -k=8 -moe_buf_size=163840
$EXE -t=12 -local_t=3 -e=256 -k=5 -local_eid=9,10,199,145
$EXE -t=67 -local_t=9 -e=555 -k=5 -local_eid=19,23,24,25,26,99
$EXE -t=99 -local_t=93 -e=121 -moe_buf_size=10244
$EXE -t=536 -local_t=345 -e=802 -k=99
$EXE -t=331 -local_t=39 -e=83 -k=33
$EXE -t=765 -local_t=654 -e=783 -k=8
$EXE -t=23 -local_t=9 -e=1 -k=1
$EXE -t=7 -local_t=0 -e=89 -k=1 -local_eid=0,8,12,33
$EXE -t=61 -local_t=0 -e=333 -k=99 -local_eid=0,8,12,33
$EXE -t=133940 -local_t=111921 -e=256 -k=17 -moe_buf_size=133940
else
$EXE -t=80 -e=17 -moe_buf_interm_dim=16 -moe_buf_elem_bytes=4
$EXE -t=111 -e=117 -moe_buf_interm_dim=4 -moe_buf_elem_bytes=4
$EXE -t=1000 -e=55 -moe_buf_interm_dim=1024 -moe_buf_elem_bytes=1
$EXE -t=99 -e=120  -moe_buf_interm_dim=10244 -moe_buf_elem_bytes=2
$EXE -t=175 -e=64 -k=8
$EXE -t=65 -e=8 -k=2
$EXE -t=1 -e=25
$EXE -t=31 -e=19 -k=15
$EXE -t=81 -e=37 -k=7
$EXE -t=23 -e=1 -k=1
$EXE -t=127 -e=99 -k=19
$EXE -t=71 -e=11 -k=11
$EXE -t=1 -e=1 -k=1
$EXE -t=99 -e=2 -k=1
$EXE -t=333 -e=99 -k=13
$EXE -t=11 -e=256 -k=5
$EXE -t=64 -e=455 -k=8
$EXE -t=777 -e=802 -k=99
$EXE -t=4097 -e=906 -k=51
$EXE -t=128 -e=32 -k=5 -local_t=6 -moe_buf_interm_dim=262144
$EXE -t=13 -e=64 -k=3 -local_eid=4,5,6,7,8,9,10,11
$EXE -t=99 -e=33 -k=9 -local_eid=6,10,11,15,19
$EXE -t=80 -e=99 -k=10 -local_eid=0,8,12,33
$EXE -t=11 -e=256 -k=5 -local_eid=99,110,129
$EXE -t=128 -e=128 -k=6 -moe_buf_interm_dim=163840 -moe_buf_elem_bytes=1
$EXE -t=8192 -e=32 -k=5 -local_t=11 -moe_buf_interm_dim=163840
$EXE -t=8192 -e=32 -k=8 -local_t=12 -moe_buf_interm_dim=163840 -moe_buf_elem_bytes=1
$EXE -t=8192 -e=256 -k=5 -local_t=13 -moe_buf_interm_dim=163840
$EXE -t=8192 -e=256 -k=8 -local_t=8 -moe_buf_interm_dim=163840
$EXE -t=163840 -e=256 -k=8 -local_t=4 -moe_buf_interm_dim=163840 -moe_buf_elem_bytes=4
$EXE -t=12 -local_t=3 -e=256 -k=5 -local_eid=9,10,199,145
$EXE -t=67 -local_t=9 -e=555 -k=5 -local_eid=19,23,24,25,26,99
$EXE -t=99 -local_t=93 -e=121 -local_t=4 -moe_buf_interm_dim=10244
$EXE -t=536 -local_t=345 -e=802 -k=99
$EXE -t=331 -local_t=39 -e=83 -k=33
$EXE -t=765 -local_t=654 -e=783 -k=8
$EXE -t=23 -local_t=9 -e=1 -k=1
$EXE -t=7 -local_t=0 -e=89 -k=1 -local_eid=0,8,12,33
$EXE -t=61 -local_t=0 -e=333 -k=99 -local_eid=0,8,12,33
$EXE -t=133940 -local_t=111921 -e=256 -k=17 -local_t=2 -moe_buf_interm_dim=133940 -moe_buf_elem_bytes=1

fi
