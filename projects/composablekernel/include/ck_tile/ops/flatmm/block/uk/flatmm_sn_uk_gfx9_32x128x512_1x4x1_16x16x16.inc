// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// clang-format off
// define the CK_TILE_** macro before include this file to change kernel variation
// we will undef everything defined in this file

#ifndef CK_TILE_FLATMM_UK_MFMA
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_BF16
#endif

#if CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_BF16
#define _UK_MFMA_ "v_mfma_f32_16x16x16_bf16"

#define _UK_PK_CVT_(x0_, x1_, y_)                            \
    "  v_cmp_u_f32   s[36:37], " x0_ ", " x0_ "          \n" \
    "  v_add3_u32    v50, " x0_ ", %[v_nan_lo], 1        \n" \
    "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[36:37]    \n" \
    "  v_cmp_u_f32   s[36:37], " x1_ ", " x1_ "          \n" \
    "  v_add3_u32    v50, " x1_ ", %[v_nan_lo], 1        \n" \
    "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[36:37]    \n" \
    "  v_perm_b32    " y_ ", v55, v54, s52               \n"

#define _UK_ATOMIC_ADD_ "global_atomic_pk_add_bf16"

#elif CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_FP16
#define _UK_MFMA_ "v_mfma_f32_16x16x16_f16"

#define _UK_PK_CVT_(x0_, x1_, y_)       \
    "  v_cvt_f16_f32  v54, " x0_ "  \n" \
    "  v_cvt_f16_f32  v55, " x1_ "  \n" \
    "  v_pack_b32_f16 " y_ ", v54, v55  \n"

#define _UK_ATOMIC_ADD_ "global_atomic_pk_add_f16"

#endif

";-------------------------------------------------------------\n"
    " s_mov_b32 s52, 0x07060302 ; v_perm\n"
    " s_mov_b64 s[38:39], exec ; save current exec\n"
    " s_mov_b32 s8,    %[s_res_o0] \n"
    " s_mov_b32 s9,    %[s_res_o1] \n"
    " s_mov_b32 s12,    %[s_res_b0] \n"
    " s_mov_b32 s13,    %[s_res_b1] \n"
    " s_mov_b32 s14,    %[s_res_b2] \n"
    " s_mov_b32 s15,    %[s_res_b3] \n"
    " ds_read_b64   v[128:129], %[v_sld_y_os] offset:0 + %[sld_a_base]                       \n"
    " ds_read_b64   v[130:131], %[v_sld_y_os] offset:128 + %[sld_a_base]                     \n"
    " ds_read_b64   v[132:133], %[v_sld_y_os] offset:1024 + %[sld_a_base]                    \n"
    " ds_read_b64   v[134:135], %[v_sld_y_os] offset:1152 + %[sld_a_base]                    \n"
    " ds_read_b64   v[136:137], %[v_sld_y_os] offset:2048 + %[sld_a_base]                    \n"
    " ds_read_b64   v[138:139], %[v_sld_y_os] offset:2176 + %[sld_a_base]                    \n"
    " ds_read_b64   v[140:141], %[v_sld_y_os] offset:3072 + %[sld_a_base]                    \n"
    " ds_read_b64   v[142:143], %[v_sld_y_os] offset:3200 + %[sld_a_base]                    \n"
    " ds_read_b64   v[144:145], %[v_sld_y_os] offset:4096 + %[sld_a_base]                    \n"
    " ds_read_b64   v[146:147], %[v_sld_y_os] offset:4224 + %[sld_a_base]                    \n"
    " ds_read_b64   v[148:149], %[v_sld_y_os] offset:5120 + %[sld_a_base]                    \n"
    " ds_read_b64   v[150:151], %[v_sld_y_os] offset:5248 + %[sld_a_base]                    \n"
    " ds_read_b64   v[152:153], %[v_sld_y_os] offset:6144 + %[sld_a_base]                    \n"
    " ds_read_b64   v[154:155], %[v_sld_y_os] offset:6272 + %[sld_a_base]                    \n"
    " ds_read_b64   v[156:157], %[v_sld_y_os] offset:7168 + %[sld_a_base]                    \n"
    " ds_read_b64   v[158:159], %[v_sld_y_os] offset:7296 + %[sld_a_base]                    \n"
    " ds_read_b64   v[160:161], %[v_sld_y_os] offset:8192 + %[sld_a_base]                    \n"
    " ds_read_b64   v[162:163], %[v_sld_y_os] offset:8320 + %[sld_a_base]                    \n"
    " ds_read_b64   v[164:165], %[v_sld_y_os] offset:9216 + %[sld_a_base]                    \n"
    " ds_read_b64   v[166:167], %[v_sld_y_os] offset:9344 + %[sld_a_base]                    \n"
    " ds_read_b64   v[168:169], %[v_sld_y_os] offset:10240 + %[sld_a_base]                    \n"
    " ds_read_b64   v[170:171], %[v_sld_y_os] offset:10368 + %[sld_a_base]                    \n"
    " ds_read_b64   v[172:173], %[v_sld_y_os] offset:11264 + %[sld_a_base]                    \n"
    " ds_read_b64   v[174:175], %[v_sld_y_os] offset:11392 + %[sld_a_base]                    \n"
    " ds_read_b64   v[176:177], %[v_sld_y_os] offset:12288 + %[sld_a_base]                    \n"
    " ds_read_b64   v[178:179], %[v_sld_y_os] offset:12416 + %[sld_a_base]                    \n"
    " ds_read_b64   v[180:181], %[v_sld_y_os] offset:13312 + %[sld_a_base]                    \n"
    " ds_read_b64   v[182:183], %[v_sld_y_os] offset:13440 + %[sld_a_base]                    \n"
    " ds_read_b64   v[184:185], %[v_sld_y_os] offset:14336 + %[sld_a_base]                    \n"
    " ds_read_b64   v[186:187], %[v_sld_y_os] offset:14464 + %[sld_a_base]                    \n"
    " ds_read_b64   v[188:189], %[v_sld_y_os] offset:15360 + %[sld_a_base]                    \n"
    " ds_read_b64   v[190:191], %[v_sld_y_os] offset:15488 + %[sld_a_base]                    \n"
    " ds_read_b64   v[192:193], %[v_sld_y_os] offset:16384 + %[sld_a_base]                    \n"
    " ds_read_b64   v[194:195], %[v_sld_y_os] offset:16512 + %[sld_a_base]                    \n"
    " ds_read_b64   v[196:197], %[v_sld_y_os] offset:17408 + %[sld_a_base]                    \n"
    " ds_read_b64   v[198:199], %[v_sld_y_os] offset:17536 + %[sld_a_base]                    \n"
    " ds_read_b64   v[200:201], %[v_sld_y_os] offset:18432 + %[sld_a_base]                    \n"
    " ds_read_b64   v[202:203], %[v_sld_y_os] offset:18560 + %[sld_a_base]                    \n"
    " ds_read_b64   v[204:205], %[v_sld_y_os] offset:19456 + %[sld_a_base]                    \n"
    " ds_read_b64   v[206:207], %[v_sld_y_os] offset:19584 + %[sld_a_base]                    \n"
    " ds_read_b64   v[208:209], %[v_sld_y_os] offset:20480 + %[sld_a_base]                    \n"
    " ds_read_b64   v[210:211], %[v_sld_y_os] offset:20608 + %[sld_a_base]                    \n"
    " ds_read_b64   v[212:213], %[v_sld_y_os] offset:21504 + %[sld_a_base]                    \n"
    " ds_read_b64   v[214:215], %[v_sld_y_os] offset:21632 + %[sld_a_base]                    \n"
    " ds_read_b64   v[216:217], %[v_sld_y_os] offset:22528 + %[sld_a_base]                    \n"
    " ds_read_b64   v[218:219], %[v_sld_y_os] offset:22656 + %[sld_a_base]                    \n"
    " ds_read_b64   v[220:221], %[v_sld_y_os] offset:23552 + %[sld_a_base]                    \n"
    " ds_read_b64   v[222:223], %[v_sld_y_os] offset:23680 + %[sld_a_base]                    \n"
    " ds_read_b64   v[224:225], %[v_sld_y_os] offset:24576 + %[sld_a_base]                    \n"
    " ds_read_b64   v[226:227], %[v_sld_y_os] offset:24704 + %[sld_a_base]                    \n"
    " ds_read_b64   v[228:229], %[v_sld_y_os] offset:25600 + %[sld_a_base]                    \n"
    " ds_read_b64   v[230:231], %[v_sld_y_os] offset:25728 + %[sld_a_base]                    \n"
    " ds_read_b64   v[232:233], %[v_sld_y_os] offset:26624 + %[sld_a_base]                    \n"
    " ds_read_b64   v[234:235], %[v_sld_y_os] offset:26752 + %[sld_a_base]                    \n"
    " ds_read_b64   v[236:237], %[v_sld_y_os] offset:27648 + %[sld_a_base]                    \n"
    " ds_read_b64   v[238:239], %[v_sld_y_os] offset:27776 + %[sld_a_base]                    \n"
    " ds_read_b64   v[240:241], %[v_sld_y_os] offset:28672 + %[sld_a_base]                    \n"
    " ds_read_b64   v[242:243], %[v_sld_y_os] offset:28800 + %[sld_a_base]                    \n"
    " ds_read_b64   v[244:245], %[v_sld_y_os] offset:29696 + %[sld_a_base]                    \n"
    " ds_read_b64   v[246:247], %[v_sld_y_os] offset:29824 + %[sld_a_base]                    \n"
    " ds_read_b64   v[248:249], %[v_sld_y_os] offset:30720 + %[sld_a_base]                    \n"
    " ds_read_b64   v[250:251], %[v_sld_y_os] offset:30848 + %[sld_a_base]                    \n"
    " ds_read_b64   v[252:253], %[v_sld_y_os] offset:31744 + %[sld_a_base]                    \n"
    " ds_read_b64   v[254:255], %[v_sld_y_os] offset:31872 + %[sld_a_base]                    \n"
    "  s_waitcnt 0                    \n"
    "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[12:15], 0 offen offset:3072 \n"
    "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[12:15], 0 offen \n"
    "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[12:15], 0 offen offset:1024 \n"
    "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[12:15], 0 offen offset:2048 \n"
    "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[12:15], 0 offen offset:3072 \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
    "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
    "  s_add_u32     s12, s86, s12                                  \n"
    "  s_addc_u32    s13, 0, s13                                    \n"
    "  s_waitcnt 0                    \n"
    "L_start%=:                    \n"
    "  s_waitcnt     vmcnt(32)                               \n"
    "  s_barrier                                             \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[0:1], v[128:129], 0 \n"
    "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[2:3], v[130:131], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[4:5], v[132:133], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[6:7], v[134:135], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[8:9], v[136:137], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[10:11], v[138:139], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[12:13], v[140:141], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[14:15], v[142:143], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[0:1], v[192:193], 0 \n"
    "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[2:3], v[194:195], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[4:5], v[196:197], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[6:7], v[198:199], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[8:9], v[200:201], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[10:11], v[202:203], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[12:13], v[204:205], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[14:15], v[206:207], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[16:17], v[128:129], 0 \n"
    "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[18:19], v[130:131], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[20:21], v[132:133], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[22:23], v[134:135], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[24:25], v[136:137], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[26:27], v[138:139], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[28:29], v[140:141], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[30:31], v[142:143], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[16:17], v[192:193], 0 \n"
    "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[18:19], v[194:195], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[20:21], v[196:197], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[22:23], v[198:199], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[24:25], v[200:201], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[26:27], v[202:203], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[28:29], v[204:205], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[30:31], "
    "v[206:207], [%[c12], %[c13], %[c14], %[c15]] \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[32:33], v[144:145], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[160:163], %[v_os_b2], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[34:35], v[146:147], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[36:37], v[148:149], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[38:39], v[150:151], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[40:41], v[152:153], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[164:167], %[v_os_b2], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[42:43], v[154:155], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[44:45], v[156:157], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[46:47], v[158:159], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[32:33], v[208:209], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[168:171], %[v_os_b2], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[34:35], v[210:211], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[36:37], v[212:213], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[38:39], v[214:215], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[40:41], v[216:217], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[172:175], %[v_os_b2], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[42:43], v[218:219], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[44:45], v[220:221], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[46:47], v[222:223], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[48:49], v[144:145], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[176:179], %[v_os_b3], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[50:51], v[146:147], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[52:53], v[148:149], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[54:55], v[150:151], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[56:57], v[152:153], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[180:183], %[v_os_b3], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[58:59], v[154:155], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[60:61], v[156:157], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[62:63], v[158:159], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[48:49], v[208:209], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[184:187], %[v_os_b3], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[50:51], v[210:211], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[52:53], v[212:213], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[54:55], v[214:215], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[56:57], v[216:217], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[188:191], %[v_os_b3], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[58:59], v[218:219], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[60:61], v[220:221], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[62:63], "
    "v[222:223], [%[c12], %[c13], %[c14], %[c15]] \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[64:65], v[160:161], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[192:195], %[v_os_b4], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[66:67], v[162:163], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[68:69], v[164:165], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[70:71], v[166:167], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[72:73], v[168:169], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[196:199], %[v_os_b4], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[74:75], v[170:171], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[76:77], v[172:173], [%[c0], %[c1], %[c2], "
    "%[c3]] \n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[78:79], v[174:175], "
    "[%[c0], %[c1], %[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[64:65], v[224:225], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[200:203], %[v_os_b4], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[66:67], v[226:227], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[68:69], v[228:229], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[70:71], v[230:231], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[72:73], v[232:233], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[204:207], %[v_os_b4], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[74:75], v[234:235], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[76:77], v[236:237], [%[c4], %[c5], %[c6], "
    "%[c7]] \n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[78:79], v[238:239], "
    "[%[c4], %[c5], %[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[80:81], v[160:161], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[208:211], %[v_os_b5], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[82:83], v[162:163], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[84:85], v[164:165], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[86:87], v[166:167], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[88:89], v[168:169], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[212:215], %[v_os_b5], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[90:91], v[170:171], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[92:93], v[172:173], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[94:95], v[174:175], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[80:81], v[224:225], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[216:219], %[v_os_b5], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[82:83], v[226:227], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[84:85], v[228:229], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[86:87], v[230:231], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[88:89], v[232:233], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[220:223], %[v_os_b5], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[90:91], v[234:235], [%[c12], %[c13], %[c14], %[c15]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[92:93], v[236:237], [%[c12], %[c13], "
    "%[c14], %[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[94:95], "
    "v[238:239], [%[c12], %[c13], %[c14], %[c15]] \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[96:97], v[176:177], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[224:227], %[v_os_b6], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[98:99], v[178:179], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[100:101], v[180:181], [%[c0], %[c1], "
    "%[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[102:103], v[182:183], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[104:105], v[184:185], [%[c0], %[c1], %[c2], %[c3]] \n"
    "  buffer_load_dwordx4  acc[228:231], %[v_os_b6], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[106:107], v[186:187], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_ "  [%[c0], %[c1], %[c2], %[c3]], acc[108:109], v[188:189], [%[c0], %[c1], "
    "%[c2], %[c3]] \n" _UK_MFMA_
    "  [%[c0], %[c1], %[c2], %[c3]], acc[110:111], v[190:191], [%[c0], %[c1], %[c2], %[c3]] "
    "\n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[96:97], v[240:241], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[232:235], %[v_os_b6], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[98:99], v[242:243], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[100:101], v[244:245], [%[c4], %[c5], "
    "%[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[102:103], v[246:247], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[104:105], v[248:249], [%[c4], %[c5], %[c6], %[c7]] \n"
    "  buffer_load_dwordx4  acc[236:239], %[v_os_b6], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[106:107], v[250:251], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_ "  [%[c4], %[c5], %[c6], %[c7]], acc[108:109], v[252:253], [%[c4], %[c5], "
    "%[c6], %[c7]] \n" _UK_MFMA_
    "  [%[c4], %[c5], %[c6], %[c7]], acc[110:111], v[254:255], [%[c4], %[c5], %[c6], %[c7]] "
    "\n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[112:113], v[176:177], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[240:243], %[v_os_b7], s[12:15], 0 offen \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[114:115], v[178:179], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[116:117], v[180:181], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[118:119], v[182:183], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[120:121], v[184:185], [%[c8], %[c9], %[c10], %[c11]] \n"
    "  buffer_load_dwordx4  acc[244:247], %[v_os_b7], s[12:15], 0 offen offset:1024 \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[122:123], v[186:187], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c8], %[c9], %[c10], %[c11]], acc[124:125], v[188:189], [%[c8], %[c9], "
    "%[c10], %[c11]] \n" _UK_MFMA_
    "  [%[c8], %[c9], %[c10], %[c11]], acc[126:127], v[190:191], [%[c8], %[c9], %[c10], %[c11]] "
    "\n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[112:113], v[240:241], [%[c12], %[c13], "
    "%[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[248:251], %[v_os_b7], s[12:15], 0 offen offset:2048 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[114:115], v[242:243], [%[c12], %[c13], %[c14], "
    "%[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[116:117], v[244:245], [%[c12], "
    "%[c13], %[c14], %[c15]] \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[118:119], v[246:247], [%[c12], %[c13], %[c14], "
    "%[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[120:121], v[248:249], [%[c12], "
    "%[c13], %[c14], %[c15]] \n"
    "  buffer_load_dwordx4  acc[252:255], %[v_os_b7], s[12:15], 0 offen offset:3072 \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[122:123], v[250:251], [%[c12], %[c13], %[c14], "
    "%[c15]] \n" _UK_MFMA_ "  [%[c12], %[c13], %[c14], %[c15]], acc[124:125], v[252:253], [%[c12], "
    "%[c13], %[c14], %[c15]] \n" _UK_MFMA_
    "  [%[c12], %[c13], %[c14], %[c15]], acc[126:127], v[254:255], [%[c12], %[c13], %[c14], "
    "%[c15]]\n"
    "  v_mul_f32     %[c0], %[scale_0], %[c0]                            \n"
    "  v_mul_f32     %[c1], %[scale_0], %[c1]                            \n"
    "  v_mul_f32     %[c2], %[scale_0], %[c2]                            \n"
    "  v_mul_f32     %[c3], %[scale_0], %[c3]                            \n"
    "  v_mul_f32     %[c4], %[scale_1], %[c4]                            \n"
    "  v_mul_f32     %[c5], %[scale_1], %[c5]                            \n"
    "  v_mul_f32     %[c6], %[scale_1], %[c6]                            \n"
    "  v_mul_f32     %[c7], %[scale_1], %[c7]                            \n"
    "  v_mul_f32     %[c8], %[scale_0], %[c8]                            \n"
    "  v_mul_f32     %[c9], %[scale_0], %[c9]                            \n"
    "  v_mul_f32     %[c10], %[scale_0], %[c10]                            \n"
    "  v_mul_f32     %[c11], %[scale_0], %[c11]                            \n"
    "  v_mul_f32     %[c12], %[scale_1], %[c12]                            \n"
    "  v_mul_f32     %[c13], %[scale_1], %[c13]                            \n"
    "  v_mul_f32     %[c14], %[scale_1], %[c14]                            \n"
    "  v_mul_f32     %[c15], %[scale_1], %[c15]                            \n" _UK_PK_CVT_(
        "%[c0]", "%[c1]", "%[c0]") _UK_PK_CVT_("%[c2]", "%[c3]", "%[c1]")
        _UK_PK_CVT_("%[c4]", "%[c5]", "%[c2]") _UK_PK_CVT_("%[c6]", "%[c7]", "%[c3]") _UK_PK_CVT_(
            "%[c8]", "%[c9]", "%[c4]") _UK_PK_CVT_("%[c10]", "%[c11]", "%[c5]")
            _UK_PK_CVT_("%[c12]", "%[c13]", "%[c6]") _UK_PK_CVT_(
                "%[c14]",
                "%[c15]",
                "%[c7]") "  ;------------------------------  \n"
                         "  ds_write_b64  %[v_sfl_sst], [%[c0],%[c1]] offset:0    + %[shfl_base]   "
                         "            \n"
                         "  ds_write_b64  %[v_sfl_sst], [%[c2],%[c3]] offset:4352 + %[shfl_base]   "
                         "            \n"
                         "  ds_write_b64  %[v_sfl_sst], [%[c4],%[c5]] offset:2176 + %[shfl_base]   "
                         "            \n"
                         "  ds_write_b64  %[v_sfl_sst], [%[c6],%[c7]] offset:6528 + %[shfl_base]   "
                         "            \n"
                         "  s_waitcnt     lgkmcnt(0)                              \n"
                         "  s_barrier                                             \n"
                         "  ds_read_b32   %[c0], %[v_sfl_sld] offset:0    + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c1], %[v_sfl_sld] offset:32   + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c2], %[v_sfl_sld] offset:64   + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c3], %[v_sfl_sld] offset:96   + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c4], %[v_sfl_sld] offset:4352 + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c5], %[v_sfl_sld] offset:4384 + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c6], %[v_sfl_sld] offset:4416 + %[shfl_base]           "
                         "         \n"
                         "  ds_read_b32   %[c7], %[v_sfl_sld] offset:4448 + %[shfl_base]           "
                         "         \n"
                         "  s_waitcnt     lgkmcnt(0)                              \n"
                         "  s_mov_b64     exec, %[s_execflag_0]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o0], %[c0], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_1]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o1], %[c1], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_2]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o2], %[c2], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_3]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o3], %[c3], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_4]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o4], %[c4], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_5]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o5], %[c5], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_6]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o6], %[c6], s[8:9]  \n"
                         "  s_mov_b64     exec, %[s_execflag_7]                    "
                         "\n" _UK_ATOMIC_ADD_ "   %[v_os_o7], %[c7], s[8:9]  \n"
                         "  s_mov_b64     exec, s[38:39]                           \n"
                         "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1     ; k--      \n"
                         "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
                         "  s_cbranch_scc0 L_end%=                                       \n"
                         "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
                         "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
                         "  s_add_u32     s12, s86, s12                                  \n"
                         "  s_addc_u32    s13, 0, s13                                    \n"
                         "  s_add_u32     s8, %[s_tile_os_o], s8                             \n"
                         "  s_addc_u32    s9, 0, s9                               \n"
                         "  s_waitcnt     vmcnt(32)                               \n"
                         "  s_barrier                                             \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[128:129], v[128:129], 0 \n"
                         "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c16],%[c17],%[c18],%[c19]], acc[130:131], "
                         "v[130:131], [%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[132:133], v[132:133], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[134:135], v[134:135], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[136:137], v[136:137], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[138:139], v[138:139], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[140:141], v[140:141], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[142:143], v[142:143], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[128:129], v[192:193], 0 \n"
                         "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[130:131], v[194:195], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[132:133], v[196:197], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[134:135], v[198:199], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[136:137], v[200:201], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[138:139], v[202:203], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[140:141], v[204:205], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[142:143], v[206:207], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[144:145], v[128:129], 0 \n"
                         "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c24],%[c25],%[c26],%[c27]], acc[146:147], "
                         "v[130:131], [%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[148:149], v[132:133], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[150:151], v[134:135], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[152:153], v[136:137], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[154:155], v[138:139], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[156:157], v[140:141], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[158:159], v[142:143], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[144:145], v[192:193], 0 \n"
                         "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[146:147], v[194:195], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[148:149], v[196:197], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[150:151], v[198:199], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[152:153], v[200:201], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[154:155], v[202:203], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[156:157], v[204:205], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[158:159], v[206:207], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[160:161], v[144:145], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c16],%[c17],%[c18],%[c19]], acc[162:163], "
                         "v[146:147], [%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[164:165], v[148:149], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[166:167], v[150:151], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[168:169], v[152:153], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[170:171], v[154:155], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[172:173], v[156:157], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[174:175], v[158:159], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[160:161], v[208:209], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[162:163], v[210:211], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[164:165], v[212:213], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[166:167], v[214:215], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[168:169], v[216:217], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[170:171], v[218:219], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[172:173], v[220:221], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[174:175], v[222:223], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[176:177], v[144:145], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c24],%[c25],%[c26],%[c27]], acc[178:179], "
                         "v[146:147], [%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[180:181], v[148:149], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[182:183], v[150:151], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[184:185], v[152:153], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[186:187], v[154:155], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[188:189], v[156:157], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[190:191], v[158:159], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[176:177], v[208:209], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[178:179], v[210:211], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[180:181], v[212:213], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[182:183], v[214:215], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[184:185], v[216:217], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[186:187], v[218:219], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[188:189], v[220:221], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[190:191], v[222:223], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[192:193], v[160:161], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c16],%[c17],%[c18],%[c19]], acc[194:195], "
                         "v[162:163], [%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[196:197], v[164:165], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[198:199], v[166:167], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[200:201], v[168:169], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[202:203], v[170:171], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[204:205], v[172:173], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[206:207], v[174:175], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[192:193], v[224:225], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[194:195], v[226:227], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[196:197], v[228:229], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[198:199], v[230:231], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[200:201], v[232:233], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[202:203], v[234:235], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[204:205], v[236:237], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[206:207], v[238:239], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[208:209], v[160:161], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c24],%[c25],%[c26],%[c27]], acc[210:211], "
                         "v[162:163], [%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[212:213], v[164:165], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[214:215], v[166:167], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[216:217], v[168:169], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[218:219], v[170:171], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[220:221], v[172:173], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[222:223], v[174:175], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[208:209], v[224:225], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[210:211], v[226:227], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[212:213], v[228:229], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[214:215], v[230:231], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[216:217], v[232:233], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[218:219], v[234:235], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[220:221], v[236:237], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[222:223], v[238:239], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[224:225], v[176:177], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c16],%[c17],%[c18],%[c19]], acc[226:227], "
                         "v[178:179], [%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[228:229], v[180:181], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[230:231], v[182:183], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[232:233], v[184:185], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n"
                         "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[234:235], v[186:187], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[236:237], v[188:189], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c16],%[c17],%[c18],%[c19]], acc[238:239], v[190:191], "
                         "[%[c16],%[c17],%[c18],%[c19]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[224:225], v[240:241], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[226:227], v[242:243], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[228:229], v[244:245], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[230:231], v[246:247], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[232:233], v[248:249], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n"
                         "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[234:235], v[250:251], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[236:237], v[252:253], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c20],%[c21],%[c22],%[c23]], acc[238:239], v[254:255], "
                         "[%[c20],%[c21],%[c22],%[c23]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[240:241], v[176:177], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[12:15], 0 offen "
                         "\n" _UK_MFMA_ "  [%[c24],%[c25],%[c26],%[c27]], acc[242:243], "
                         "v[178:179], [%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[244:245], v[180:181], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[246:247], v[182:183], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[248:249], v[184:185], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n"
                         "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[12:15], 0 offen "
                         "offset:1024 \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[250:251], v[186:187], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[252:253], v[188:189], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c24],%[c25],%[c26],%[c27]], acc[254:255], v[190:191], "
                         "[%[c24],%[c25],%[c26],%[c27]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[240:241], v[240:241], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[12:15], 0 offen "
                         "offset:2048 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[242:243], v[242:243], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[244:245], v[244:245], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[246:247], v[246:247], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[248:249], v[248:249], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n"
                         "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[12:15], 0 offen "
                         "offset:3072 \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[250:251], v[250:251], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[252:253], v[252:253], "
                         "[%[c28],%[c29],%[c30],%[c31]] \n" _UK_MFMA_
                         "  [%[c28],%[c29],%[c30],%[c31]], acc[254:255], v[254:255], "
                         "[%[c28],%[c29],%[c30],%[c31]]\n"
                         "  v_mul_f32     %[c16], %[scale_0], %[c16]                            \n"
                         "  v_mul_f32     %[c17], %[scale_0], %[c17]                            \n"
                         "  v_mul_f32     %[c18], %[scale_0], %[c18]                            \n"
                         "  v_mul_f32     %[c19], %[scale_0], %[c19]                            \n"
                         "  v_mul_f32     %[c20], %[scale_1], %[c20]                            \n"
                         "  v_mul_f32     %[c21], %[scale_1], %[c21]                            \n"
                         "  v_mul_f32     %[c22], %[scale_1], %[c22]                            \n"
                         "  v_mul_f32     %[c23], %[scale_1], %[c23]                            \n"
                         "  v_mul_f32     %[c24], %[scale_0], %[c24]                            \n"
                         "  v_mul_f32     %[c25], %[scale_0], %[c25]                            \n"
                         "  v_mul_f32     %[c26], %[scale_0], %[c26]                            \n"
                         "  v_mul_f32     %[c27], %[scale_0], %[c27]                            \n"
                         "  v_mul_f32     %[c28], %[scale_1], %[c28]                            \n"
                         "  v_mul_f32     %[c29], %[scale_1], %[c29]                            \n"
                         "  v_mul_f32     %[c30], %[scale_1], %[c30]                            \n"
                         "  v_mul_f32     %[c31], %[scale_1], %[c31]                            \n"

    _UK_PK_CVT_("%[c16]", "%[c17]", "%[c16]") _UK_PK_CVT_("%[c18]", "%[c19]", "%[c17]") _UK_PK_CVT_(
        "%[c20]", "%[c21]", "%[c18]") _UK_PK_CVT_("%[c22]", "%[c23]", "%[c19]")
        _UK_PK_CVT_("%[c24]", "%[c25]", "%[c20]") _UK_PK_CVT_(
            "%[c26]", "%[c27]", "%[c21]") _UK_PK_CVT_("%[c28]",
                                                      "%[c29]",
                                                      "%[c22]") _UK_PK_CVT_("%[c30]",
                                                                            "%[c31]",
                                                                            "%[c23]")

            "  ;------------------------------  \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c16],%[c17]] offset:0    + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c18],%[c19]] offset:4352 + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c20],%[c21]] offset:2176 + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c22],%[c23]] offset:6528 + %[shfl_base]         \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            "  s_barrier                                             \n"
            "  ds_read_b32   %[c16], %[v_sfl_sld] offset:0    + %[shfl_base]                  \n"
            "  ds_read_b32   %[c17], %[v_sfl_sld] offset:32   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c18], %[v_sfl_sld] offset:64   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c19], %[v_sfl_sld] offset:96   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c20], %[v_sfl_sld] offset:4352 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c21], %[v_sfl_sld] offset:4384 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c22], %[v_sfl_sld] offset:4416 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c23], %[v_sfl_sld] offset:4448 + %[shfl_base]                  \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            "  s_mov_b64     exec, %[s_execflag_0]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o0], %[c16], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_1]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o1], %[c17], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_2]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o2], %[c18], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_3]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o3], %[c19], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_4]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o4], %[c20], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_5]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o5], %[c21], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_6]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o6], %[c22], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_7]                    \n" _UK_ATOMIC_ADD_
            "   %[v_os_o7], %[c23], s[8:9]  \n"
            "  s_mov_b64     exec, s[38:39]                           \n"
            "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1     ; k--      \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
            "  s_cbranch_scc0 L_end%=                                       \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s12, s86, s12                                  \n"
            "  s_addc_u32    s13, 0, s13                                    \n"
            "  s_add_u32     s8, %[s_tile_os_o], s8                             \n"
            "  s_addc_u32    s9, 0, s9                               \n"
            "  s_branch      L_start%=          \n"
            "L_end%=:                                                \n"

#undef _UK_MFMA_
#undef _UK_PK_CVT_
#undef _UK_ATOMIC_ADD_
#undef CK_TILE_FLATMM_UK_MFMA
    // clang-format on
