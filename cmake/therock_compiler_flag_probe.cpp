// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Trivial translation unit used by therock_probe_control_flow_guard() to ask a
// compiler whether it accepts a flag. It is only ever syntax-checked, never
// linked, so it deliberately has no includes and no main().

int therock_compiler_flag_probe() { return 0; }
