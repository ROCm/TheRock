// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocfft_profile.h"

void rocfft_profile_fixture_other_tu() {
  rocfft_build_profile::add(rocfft_build_profile::state().source_bytes, 1234);
}
