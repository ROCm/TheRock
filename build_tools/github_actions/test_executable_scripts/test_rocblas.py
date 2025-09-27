import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

PLATFORM = os.getenv("PLATFORM")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

cmd = [
    f"{THEROCK_BIN_DIR}/rocblas-test",
    f"--gtest_filter=*quick*:*pre_checkin*-*known_bug*",
]

tests_to_exclude = {
    # Related issue: https://github.com/ROCm/TheRock/issues/1605
    # observed seg faults all have dot functions, f32_c type, batch count 257, among other similarities
    "gfx94X-dcgpu": {
        "linux": [
            "_/dot_batched.blas1/quick_blas1_batched_f32_c_13000_n3_n3_257_0",
            "_/dot_batched_ex.blas1_ex/quick_blas1_batched_f32_c_f32_c_f32_c_f32_c_13000_n3_n3_257",
            "_/dot_strided_batched.blas1/quick_blas1_strided_batched_f32_c_13000_n3_39000_n3_39000_257_0",
            "_/dot_strided_batched_ex.blas1_ex/quick_blas1_strided_batched_f32_c_f32_c_f32_c_f32_c_13000_n3_39000_n3_39000_257",
            "_/dotc_batched.blas1/quick_blas1_batched_f32_c_13000_n3_n3_257_0",
            "_/dotc_batched_ex.blas1_ex/quick_blas1_batched_f32_c_f32_c_f32_c_f32_c_13000_n3_n3_257",
            "_/dotc_strided_batched.blas1/quick_blas1_strided_batched_f32_c_13000_n3_39000_n3_39000_257_0",
            "_/dotc_strided_batched_ex.blas1_ex/quick_blas1_strided_batched_f32_c_f32_c_f32_c_f32_c_13000_n3_39000_n3_39000_257",
        ]
    }
}

if AMDGPU_FAMILIES in tests_to_exclude and PLATFORM in tests_to_exclude.get(
    AMDGPU_FAMILIES, {}
):
    exclusion_list = ":".join(tests_to_exclude[AMDGPU_FAMILIES][PLATFORM])
    cmd.append(f":{exclusion_list}")

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
