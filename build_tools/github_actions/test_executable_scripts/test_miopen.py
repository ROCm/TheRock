import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

###########################################

positive_filter = []
negative_filter = []

# Fusion #
positive_filter.append("*Fusion*")

# Batch Normalization #
positive_filter.append("*/GPU_BNBWD*_*")
positive_filter.append("*/GPU_BNOCLBWD*_*")
positive_filter.append("*/GPU_BNFWD*_*")
positive_filter.append("*/GPU_BNOCLFWD*_*")
positive_filter.append("*/GPU_BNInfer*_*")
positive_filter.append("*/GPU_BNOCLInfer*_*")
positive_filter.append("*/GPU_bn_infer*_*")

# CPU tests
positive_filter.append("CPU_*")  # tests without a suite
positive_filter.append("*/CPU_*")  # tests with a suite

# Different
positive_filter.append("*/GPU_Cat_*")
positive_filter.append("*/GPU_ConvBiasActiv*")

# Convolutions
positive_filter.append("*/GPU_Conv*")
positive_filter.append("*/GPU_conv*")

# Solvers
positive_filter.append("*/GPU_UnitTestConv*")

negative_filter.append("*DBSync*")
negative_filter.append("*DeepBench*")
negative_filter.append("*MIOpenTestConv*")

# Temporary fails
negative_filter.append("*ConvBiasResAddActivation*")
negative_filter.append("*ConvFwdBiasResAddActiv*")
negative_filter.append("*GPU_FusionSetArg_FP16*")

gtest_final_filter_cmd = (
    "--gtest_filter=" + ":".join(positive_filter) + "-" + ":".join(negative_filter)
)

#############################################

cmd = [f"{THEROCK_BIN_DIR}/miopen_gtest", gtest_final_filter_cmd]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
