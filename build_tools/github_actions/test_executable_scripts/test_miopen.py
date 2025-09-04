import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

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
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
