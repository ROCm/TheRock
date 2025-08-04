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

positive_filter=[]
negative_filter=[]

# Fusion #
positive_filter.append("*Fusion*")

# Batch Normalization #
positive_filter.append("*_BNBWD*_*")
positive_filter.append("*_BNOCLBWD*_*")
positive_filter.append("*_BNFWD*_*")
positive_filter.append("*_BNOCLFWD*_*")
positive_filter.append("*_BNInfer*_*")
positive_filter.append("*_BNOCLInfer*_*")
positive_filter.append("*_bn_infer*_*")

negative_filter.append("*_BN*Large*_*")
negative_filter.append("*_BN*SerialRun*_*")

gtest_final_filter_cmd = "--gtest_filter=" + ':'.join(positive_filter) + "-" + ':'.join(negative_filter)

#############################################

cmd = [f"{THEROCK_BIN_DIR}/miopen_gtest", gtest_final_filter_cmd]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
