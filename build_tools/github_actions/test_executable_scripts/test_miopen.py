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
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

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
positive_filter.append("*/GPU_BNActivInfer_*")
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

# Misc

positive_filter.append("*/GPU_GetitemBwd*")
positive_filter.append("*/GPU_GLU_*")

positive_filter.append("*/GPU_GroupConv*")
positive_filter.append("*/GPU_GroupNorm_*")
positive_filter.append("*/GPU_GRUExtra_*")
positive_filter.append("*/GPU_TestActivation*")
positive_filter.append("*/GPU_HipBLASLtGEMMTest*")
positive_filter.append("*/GPU_KernelTuningNetTestConv*")
positive_filter.append("*/GPU_Kthvalue_*")
positive_filter.append("*/GPU_LayerNormTest*")
positive_filter.append("*/GPU_LayoutTransposeTest_*")
positive_filter.append("*/GPU_Lrn*")
positive_filter.append("*/GPU_lstm_extra*")

positive_filter.append("*/GPU_MultiMarginLoss_*")
positive_filter.append("*/GPU_ConvNonpack*")
positive_filter.append("*/GPU_PerfConfig_HipImplicitGemm*")
positive_filter.append("*/GPU_AsymPooling2d_*")
positive_filter.append("*/GPU_WidePooling2d_*")
positive_filter.append("*/GPU_PReLU_*")
positive_filter.append("*/GPU_Reduce*")
positive_filter.append("*/GPU_reduce_custom_*")
positive_filter.append("*/GPU_regression_issue_*")
positive_filter.append("*/GPU_RNNExtra_*")
positive_filter.append("*/GPU_RoPE*")
positive_filter.append("*/GPU_SoftMarginLoss*")
positive_filter.append("*/GPU_T5LayerNormTest_*")
positive_filter.append("*/GPU_Op4dTensorGenericTest_*")
positive_filter.append("*/GPU_TernaryTensorOps_*")
positive_filter.append("*/GPU_unaryTensorOps_*")
positive_filter.append("*/GPU_Transformers*")
positive_filter.append("*/GPU_TunaNetTest_*")
positive_filter.append("*/GPU_UnitTestActivationDescriptor_*")
positive_filter.append("*/GPU_FinInterfaceTest*")
positive_filter.append("*/GPU_VecAddTest_*")

#############################################

negative_filter.append("*DeepBench*")
negative_filter.append("*MIOpenTestConv*")

# Failing tests
negative_filter.append("*/GPU_KernelTuningNetTest*")
negative_filter.append("*DBSync*")
negative_filter.append("*/GPU_MIOpenDriver*")
negative_filter.append("*GPU_TestMhaFind20*")

# Failing because of  fatal error: 'rocrand/rocrand_xorwow.h' file not found
negative_filter.append("*/GPU_Bwd_Mha_*")
negative_filter.append("*/GPU_Fwd_Mha_*")
negative_filter.append("*/GPU_Softmax*")
negative_filter.append("*/GPU_Dropout*")
negative_filter.append("*/GPU_MhaBackward_*")
negative_filter.append("*/GPU_MhaForward_*")


# For sake of time saving on pre-commit step
####################################################
negative_filter.append("Full/GPU_Reduce_FP64")  # 4 min 19 sec
negative_filter.append("Full/GPU_BNOCLFWDTrainSerialRun3D_BFP16")  # 3 min 37 sec
negative_filter.append("Full/GPU_Lrn_FP32")  # 2 min 50 sec
negative_filter.append("Full/GPU_Lrn_FP16")  # 2 min 20 sec
negative_filter.append("Full/GPU_BNOCLInferSerialRun3D_BFP16")  # 2 min 19 sec
negative_filter.append("Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16")  # 1 min 55 sec
negative_filter.append("Smoke/GPU_BNOCLInferLarge2D_BFP16")  # 1 min 48 sec
negative_filter.append("Full/GPU_BNOCLBWDSerialRun3D_BFP16")  # 1 min 28 sec
negative_filter.append("Smoke/GPU_BNOCLBWDLarge2D_BFP16")  # 1 min 19 sec

negative_filter.append("Full/GPU_UnitTestActivationDescriptor_FP32")  # 1 min 23 sec
negative_filter.append("Full/GPU_UnitTestActivationDescriptor_FP16")  # 1 min 0 sec

negative_filter.append("Smoke/GPU_BNOCLBWDLargeFusedActivation2D_BFP16")  # 0 min 52 sec
negative_filter.append("Smoke/GPU_BNOCLBWDLargeFusedActivation2D_FP16")  # 0 min 49 sec

negative_filter.append("Full/GPU_ConvGrpBiasActivInfer_BFP16")  # 0 min 40 sec
negative_filter.append("Full/GPU_ConvGrpBiasActivInfer_FP32")  # 0 min 38 sec
negative_filter.append("Full/GPU_ConvGrpBiasActivInfer_FP16")  # 0 min 25 sec

negative_filter.append("Full/GPU_ConvGrpActivInfer_BFP16")  # 0 min 42 sec
negative_filter.append("Full/GPU_ConvGrpActivInfer_FP32")  # 0 min 35 sec
negative_filter.append("Full/GPU_ConvGrpActivInfer_FP16")  # 0 min 25 sec

negative_filter.append("Full/GPU_ConvGrpBiasActivInfer3D_BFP16")  # 0 min 27 sec
negative_filter.append("Full/GPU_ConvGrpBiasActivInfer3D_FP32")  # 0 min 25 sec
negative_filter.append("Full/GPU_ConvGrpBiasActivInfer3D_FP16")  # 0 min 19 sec

negative_filter.append("Full/GPU_ConvGrpActivInfer3D_BFP16")  # 0 min 27 sec
negative_filter.append("Full/GPU_ConvGrpActivInfer3D_FP32")  # 0 min 22 sec
negative_filter.append("Full/GPU_ConvGrpActivInfer3D_FP16")  # 0 min 16 sec

####################################################

gtest_final_filter_cmd = (
    "--gtest_filter=" + ":".join(positive_filter) + "-" + ":".join(negative_filter)
)

#############################################

cmd = [f"{THEROCK_BIN_DIR}/miopen_gtest", gtest_final_filter_cmd]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
