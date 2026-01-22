cmake_minimum_required(VERSION 3.30)
enable_testing()
project(HIPBLASTLT_BENCHMARK_TESTS)

set (CONFIGS
	# NN_input_shapes
    "N  N  8192    320   320  1"
    "N  N  2048    640   640  1"
    "N  N   512   1280  1280  1"
    "N  N  8192    320  1280  1"
    "N  N   512  10240  1280  1"
    "N  N  2048   5120   640  1"
    "N  N  8192   2560   320  1"
    "N  N   512   1280  5120  1"
    "N  N  2048    640  2560  1"
    "N  N   154    320   768  1"
    "N  N   154   1280   768  1"
    "N  N  4096     40  4096  16"
    "N  N  1024     80  1024  16"
    "N  N  1024     80    77  16"
    # NT_input_shapes
    "N  T  4096   4096   40   16"
    "N  T  1024   1024   80   16"
    "N  T  4096     77   40   16"
    "N  T   256     77  160   16"
    "N  T  1024     77   80   16"
)

foreach(_CONFIG ${CONFIGS})
	string(REGEX REPLACE " +" ";" CONFIG "${_CONFIG}")
	list(GET CONFIG 0 TRANS_A)
	list(GET CONFIG 1 TRANS_B)
	list(GET CONFIG 2 M)
	list(GET CONFIG 3 N)
	list(GET CONFIG 4 K)
	list(GET CONFIG 5 BATCH_COUNT)
	math(EXPR STRIDE_A "${M} * ${K}")
	math(EXPR STRIDE_B "${N} * ${K}")
	math(EXPR STRIDE_C "${M} * ${N}")
	math(EXPR STRIDE_D "${M} * ${N}")
    add_test(NAME "${TRANS_A}_${TRANS_B}.${M}_${N}_${K}_${BATCH_COUNT}"
		COMMAND "$ENV{ROCKDIR}/bin/hipblaslt-bench"
			-v --transA ${TRANS_A} --transB ${TRANS_B}
			-m ${M} -n ${N} -k ${K}
			--alpha 1 --beta 0
			--lda ${M} --stride_a ${STRIDE_A}
			--ldb ${K} --stride_b ${STRIDE_B}
			--ldc ${M} --stride_c ${STRIDE_C}
			--ldd ${M} --stride_d ${STRIDE_D}
			--precision f16_r
			--compute_type f32_r
			--activation_type none
			--iters 1000
			--cold_iters 1000
			--batch_count ${BATCH_COUNT}
	)
endforeach()
