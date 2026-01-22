#!/usr/bin/python3
import os
import json
import pytest


class TestHipBLASLT:
    """This is an Pytest Test Suite Class to test hipblaslt component of TheRock"""

    @pytest.mark.parametrize(
        argnames=('transA,transB,M,N,K,batch_count'),
        argvalues=json.load(open(f'{os.path.dirname(__file__)}/config.json'))
    )
    def test_hipblaslt_bench_pytest(self, transA, transB, M, N, K, batch_count, orch, therock_path, result):
        """A Test case to verify hipblaslt benchmark tests"""
        ret, _, _ = orch.node.runCmd(
            './hipblaslt-bench', '-v',
			'-m', str(M), '-n', str(N), '-k', str(K),
			'--alpha', '1', '--beta', '0',
			'--lda', str(M), '--stride_a', str(M*K),
			'--ldb', str(K), '--stride_b', str(N*K),
			'--ldc', str(M), '--stride_c', str(M*N),
			'--ldd', str(M), '--stride_d', str(M*N),
			'--precision', 'f16_r',
			'--compute_type', 'f32_r',
			'--activation_type', 'none',
			'--iters', '1000',
			'--cold_iters', '1000',
			'--batch_count', str(batch_count),
            cwd=f'{therock_path}/bin',
        )
        result.testVerdict = bool(ret == 0)
        assert result.testVerdict
