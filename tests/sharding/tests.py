import os
import re
import sys
import time
import pytest

from libs import utils
from libs.utils import log


class TestSuite:

	def test_rocminfo(self, orch, rock, result):
		exprList = (
			r'ROCk module.*? is loaded',
			r'Name:\s+gfx',
			r'Vendor Name:\s+AMD',
			r'Device Type:\s+GPU',
			r'L2:\s+.*? KB',
		)
		ret, out = orch.runBinary('./bin/rocminfo', cwd=rock, out=True)
		result.testVerdict = all((bool(ret == 0), *[
			re.search(expr, out) or log(f'Expr Not Match: {expr}') for expr in exprList
		]))
		assert result.testVerdict


	def test_roctracer(self, orch, rock, result):
		result.testVerdict = orch.runBinary('./run_tests.sh', cwd=f'{rock}/share/roctracer') == 0
		assert result.testVerdict


	def test_hipcub(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runCtest(env=ompEnv, cwd=f'{rock}/bin/hipcub')
		assert result.testVerdict


	def test_rocprim(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runCtest(env=ompEnv, cwd=f'{rock}/bin/rocprim')
		assert result.testVerdict


	def test_rocrand(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runCtest(env=ompEnv, cwd=f'{rock}/bin/rocRAND')
		assert result.testVerdict


	def test_rocthrust(self, orch, rock, result):
		result.testVerdict = orch.runCtest(cwd=f'{rock}/bin/rocthrust')
		assert result.testVerdict


	def test_hipblas(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./hipblas-test', env=ompEnv, cwd=f'{rock}/bin')
		assert result.testVerdict


	def test_hipblaslt(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./hipblaslt-test', env=ompEnv, cwd=f'{rock}/bin')
		assert result.testVerdict


	def test_rocblas(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./rocblas-test',
			gfilter='*quick*:*pre_checkin*-*known_bug*',
			env=ompEnv,
			cwd=f'{rock}/bin',
		)
		assert result.testVerdict


	def test_hipsolver(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./hipsolver-test',
			gfilter='*float_complex*-*known_bug*',
			env=ompEnv,
			cwd=f'{rock}/bin',
		)
		assert result.testVerdict


	def test_rocsolver(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./rocsolver-test', env=ompEnv, cwd=f'{rock}/bin')
		assert result.testVerdict


	def test_rocsparse(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./rocsparse-test',
			gfilter='*quick*',
			env={'ROCSPARSE_CLIENTS_MATRICES_DIR': f'{rock}/clients/matrices', **ompEnv},
			cwd=f'{rock}/bin',
		)
		assert result.testVerdict


	def test_hipsparse(self, orch, ompEnv, rock, result):
		result.testVerdict = orch.runGtest('./hipsparse-test',
			env={'HIPSPARSE_CLIENTS_MATRICES_DIR': f'{rock}/clients/matrices', **ompEnv},
			cwd=f'{rock}/bin',
		)
		assert result.testVerdict


	def test_rccl(self, orch, rock, result):
		result.testVerdict = orch.runGtest('./rccl-UnitTests', cwd=f'{rock}/bin')
		assert result.testVerdict


	def test_miopen(self, orch, rock, result):
		result.testVerdict = orch.runGtest('./miopen_gtest',
			gfilter='Smoke*',
			cwd=f'{rock}/bin',
		)
		assert result.testVerdict
