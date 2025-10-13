import os
import re
import sys
import time
import pytest

from libs import utils
from libs.utils import log


class TestRock:
	''' This is an Pytest Test Suite Class to test various components of TheRock '''

	def test_hipcub(self, orch, rock, result):
		''' A Test case to verify hipcub '''
		result.testVerdict = orch.runCtest(cwd=f'{rock}/bin/hipcub')
		assert result.testVerdict


	def test_rocprim(self, orch, rock, result):
		''' A Test case to verify rocprim '''
		result.testVerdict = orch.runCtest(cwd=f'{rock}/bin/rocprim')
		assert result.testVerdict


	def test_rocrand(self, orch, rock, result):
		''' A Test case to verify rocrand '''
		result.testVerdict = orch.runCtest(cwd=f'{rock}/bin/rocRAND')
		assert result.testVerdict


	def test_rocthrust(self, orch, rock, result):
		''' A Test case to verify rocthrust '''
		result.testVerdict = orch.runCtest(cwd=f'{rock}/bin/rocthrust')
		assert result.testVerdict
