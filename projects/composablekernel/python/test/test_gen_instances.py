# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import logging
import unittest

from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_library as gen_gemm_ops_library,
)
from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_preselected as gen_gemm_ops_preselected,
)
from ck4inductor.grouped_conv_fwd.gen_instances import (
    gen_conv_ops_library as gen_conv_ops_library,
)
from ck4inductor.batched_universal_gemm.gen_instances import (
    gen_ops_library as gen_batched_gemm_ops_library,
)
from ck4inductor.ck_tile_universal_gemm.gen_instances import (
    ops as gen_ck_tile_gemm_ops_library,
)

log = logging.getLogger(__name__)


class TestGenInstances(unittest.TestCase):
    def test_gen_gemm_instances(self):
        instances = gen_gemm_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_preselected_gemm_instances(self):
        instances = gen_gemm_ops_preselected()

        log.debug("%d preselected gemm instances" % len(instances))
        self.assertTrue(instances)

    def test_gen_conv_instances(self):
        instances = gen_conv_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_gen_batched_gemm_instances(self):
        instances = gen_batched_gemm_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_gen_ck_tile_universal_gemm_instances(self):
        instances = gen_ck_tile_gemm_ops_library()

        log.debug("%d ck-tile gemm instances from library" % len(instances))
        self.assertTrue(instances)
