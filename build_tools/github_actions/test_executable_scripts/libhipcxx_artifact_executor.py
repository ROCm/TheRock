# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""lit executor that makes Windows test EXEs load the artifact HIP runtime.

Windows searches the executable's own directory before System32, and only
checks PATH last. Running each test through a hardlink in the artifact bin
makes that bin the application directory, so its amdhip64_7.dll wins over
C:\\Windows\\System32\\amdhip64_7.dll.

Selected via -DLIBCUDACXX_EXECUTOR; lit imports this module, so it must be on
PYTHONPATH.
"""

import contextlib
import os
import shutil
import uuid

from libcudacxx.test.executor import LocalExecutor


class ArtifactBinExecutor(LocalExecutor):
    def __init__(self, bin_dir, timeout=0):
        super().__init__()
        if not os.path.isdir(bin_dir):
            raise FileNotFoundError(f"Artifact bin dir not found: {bin_dir}")
        self.bin_dir = bin_dir
        # lit only applies LIBCUDACXX_TEST_TIMEOUT to its default executor.
        self.timeout = timeout

    def run(self, exe_path, cmd=None, work_dir=".", file_deps=None, env=None):
        cmd = cmd or [exe_path]
        staged = os.path.join(self.bin_dir, f"lit-{uuid.uuid4().hex}.exe")
        try:
            os.link(exe_path, staged)
        except OSError:
            # Hardlinks fail across volumes.
            shutil.copy2(exe_path, staged)
        try:
            staged_cmd = [staged if arg == exe_path else arg for arg in cmd]
            _, out, err, rc = super().run(staged, staged_cmd, work_dir, file_deps, env)
            # Report the real test executable, not the deleted staged copy.
            return cmd, out, err, rc
        finally:
            with contextlib.suppress(OSError):
                os.remove(staged)
