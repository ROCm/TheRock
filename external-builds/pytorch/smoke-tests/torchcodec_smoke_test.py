# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for the torchcodec wheel.

torchcodec is optional, so every test is skipped when it is not installed.
torchcodec loads the FFmpeg shared libraries from the system at import time, so
the tests are also skipped when those libraries are not installed. The decode
test also needs the `ffmpeg` command line tool to create a small test video and
is skipped without it.
"""

import ctypes.util
import importlib.util
import shutil
import subprocess

from packaging.version import Version
import pytest

pytest.importorskip("torch")

# FFmpeg shared libraries that the torchcodec core libraries are linked against.
FFMPEG_SHARED_LIBS = ["avformat", "avcodec", "avutil", "swscale", "swresample"]


def import_torchcodec():
    if importlib.util.find_spec("torchcodec") is None:
        pytest.skip("torchcodec is not installed")
    missing = [
        name for name in FFMPEG_SHARED_LIBS if ctypes.util.find_library(name) is None
    ]
    if missing:
        pytest.skip(f"FFmpeg shared libraries are not installed: {missing}")
    # With FFmpeg present an import error is a real failure of the wheel.
    import torchcodec

    return torchcodec


class TestTorchcodecImport:
    def test_import_and_version(self):
        torchcodec = import_torchcodec()
        version = Version(torchcodec.__version__)
        print(f"torchcodec_version: {torchcodec.__version__}")
        assert version.release, "torchcodec has no release version"

    def test_decoders_available(self):
        import_torchcodec()
        from torchcodec.decoders import AudioDecoder, VideoDecoder

        assert VideoDecoder is not None
        assert AudioDecoder is not None


class TestTorchcodecDecode:
    def test_decode_small_video_on_cpu(self, tmp_path):
        import_torchcodec()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("the ffmpeg command line tool is needed to create a test video")
        from torchcodec.decoders import VideoDecoder

        width, height, frame_rate, seconds = 64, 48, 10, 1
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={seconds}:size={width}x{height}:rate={frame_rate}",
                "-c:v",
                "mpeg4",
                "-y",
                str(video),
            ],
            check=True,
        )

        decoder = VideoDecoder(str(video), device="cpu")
        assert len(decoder) == frame_rate * seconds
        frame = decoder[0]
        assert tuple(frame.shape) == (3, height, width)
        assert frame.dtype.is_floating_point is False
        assert frame.device.type == "cpu"
