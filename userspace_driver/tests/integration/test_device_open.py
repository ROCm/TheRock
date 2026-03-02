"""Integration tests for device opening and topology discovery."""

import pytest

from tests.integration.conftest import requires_gpu


@requires_gpu
class TestDeviceOpen:
    """Test opening a KFD device."""

    def test_open_default_device(self, amd_device):
        assert amd_device.name
        assert amd_device.gfx_target != "unknown"

    def test_device_has_vram(self, amd_device):
        assert amd_device.vram_size > 0

    def test_device_gfx_target(self, amd_device):
        target = amd_device.gfx_target
        assert target.startswith("gfx")

    def test_device_repr(self, amd_device):
        r = repr(amd_device)
        assert "AMDDevice" in r

    def test_context_manager(self):
        from amd_gpu_driver import AMDDevice

        with AMDDevice() as dev:
            assert dev.name
            assert dev.vram_size > 0

    def test_invalid_device_index(self):
        from amd_gpu_driver import AMDDevice, DeviceNotFoundError

        with pytest.raises(DeviceNotFoundError):
            AMDDevice(device_index=999)
