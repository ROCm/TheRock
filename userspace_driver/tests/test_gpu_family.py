"""Tests for GPU family registry."""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, get_gpu_family, get_all_families


class TestGPUFamilyConfig:
    """Test GPUFamilyConfig dataclass."""

    def test_gfx_target_version(self):
        config = GPUFamilyConfig(
            name="test", architecture="TEST", gfx_version=(9, 4, 2)
        )
        # 9 * 10000 + 4 * 100 + 2 = 90402
        assert config.gfx_target_version == 90402

    def test_gfx_target_version_rdna3(self):
        config = GPUFamilyConfig(
            name="test", architecture="TEST", gfx_version=(11, 0, 0)
        )
        assert config.gfx_target_version == 110000

    def test_wave_size_default(self):
        config = GPUFamilyConfig(
            name="test", architecture="TEST", gfx_version=(9, 0, 0)
        )
        assert config.wave_size == 64

    def test_frozen(self):
        config = GPUFamilyConfig(
            name="test", architecture="TEST", gfx_version=(9, 0, 0)
        )
        try:
            config.name = "changed"
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestGPUFamilyRegistry:
    """Test GPU family registration and lookup."""

    def test_cdna3_gfx942(self):
        # Import triggers registration
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(90402)
        assert config is not None
        assert config.name == "gfx942"
        assert config.architecture == "CDNA3"
        assert config.wave_size == 64
        assert config.gfx_version == (9, 4, 2)

    def test_cdna2_gfx90a(self):
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(90010)
        assert config is not None
        assert config.name == "gfx90a"
        assert config.architecture == "CDNA2"
        assert config.wave_size == 64

    def test_rdna2_gfx1030(self):
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(100300)
        assert config is not None
        assert config.name == "gfx1030"
        assert config.architecture == "RDNA2"
        assert config.wave_size == 32

    def test_rdna3_gfx1100(self):
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(110000)
        assert config is not None
        assert config.name == "gfx1100"
        assert config.architecture == "RDNA3"
        assert config.wave_size == 32

    def test_rdna4_gfx1200(self):
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(120000)
        assert config is not None
        assert config.name == "gfx1200"
        assert config.architecture == "RDNA4"
        assert config.wave_size == 32

    def test_unknown_family(self):
        import amd_gpu_driver.gpu  # noqa: F401

        config = get_gpu_family(999999)
        assert config is None

    def test_all_families_registered(self):
        import amd_gpu_driver.gpu  # noqa: F401

        families = get_all_families()
        # At minimum: 1 CDNA2 + 1 CDNA3 + 7 RDNA2 + 4 RDNA3 + 2 RDNA4 = 15
        assert len(families) >= 15

    def test_all_families_have_required_fields(self):
        import amd_gpu_driver.gpu  # noqa: F401

        for version, config in get_all_families().items():
            assert config.name, f"Missing name for {version}"
            assert config.architecture, f"Missing architecture for {config.name}"
            assert config.gfx_version[0] > 0, f"Invalid major version for {config.name}"
            assert config.wave_size in (32, 64), f"Invalid wave_size for {config.name}"
            assert config.eop_buffer_size > 0, f"Invalid eop_buffer_size for {config.name}"


class TestGFXVersionParsing:
    """Test gfx version tuple encoding/decoding."""

    def test_roundtrip(self):
        for major, minor, stepping in [
            (9, 0, 10), (9, 4, 2), (10, 3, 0), (11, 0, 0), (12, 0, 1),
        ]:
            config = GPUFamilyConfig(
                name="test",
                architecture="TEST",
                gfx_version=(major, minor, stepping),
            )
            encoded = config.gfx_target_version
            decoded_major = encoded // 10000
            decoded_minor = (encoded // 100) % 100
            decoded_step = encoded % 100
            assert (decoded_major, decoded_minor, decoded_step) == (major, minor, stepping)
