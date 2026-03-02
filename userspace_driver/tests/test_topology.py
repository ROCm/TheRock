"""Tests for KFD topology parser."""

import tempfile
from pathlib import Path

from amd_gpu_driver.topology import GPUNode, MemoryBank, discover_gpu_nodes, _parse_properties


class TestParseProperties:
    """Test key-value property file parsing."""

    def test_basic_properties(self, tmp_path):
        props_file = tmp_path / "properties"
        props_file.write_text(
            "gpu_id 42\n"
            "gfx_target_version 90402\n"
            "simd_count 304\n"
            "vendor_id 4098\n"
            "device_id 29832\n"
        )
        props = _parse_properties(props_file)
        assert props["gpu_id"] == 42
        assert props["gfx_target_version"] == 90402
        assert props["simd_count"] == 304
        assert props["vendor_id"] == 4098

    def test_missing_file(self, tmp_path):
        props = _parse_properties(tmp_path / "nonexistent")
        assert props == {}

    def test_malformed_values_skipped(self, tmp_path):
        props_file = tmp_path / "properties"
        props_file.write_text(
            "gpu_id 42\n"
            "name not_a_number\n"
            "simd_count 100\n"
        )
        props = _parse_properties(props_file)
        assert "gpu_id" in props
        assert "name" not in props
        assert "simd_count" in props


class TestGPUNode:
    """Test GPUNode properties."""

    def test_is_gpu(self):
        node = GPUNode(simd_count=304, gfx_target_version=90402)
        assert node.is_gpu is True

    def test_is_not_gpu_cpu(self):
        node = GPUNode(simd_count=0, gfx_target_version=0)
        assert node.is_gpu is False

    def test_gfx_version_tuple(self):
        node = GPUNode(gfx_target_version=90402)
        assert node.gfx_version_tuple == (9, 4, 2)

    def test_gfx_version_tuple_rdna3(self):
        node = GPUNode(gfx_target_version=110000)
        assert node.gfx_version_tuple == (11, 0, 0)

    def test_gfx_name(self):
        node = GPUNode(gfx_target_version=90402)
        assert node.gfx_name == "gfx942"

    def test_gfx_name_rdna2(self):
        node = GPUNode(gfx_target_version=100300)
        assert node.gfx_name == "gfx1030"

    def test_gfx_name_rdna3(self):
        node = GPUNode(gfx_target_version=110000)
        assert node.gfx_name == "gfx1100"

    def test_gfx_name_cdna2(self):
        node = GPUNode(gfx_target_version=90010)
        assert node.gfx_name == "gfx90a"

    def test_drm_render_path(self):
        node = GPUNode(drm_render_minor=128)
        assert node.drm_render_path == "/dev/dri/renderD128"

    def test_vram_size(self):
        node = GPUNode(
            mem_banks=[
                MemoryBank(heap_type=MemoryBank.HEAP_TYPE_FB_PUBLIC, size_in_bytes=8 * 1024**3),
                MemoryBank(heap_type=MemoryBank.HEAP_TYPE_SYSTEM, size_in_bytes=64 * 1024**3),
            ]
        )
        assert node.vram_size == 8 * 1024**3

    def test_vram_size_no_banks(self):
        node = GPUNode()
        assert node.vram_size == 0


class TestDiscoverGPUNodes:
    """Test topology discovery from sysfs."""

    def test_empty_topology(self, tmp_path):
        nodes = discover_gpu_nodes(topology_path=tmp_path)
        assert nodes == []

    def test_single_gpu_node(self, tmp_path):
        # Create a GPU node
        node_dir = tmp_path / "0"
        node_dir.mkdir()
        props = node_dir / "properties"
        props.write_text(
            "cpu_cores_count 0\n"
            "simd_count 304\n"
            "gpu_id 42\n"
            "gfx_target_version 90402\n"
            "vendor_id 4098\n"
            "device_id 29832\n"
            "drm_render_minor 128\n"
            "max_waves_per_simd 8\n"
            "num_sdma_engines 2\n"
        )
        # Create memory bank
        mem_dir = node_dir / "mem_banks" / "0"
        mem_dir.mkdir(parents=True)
        mem_props = mem_dir / "properties"
        mem_props.write_text(
            "heap_type 1\n"
            "size_in_bytes 68719476736\n"
        )

        nodes = discover_gpu_nodes(topology_path=tmp_path)
        assert len(nodes) == 1

        gpu = nodes[0]
        assert gpu.gpu_id == 42
        assert gpu.gfx_target_version == 90402
        assert gpu.simd_count == 304
        assert gpu.drm_render_minor == 128
        assert gpu.is_gpu is True
        assert len(gpu.mem_banks) == 1
        assert gpu.vram_size == 68719476736

    def test_cpu_nodes_filtered(self, tmp_path):
        # CPU node (no SIMD, no gfx version)
        cpu_dir = tmp_path / "0"
        cpu_dir.mkdir()
        (cpu_dir / "properties").write_text(
            "cpu_cores_count 128\n"
            "simd_count 0\n"
            "gpu_id 0\n"
            "gfx_target_version 0\n"
        )

        # GPU node
        gpu_dir = tmp_path / "1"
        gpu_dir.mkdir()
        (gpu_dir / "properties").write_text(
            "cpu_cores_count 0\n"
            "simd_count 304\n"
            "gpu_id 42\n"
            "gfx_target_version 90402\n"
        )

        nodes = discover_gpu_nodes(topology_path=tmp_path)
        assert len(nodes) == 1
        assert nodes[0].gpu_id == 42

    def test_multiple_gpu_nodes(self, tmp_path):
        for i in range(3):
            node_dir = tmp_path / str(i)
            node_dir.mkdir()
            (node_dir / "properties").write_text(
                f"simd_count 304\n"
                f"gpu_id {100 + i}\n"
                f"gfx_target_version 90402\n"
            )

        nodes = discover_gpu_nodes(topology_path=tmp_path)
        assert len(nodes) == 3
        gpu_ids = [n.gpu_id for n in nodes]
        assert gpu_ids == [100, 101, 102]
