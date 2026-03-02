"""Parse /sys/devices/virtual/kfd/kfd/topology to enumerate GPU nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KFD_TOPOLOGY_PATH = Path("/sys/devices/virtual/kfd/kfd/topology/nodes")


@dataclass
class MemoryBank:
    """A memory bank (VRAM or GTT) on a GPU node."""

    heap_type: int = 0
    size_in_bytes: int = 0

    # heap_type constants
    HEAP_TYPE_SYSTEM = 0
    HEAP_TYPE_FB_PUBLIC = 1
    HEAP_TYPE_FB_PRIVATE = 2
    HEAP_TYPE_GPU_GDS = 3
    HEAP_TYPE_GPU_LDS = 4
    HEAP_TYPE_GPU_SCRATCH = 5


@dataclass
class GPUNode:
    """A GPU node discovered from KFD topology."""

    node_id: int = 0
    gpu_id: int = 0
    gfx_target_version: int = 0
    simd_count: int = 0
    vendor_id: int = 0
    device_id: int = 0
    drm_render_minor: int = -1
    max_waves_per_simd: int = 0
    num_sdma_engines: int = 0
    num_sdma_xgmi_engines: int = 0
    num_cp_queues: int = 0
    max_engine_clk_ccompute: int = 0
    local_mem_size: int = 0
    fw_version: int = 0
    capability: int = 0
    num_xcc: int = 1
    simd_per_cu: int = 4
    array_count: int = 1
    simd_arrays_per_engine: int = 1
    lds_size_in_kb: int = 64
    mem_banks: list[MemoryBank] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        """True if this node is a GPU (not a CPU)."""
        return self.simd_count > 0 and self.gfx_target_version > 0

    @property
    def gfx_version_tuple(self) -> tuple[int, int, int]:
        """Parse gfx_target_version into (major, minor, stepping).

        The kernel reports gfx_target_version in decimal-packed format:
        major * 10000 + minor * 100 + stepping.
        E.g. gfx942 = 90402, gfx1100 = 110000, gfx90a = 90010.
        """
        v = self.gfx_target_version
        return (v // 10000, (v // 100) % 100, v % 100)

    @property
    def gfx_name(self) -> str:
        """Return the GFX target name like 'gfx942'."""
        major, minor, stepping = self.gfx_version_tuple
        return f"gfx{major}{minor:x}{stepping:x}"

    @property
    def drm_render_path(self) -> str:
        """Path to the DRM render device."""
        return f"/dev/dri/renderD{self.drm_render_minor}"

    @property
    def vram_size(self) -> int:
        """Total VRAM size in bytes."""
        return sum(
            mb.size_in_bytes
            for mb in self.mem_banks
            if mb.heap_type in (MemoryBank.HEAP_TYPE_FB_PUBLIC, MemoryBank.HEAP_TYPE_FB_PRIVATE)
        )


def _parse_properties(path: Path) -> dict[str, int]:
    """Parse a KFD topology properties file into key-value pairs."""
    props: dict[str, int] = {}
    if not path.exists():
        return props
    text = path.read_text()
    for line in text.strip().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return props


def _parse_mem_banks(node_path: Path) -> list[MemoryBank]:
    """Parse memory banks for a node."""
    banks: list[MemoryBank] = []
    mem_path = node_path / "mem_banks"
    if not mem_path.exists():
        return banks
    for bank_dir in sorted(mem_path.iterdir()):
        if not bank_dir.is_dir():
            continue
        props = _parse_properties(bank_dir / "properties")
        bank = MemoryBank(
            heap_type=props.get("heap_type", 0),
            size_in_bytes=props.get("size_in_bytes", 0),
        )
        banks.append(bank)
    return banks


def compute_queue_sizes(node: GPUNode) -> dict[str, int]:
    """Compute ctl_stack_size, cwsr_size, eop_buffer_size, debug_memory_size.

    Replicates the kernel's kfd_queue_ctx_save_restore_size() logic from
    drivers/gpu/drm/amd/amdkfd/kfd_queue.c.
    """
    gfxv = node.gfx_target_version
    if gfxv < 80001:
        return {"ctl_stack_size": 0, "cwsr_size": 0, "eop_buffer_size": 0, "debug_memory_size": 0}

    PAGE_SIZE = 4096

    def align_up(x: int, a: int) -> int:
        return (x + a - 1) & ~(a - 1)

    # VGPR size per CU (kernel: kfd_get_vgpr_size_per_cu)
    if gfxv in (90402, 90010, 90008, 90500):
        vgpr_size = 0x80000
    elif gfxv in (110000, 110001, 120000, 120001):
        vgpr_size = 0x60000
    else:
        vgpr_size = 0x40000

    SGPR_SIZE_PER_CU = 0x4000
    LDS_SIZE_PER_CU = 0x10000
    HWREG_SIZE_PER_CU = 0x1000

    if gfxv == 90500:
        lds_size = node.lds_size_in_kb * 1024
    else:
        lds_size = LDS_SIZE_PER_CU

    wg_data_per_cu = vgpr_size + SGPR_SIZE_PER_CU + lds_size + HWREG_SIZE_PER_CU

    simd_per_cu = node.simd_per_cu if node.simd_per_cu > 0 else 4
    num_xcc = node.num_xcc if node.num_xcc > 0 else 1
    cu_num = node.simd_count // simd_per_cu // num_xcc

    if gfxv < 100100:  # Pre-RDNA
        array_count = node.array_count if node.array_count > 0 else 1
        simd_arrays_per_engine = node.simd_arrays_per_engine if node.simd_arrays_per_engine > 0 else 1
        wave_num = min(cu_num * 40, array_count // simd_arrays_per_engine * 512)
    else:
        wave_num = cu_num * 32

    wg_data_size = align_up(cu_num * wg_data_per_cu, PAGE_SIZE)

    cntl_stack_bytes_per_wave = 12 if gfxv >= 100100 else 8
    ctl_stack_size = wave_num * cntl_stack_bytes_per_wave + 8
    ctl_stack_size = align_up(40 + ctl_stack_size, PAGE_SIZE)  # 40 = SIZEOF_HSA_USER_CONTEXT_SAVE_AREA_HEADER

    # GFX10 cap
    if (gfxv // 10000 * 10000) == 100000:
        ctl_stack_size = min(ctl_stack_size, 0x7000)

    cwsr_size = ctl_stack_size + wg_data_size
    debug_memory_size = align_up(wave_num * 32, 64)

    if gfxv == 80002:  # Tonga
        eop_buffer_size = 0x8000
    else:
        eop_buffer_size = 4096

    return {
        "ctl_stack_size": ctl_stack_size,
        "cwsr_size": cwsr_size,
        "eop_buffer_size": eop_buffer_size,
        "debug_memory_size": debug_memory_size,
    }


def discover_gpu_nodes(
    topology_path: Path = KFD_TOPOLOGY_PATH,
) -> list[GPUNode]:
    """Discover all GPU nodes from KFD topology.

    Returns a list of GPUNode objects for nodes that are GPUs (not CPUs).
    """
    nodes: list[GPUNode] = []
    if not topology_path.exists():
        return nodes

    for node_dir in sorted(topology_path.iterdir()):
        if not node_dir.is_dir():
            continue
        try:
            node_id = int(node_dir.name)
        except ValueError:
            continue

        props = _parse_properties(node_dir / "properties")
        mem_banks = _parse_mem_banks(node_dir)

        node = GPUNode(
            node_id=node_id,
            gpu_id=props.get("gpu_id", 0),
            gfx_target_version=props.get("gfx_target_version", 0),
            simd_count=props.get("simd_count", 0),
            vendor_id=props.get("vendor_id", 0),
            device_id=props.get("device_id", 0),
            drm_render_minor=props.get("drm_render_minor", -1),
            max_waves_per_simd=props.get("max_waves_per_simd", 0),
            num_sdma_engines=props.get("num_sdma_engines", 0),
            num_sdma_xgmi_engines=props.get("num_sdma_xgmi_engines", 0),
            num_cp_queues=props.get("num_cp_queues", 0),
            max_engine_clk_ccompute=props.get("max_engine_clk_ccompute", 0),
            local_mem_size=props.get("local_mem_size", 0),
            fw_version=props.get("fw_version", 0),
            capability=props.get("capability", 0),
            num_xcc=props.get("num_xcc", 1),
            simd_per_cu=props.get("simd_per_cu", 4),
            array_count=props.get("array_count", 1),
            simd_arrays_per_engine=props.get("simd_arrays_per_engine", 1),
            lds_size_in_kb=props.get("lds_size_in_kb", 64),
            mem_banks=mem_banks,
        )

        if node.is_gpu:
            nodes.append(node)

    return nodes
