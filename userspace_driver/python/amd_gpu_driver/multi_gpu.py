"""MultiGPUContext: coordinator for multi-GPU P2P operations."""

from __future__ import annotations

from amd_gpu_driver.backends.base import QueueHandle, QueueType
from amd_gpu_driver.backends.kfd import KFDDevice
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
from amd_gpu_driver.device import AMDDevice
from amd_gpu_driver.memory.buffer import Buffer
from amd_gpu_driver.sync.timeline import TimelineSemaphore
from amd_gpu_driver.topology import discover_gpu_nodes


class MultiGPUContext:
    """Coordinate multiple AMD GPUs with P2P memory access and cross-GPU copies.

    Usage:
        with MultiGPUContext() as ctx:
            src = ctx.devices[0].alloc(4096, location="vram")
            dst = ctx.devices[1].alloc(4096, location="vram")
            ctx.enable_peer_access(src, ctx.devices[1])
            ctx.enable_peer_access(dst, ctx.devices[0])
            ctx.copy_peer(dst, ctx.devices[1], src, ctx.devices[0])
            ctx.synchronize_all()

    Args:
        device_indices: Specific GPU indices to open. None = all available.
        max_devices: Maximum number of GPUs to open (default 8).
    """

    def __init__(
        self,
        device_indices: list[int] | None = None,
        *,
        max_devices: int = 8,
    ) -> None:
        self._devices: list[AMDDevice] = []
        self._xgmi_queues: dict[int, QueueHandle] = {}
        self._timelines: dict[int, TimelineSemaphore] = {}

        if device_indices is None:
            nodes = discover_gpu_nodes()
            device_indices = list(range(min(len(nodes), max_devices)))

        for idx in device_indices:
            dev = AMDDevice(device_index=idx)
            self._devices.append(dev)

    @property
    def devices(self) -> list[AMDDevice]:
        """All opened GPU devices."""
        return list(self._devices)

    @property
    def num_devices(self) -> int:
        """Number of opened GPUs."""
        return len(self._devices)

    def device(self, index: int) -> AMDDevice:
        """Get a device by its position in the context (not device_index)."""
        return self._devices[index]

    def enable_peer_access(self, buf: Buffer, *peers: AMDDevice) -> None:
        """Map a buffer's memory into peer GPU page tables for P2P access.

        After this call, the peer GPUs can directly read/write the buffer
        via XGMI without going through system memory.

        Args:
            buf: Buffer to share (must have been allocated on a device
                 in this context).
            peers: One or more peer AMDDevice instances to grant access to.
        """
        # Find the owning device's backend
        owner_backend = buf._backend
        if not isinstance(owner_backend, KFDDevice):
            raise RuntimeError("enable_peer_access requires KFD backend")

        peer_gpu_ids = []
        for peer in peers:
            peer_backend = peer.backend
            if not isinstance(peer_backend, KFDDevice):
                raise RuntimeError("enable_peer_access requires KFD backend")
            peer_gpu_ids.append(peer_backend.gpu_id)

        owner_backend.map_memory_to_peers(buf.handle, peer_gpu_ids)

    def copy_peer(
        self,
        dst: Buffer,
        dst_dev: AMDDevice,
        src: Buffer,
        src_dev: AMDDevice,
        size: int | None = None,
    ) -> None:
        """Copy data between buffers on different GPUs via XGMI SDMA.

        The copy is performed by the source device's XGMI SDMA engine.
        Both buffers must have peer access enabled for the other device.

        Args:
            dst: Destination buffer.
            dst_dev: Device that owns the destination buffer.
            src: Source buffer.
            src_dev: Device that owns the source buffer.
            size: Bytes to copy (None = min of both buffers).
        """
        if size is None:
            size = min(dst.size, src.size)

        src_backend = src_dev.backend
        if not isinstance(src_backend, KFDDevice):
            raise RuntimeError("copy_peer requires KFD backend")

        src_idx = src_dev.device_index

        # Get or lazily create XGMI SDMA queue on the source device.
        # Falls back to regular SDMA if no XGMI engines are available.
        queue = self._xgmi_queues.get(src_idx)
        if queue is None:
            node = src_backend.node
            has_xgmi = node is not None and node.num_sdma_xgmi_engines > 0
            if has_xgmi:
                queue = src_backend.create_xgmi_sdma_queue()
            else:
                queue = src_backend.create_sdma_queue()
            self._xgmi_queues[src_idx] = queue

        # Get or create timeline semaphore for the source device
        timeline = self._timelines.get(src_idx)
        if timeline is None:
            timeline = TimelineSemaphore(src_backend)
            self._timelines[src_idx] = timeline

        # Build SDMA copy + fence packets
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(dst.gpu_addr, src.gpu_addr, size)

        fence_value = timeline.next_value()
        sdma.fence(timeline.signal_addr, fence_value)

        src_backend.submit_packets(queue, sdma.build())

    def synchronize(self, dev: AMDDevice) -> None:
        """Wait for pending cross-GPU operations on a specific device."""
        timeline = self._timelines.get(dev.device_index)
        if timeline is not None:
            timeline.cpu_wait()

    def synchronize_all(self) -> None:
        """Wait for all pending cross-GPU operations on all devices."""
        for timeline in self._timelines.values():
            timeline.cpu_wait()

    def close(self) -> None:
        """Release all multi-GPU resources and close devices."""
        # Destroy timelines
        for timeline in self._timelines.values():
            timeline.destroy()
        self._timelines.clear()

        # Destroy XGMI queues
        for dev in self._devices:
            idx = dev.device_index
            queue = self._xgmi_queues.get(idx)
            if queue is not None:
                dev.backend.destroy_queue(queue)
        self._xgmi_queues.clear()

        # Close all devices
        for dev in self._devices:
            dev.close()
        self._devices.clear()

    def __enter__(self) -> MultiGPUContext:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"MultiGPUContext(num_devices={self.num_devices})"
