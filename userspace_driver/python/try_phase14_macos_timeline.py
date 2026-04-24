"""Phase 14: smoke macOS TimelineSemaphore signaling via WRITE_DATA."""
from __future__ import annotations

from amd_gpu_driver.device import AMDDevice
from amd_gpu_driver.sync.timeline import TimelineSemaphore


def main() -> None:
    dev = AMDDevice(backend="macos")
    print(f"device={dev.name} gfx={dev.gfx_target}")

    queue = dev.backend.create_compute_queue()
    timeline = TimelineSemaphore(dev.backend)

    value = timeline.next_value()
    print(f"timeline gpu=0x{timeline.signal_addr:x} value={value}")
    dev.backend.submit_packets(queue, timeline.signal_packets(value))
    timeline.cpu_wait(value, timeout_ms=5000)
    print(f"timeline value={timeline.gpu_value}")
    print("macOS TimelineSemaphore WRITE_DATA ✓")


if __name__ == "__main__":
    main()
