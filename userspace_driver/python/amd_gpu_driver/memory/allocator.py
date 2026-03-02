"""VA space tracking for GPU memory allocations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VARange:
    """A range of virtual addresses."""

    start: int
    size: int

    @property
    def end(self) -> int:
        return self.start + self.size


class VASpaceTracker:
    """Tracks allocated virtual address ranges.

    Used to avoid VA conflicts when making multiple allocations
    and for debugging/diagnostics.
    """

    def __init__(self) -> None:
        self._ranges: list[VARange] = []

    def add(self, start: int, size: int) -> None:
        """Record a VA allocation."""
        self._ranges.append(VARange(start=start, size=size))
        self._ranges.sort(key=lambda r: r.start)

    def remove(self, start: int) -> None:
        """Remove a VA allocation by start address."""
        self._ranges = [r for r in self._ranges if r.start != start]

    def contains(self, addr: int) -> bool:
        """Check if an address falls within any tracked range."""
        return any(r.start <= addr < r.end for r in self._ranges)

    def find(self, addr: int) -> VARange | None:
        """Find the range containing an address."""
        for r in self._ranges:
            if r.start <= addr < r.end:
                return r
        return None

    @property
    def total_allocated(self) -> int:
        """Total bytes tracked."""
        return sum(r.size for r in self._ranges)

    @property
    def num_ranges(self) -> int:
        return len(self._ranges)

    def clear(self) -> None:
        self._ranges.clear()
