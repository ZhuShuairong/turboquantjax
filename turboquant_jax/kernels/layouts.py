from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedTensorLayout:
    original_shape: tuple[int, ...]
    bits: int

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.original_shape:
            total *= int(dim)
        return total
