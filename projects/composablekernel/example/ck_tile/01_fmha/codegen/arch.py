# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Any, List, Callable


@dataclass(frozen=True)
class ArchTrait:
    name: str
    preprocessor_check: str = field(default=None)
    device_name_check: str = field(default=None)
    tag: str = field(default=None)
    filename_suffix: str = field(default=None)

    def __post_init__(self):
        if self.preprocessor_check is None:
            object.__setattr__(self, "preprocessor_check", f"defined(__{self.name}__)")
        if self.device_name_check is None:
            object.__setattr__(
                self,
                "device_name_check",
                f'device_name.compare(0, {len(self.name)}, "{self.name}") == 0',
            )
        if self.tag is None:
            object.__setattr__(self, "tag", f"ck_tile::{self.name}_t")
        if self.filename_suffix is None:
            object.__setattr__(self, "filename_suffix", f"_{self.name}")


def get_factories_for_targets(
    targets: List[str], get_factory: Callable[[str], Any]
) -> List[Any]:
    factories = dict()
    for target in targets:
        factory = get_factory(target)
        factories[factory.arch.name] = factory
    # Place more specific architectures first
    factories = sorted(
        list(factories.values()), key=lambda f: len(f.arch.name), reverse=True
    )
    return factories
