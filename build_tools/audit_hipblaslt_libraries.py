#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit hipBLASLt Tensile library logic YAML files: statistics and validation.

Single-module implementation (formerly split across gemm_name, lib_access, cache,
validators, report_html). Suitable for CI: exits with code 1 when validation
errors are found. Uses multiprocessing (up to 32 workers) and a disk cache.

Example:
    python3 build_tools/audit_hipblaslt_libraries.py \
        --hipblaslt-path /workspace/rocm-libraries/projects/hipblaslt \
        --subset gfx950/gfx950/Equality \
        --workers 16
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

import yaml

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


# ========================================================================
# gemm_name
# ========================================================================

# Maps short name tokens (filename) to Tensile DataType enum indices.
NAME_TO_TYPE_INDEX: dict[str, int] = {
    "S": 0,
    "D": 1,
    "H": 4,
    "I8": 5,
    "I": 6,
    "B": 7,
    "F8": 15,
    "B8": 16,
}

TYPE_INDEX_TO_SHORT: dict[int, str] = {
    0: "S",
    4: "H",
    5: "I8",
    6: "I",
    7: "B",
    15: "F8",
    16: "B8",
}

# Filename pattern: {arch}_Cijk_{Alayout}_{Blayout}_{typeToken}_...
FILENAME_RE = re.compile(
    r"^(?P<arch>[a-z0-9]+)_Cijk_(?P<alayout>A[a-z]{3})_(?P<blayout>B[a-z]{3})_(?P<type>.+?)_"
    r"(?:BH_|B_|UserArgs|HAS)",
    re.IGNORECASE,
)

EPILOGUE_FROM_BASENAME_RE = re.compile(
    r"Cijk_A[a-z]{3}_B[a-z]{3}_(?P<epilogue>.+?)_UserArgs",
    re.IGNORECASE,
)


def transpose_a_from_layout(alayout: str) -> bool:
    """Return TransposeA from an ``Axxx`` layout token (e.g. ``Alik``, ``Ailk``).

    Args:
        alayout: Four-character A-matrix layout token.

    Returns:
        TransposeA boolean ('l' in position 1 => True).

    Raises:
        ValueError: If ``alayout`` is not a valid A-layout token.
    """
    if len(alayout) != 4 or alayout[0] != "A":
        raise ValueError(f"Invalid A layout token: {alayout!r}")
    return alayout[1] == "l"


def transpose_b_from_layout(blayout: str) -> bool:
    """Return TransposeB from a ``Bxxx`` layout token (e.g. ``Bjlk``, ``Bljk``).

    Args:
        blayout: Four-character B-matrix layout token.

    Returns:
        TransposeB boolean ('l' in position 2 => True).

    Raises:
        ValueError: If ``blayout`` is not a valid B-layout token.
    """
    if len(blayout) != 4 or blayout[0] != "B":
        raise ValueError(f"Invalid B layout token: {blayout!r}")
    return blayout[2] == "l"


def _slice_to_alphabet(value: str) -> list[str]:
    slices: list[str] = []
    current = ""
    for char in value:
        if char.isalpha():
            if current:
                slices.append(current)
            current = char
        else:
            current += char
    if current:
        slices.append(current)
    return slices


def decode_gemm_type_token(token: str) -> Tuple[int, int, int, int, int]:
    """Decode a filename type token (e.g. ``B8BS``, ``F8F8S``) to type indices.

    Args:
        token: Type substring from the library filename.

    Returns:
        Tuple of (DataType/A, DataTypeB or same as A, DestDataType, DataTypeE, ComputeDataType)
        as Tensile enum indices. Compute uses index 0 (FP32) for HPA kernels.

    Raises:
        ValueError: If the token cannot be decoded.
    """
    if token == "S_MX":
        return 0, 0, 0, 0, 0

    input_name = ""
    gemm_name = token
    if "_" in token:
        input_name, gemm_name = token.split("_", 1)

    slices = _slice_to_alphabet(gemm_name)
    if len(slices) == 3:
        a = b = NAME_TO_TYPE_INDEX[slices[0]]
        c = d = NAME_TO_TYPE_INDEX[slices[1]]
        compute = 0
    elif len(slices) == 4:
        a = NAME_TO_TYPE_INDEX[slices[0]]
        b = NAME_TO_TYPE_INDEX[slices[1]]
        c = d = NAME_TO_TYPE_INDEX[slices[2]]
        compute = 0
    else:
        raise ValueError(f"Cannot decode GEMM type token {token!r}")

    if input_name:
        in_slices = _slice_to_alphabet(input_name)
        if len(in_slices) == 2:
            a = NAME_TO_TYPE_INDEX[in_slices[0]]
            b = NAME_TO_TYPE_INDEX[in_slices[1]]

    return a, b, c, d, compute


def parse_filename(stem: str) -> Optional[dict[str, object]]:
    """Parse architecture, layouts, and datatype token from a library filename stem.

    Args:
        stem: Filename without directory or extension.

    Returns:
        Dictionary of parsed fields, or None if the pattern does not match.
    """
    match = FILENAME_RE.match(stem)
    if not match:
        return None
    alayout = match.group("alayout")
    blayout = match.group("blayout")
    ta_a = transpose_a_from_layout(alayout)
    tb_b = transpose_b_from_layout(blayout)
    try:
        types = decode_gemm_type_token(match.group("type"))
    except ValueError:
        types = None
    return {
        "architecture": match.group("arch").lower(),
        "alayout": alayout,
        "blayout": blayout,
        "type_token": match.group("type"),
        "transpose_a": ta_a,
        "transpose_b": tb_b,
        "types": types,
    }


def extract_epilogue_key(name: str) -> str:
    """Extract epilogue suffix from a solution or library basename.

    Args:
        name: ``BaseName``, ``KernelNameMin``, or library filename stem.

    Returns:
        Epilogue key string, or empty string if not found.
    """
    match = EPILOGUE_FROM_BASENAME_RE.search(name)
    return match.group("epilogue") if match else ""

# ========================================================================
# lib_access
# ========================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DEVICE_RE = re.compile(r"device\s+([0-9a-f]+)", re.IGNORECASE)
DICT_BASED_ARCHITECTURES = frozenset({"gfx1250"})

# Library types that use device-specific directories when those directories exist.
DEVICE_SCOPED_LIBRARY_TYPES = frozenset({"Equality"})

# Default (generic) device ids per architecture for path placement.
DEFAULT_DEVICE_BY_ARCH: dict[str, str] = {
    "gfx950": "75a0",
    "gfx1250": "73f0",
}


@dataclass(frozen=True)
class LibraryRecord:
    """Parsed metadata for one hipBLASLt logic YAML file."""

    path: Path
    format: str  # "list" or "dict"
    architecture: str
    schedule_name: str
    device_ids: tuple[str, ...]
    library_type: str
    problem_type: dict[str, Any]
    solutions: list[dict[str, Any]]
    index_order: Any
    exact_logic: Any
    range_logic: Any
    perf_metric: Optional[str]


def _parse_devices(device_names: Any) -> tuple[str, ...]:
    if not device_names:
        return ()
    ids: list[str] = []
    for entry in device_names:
        if isinstance(entry, str):
            match = DEVICE_RE.search(entry)
            if match:
                ids.append(match.group(1).lower())
    return tuple(ids)


def _architecture_from_field(field: Any) -> str:
    if isinstance(field, dict):
        return str(field.get("Architecture", "")).lower()
    return str(field).lower()


def normalize_library(data: Any, path: Path) -> LibraryRecord:
    """Convert raw YAML data into a ``LibraryRecord``.

    Args:
        data: Parsed YAML root (list or dict).
        path: Path to the source file.

    Returns:
        Normalized library record.

    Raises:
        ValueError: If the document structure is not recognized.
    """
    if isinstance(data, list):
        if len(data) < 12:
            raise ValueError(f"List library has {len(data)} fields, expected at least 12")
        architecture = _architecture_from_field(data[2])
        return LibraryRecord(
            path=path,
            format="list",
            architecture=architecture,
            schedule_name=str(data[1]),
            device_ids=_parse_devices(data[3]),
            library_type=str(data[11]),
            problem_type=data[4],
            solutions=list(data[5]),
            index_order=data[6],
            exact_logic=data[7],
            range_logic=data[8],
            perf_metric=str(data[10]) if len(data) > 10 and data[10] is not None else None,
        )

    if isinstance(data, dict):
        architecture = str(data.get("ArchitectureName", "")).lower()
        library_type = str(data.get("LibraryType", ""))
        lib_table = data.get("Library", {})
        if isinstance(lib_table, dict) and lib_table.get("distance"):
            library_type = str(lib_table["distance"])
        return LibraryRecord(
            path=path,
            format="dict",
            architecture=architecture,
            schedule_name=str(data.get("ScheduleName", "")),
            device_ids=_parse_devices(data.get("DeviceNames")),
            library_type=library_type,
            problem_type=dict(data.get("ProblemType", {})),
            solutions=list(data.get("Solutions", [])),
            index_order=data.get("IndexOrder"),
            exact_logic=data.get("ExactLogic"),
            range_logic=data.get("RangeLogic"),
            perf_metric=data.get("PerfMetric"),
        )

    raise ValueError(f"Unsupported YAML root type: {type(data).__name__}")


def iter_gemm_sizes(exact_logic: Any) -> Iterable[list[Any]]:
    """Yield problem-size vectors from ExactLogic entries.

    Args:
        exact_logic: ExactLogic field from a library file.

    Yields:
        Size vectors (typically length 4 or 8).
    """
    if not exact_logic:
        return
    for entry in exact_logic:
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            continue
        size = entry[0]
        if isinstance(size, (list, tuple)):
            yield list(size)


def iter_performance_values(exact_logic: Any) -> Iterable[float]:
    """Yield performance numbers stored in ExactLogic mappings.

    Args:
        exact_logic: ExactLogic field from a library file.

    Yields:
        Floating-point performance values when present.
    """
    if not exact_logic:
        return
    for entry in exact_logic:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        mapping = entry[1]
        if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
            perf = mapping[1]
            if isinstance(perf, (int, float)):
                yield float(perf)


def solution_tiles(solutions: list[dict[str, Any]]) -> set[tuple[int, int, int]]:
    """Collect unique macro-tile shapes from solutions.

    Args:
        solutions: Solution dictionaries from the library file.

    Returns:
        Set of (MacroTile0, MacroTile1, DepthU) tuples.
    """
    tiles: set[tuple[int, int, int]] = set()
    for sol in solutions:
        if not isinstance(sol, dict):
            continue
        m0 = sol.get("MacroTile0")
        m1 = sol.get("MacroTile1")
        du = sol.get("DepthU")
        if m0 is not None and m1 is not None and du is not None:
            tiles.add((int(m0), int(m1), int(du)))
    return tiles


def solution_epilogue_keys(solutions: list[dict[str, Any]]) -> set[str]:
    """Collect epilogue keys from solution names.

    Args:
        solutions: Solution dictionaries.

    Returns:
        Set of epilogue key strings.
    """
    keys: set[str] = set()
    for sol in solutions:
        if not isinstance(sol, dict):
            continue
        for field in ("BaseName", "KernelNameMin", "SolutionNameMin"):
            value = sol.get(field)
            if isinstance(value, str):
                key = extract_epilogue_key(value)
                if key:
                    keys.add(key)
                    break
    return keys


def path_device_id(path: Path, architecture: str) -> Optional[str]:
    """Extract device id from a path segment like ``gfx950_id75a3``.

    Args:
        path: Library file path.
        architecture: Architecture string (e.g. ``gfx950``).

    Returns:
        Lowercase device id or None.
    """
    prefix = f"{architecture}_id"
    for part in path.parts:
        lower = part.lower()
        if lower.startswith(prefix):
            return lower[len(prefix) :]
    return None


def display_library_type(library_type: str) -> str:
    """Map YAML library type to the report label.

    Prediction libraries live under Origami and are reported as Origami.

    Args:
        library_type: Raw ``LibraryType`` from the YAML file.

    Returns:
        Library type string used in statistics and reports.
    """
    if library_type == "Prediction":
        return "Origami"
    return library_type


def path_library_folder(path: Path) -> str:
    """Return the library-type folder name (Equality, Origami, GridBased, ...).

    Args:
        path: Library file path.

    Returns:
        Folder name, or empty string if unknown.
    """
    known = {
        "Equality",
        "Origami",
        "GridBased",
        "Range",
        "Prediction",
        "FreeSize",
    }
    for part in path.parts:
        if part in known:
            return part
    return ""


def discover_device_folders(logic_root: Path) -> dict[str, set[str]]:
    """Map architecture to device ids that have dedicated ``{arch}_id*`` folders.

    Args:
        logic_root: Root hipBLASLt logic library directory.

    Returns:
        Dict arch -> set of device ids.
    """
    result: dict[str, set[str]] = {}
    for arch_dir in logic_root.iterdir():
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name.lower()
        ids: set[str] = set()
        for child in arch_dir.iterdir():
            if child.is_dir() and child.name.lower().startswith(f"{arch}_id"):
                ids.add(child.name.lower()[len(f"{arch}_id") :])
        if ids:
            result[arch] = ids
    return result

# ========================================================================
# cache
# ========================================================================

from pathlib import Path
from typing import Any, Optional

# Bump when validation rules or result schema change.
CACHE_VERSION = 5

# Relative path from hipBLASLt project root to Tensile logic libraries on disk.
HIPBLASLT_LOGIC_REL = Path(
    "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)


def logic_dir_from_hipblaslt(hipblaslt_path: Path) -> Path:
    """Resolve hipBLASLt logic library directory from a project root.

    Args:
        hipblaslt_path: Path to ``rocm-libraries/projects/hipblaslt``.

    Returns:
        Absolute path to ``Tensile/Logic/asm_full``.
    """
    return hipblaslt_path.resolve() / HIPBLASLT_LOGIC_REL


def default_hipblaslt_path() -> Path:
    """Return the default hipBLASLt project path for this environment.

    Returns:
        First existing candidate path, otherwise the first default candidate.
    """
    workspace = os.environ.get("GITHUB_WORKSPACE")
    candidates: list[Path] = []
    if workspace:
        candidates.append(Path(workspace) / "projects" / "hipblaslt")
    candidates.extend(
        [
            Path("/workspace/rocm-libraries/projects/hipblaslt"),
            Path(__file__).resolve().parents[1] / "rocm-libraries/projects/hipblaslt",
        ]
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def resolve_logic_dir(hipblaslt_path: Optional[Path] = None) -> Path:
    """Resolve the hipBLASLt logic library directory from CLI arguments.

    The directory is derived from ``--hipblaslt-path`` or the default hipBLASLt root.

    Args:
        hipblaslt_path: Optional hipBLASLt project root.

    Returns:
        Resolved logic directory path.
    """
    base = hipblaslt_path.resolve() if hipblaslt_path is not None else default_hipblaslt_path()
    return logic_dir_from_hipblaslt(base)


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


class AuditCache:
    """JSON-backed cache of per-library audit outcomes."""

    def __init__(self, cache_dir: Path, logic_dir: Path, enabled: bool = True) -> None:
        """Initialize cache storage.

        Args:
            cache_dir: Directory for ``entries.json``.
            logic_dir: hipBLASLt logic root (stored in manifest for invalidation).
            enabled: When False, all lookups miss and nothing is written.
        """
        self.cache_dir = cache_dir
        self.logic_dir = logic_dir.resolve()
        self.enabled = enabled
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0
        if enabled:
            self._load()

    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _entries_path(self) -> Path:
        return self.cache_dir / "entries.json"

    def _load(self) -> None:
        manifest_path = self._manifest_path()
        entries_path = self._entries_path()
        if not manifest_path.is_file() or not entries_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != CACHE_VERSION:
                return
            if Path(manifest.get("logic_dir", "")).resolve() != self.logic_dir:
                return
            self._entries = json.loads(entries_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def save(self) -> None:
        """Persist cache to disk if entries changed.

        Raises:
            OSError: If writing cache files fails.
        """
        if not self.enabled or not self._dirty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": CACHE_VERSION,
            "logic_dir": str(self.logic_dir),
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._entries_path().write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
        self._dirty = False

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counts.

        Returns:
            Dict with ``hits`` and ``misses`` keys.
        """
        return {"hits": self._hits, "misses": self._misses}

    def lookup(self, path: Path) -> Optional[dict[str, Any]]:
        """Return cached result if file signature matches.

        Args:
            path: Library YAML path.

        Returns:
            Cached result dict, or None on miss.
        """
        if not self.enabled:
            self._misses += 1
            return None
        key = self._rel_key(path)
        entry = self._entries.get(key)
        if not entry:
            self._misses += 1
            return None
        try:
            if entry["signature"] != _file_signature(path):
                self._misses += 1
                return None
        except OSError:
            self._misses += 1
            return None
        self._hits += 1
        return entry["result"]

    def store(self, path: Path, result: dict[str, Any]) -> None:
        """Store audit result for a file.

        Args:
            path: Library YAML path.
            result: Serializable per-file audit outcome.

        Raises:
            OSError: If the file cannot be stat'd.
        """
        if not self.enabled:
            return
        key = self._rel_key(path)
        self._entries[key] = {
            "signature": _file_signature(path),
            "result": result,
        }
        self._dirty = True

    def _rel_key(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.logic_dir))
        except ValueError:
            return str(path.resolve())

# ========================================================================
# validators
# ========================================================================

ErrorSink = Callable[[str, str, str], None]

ORIGAMI_ASSERT_FIELDS = (
    "AssertFree0ElementMultiple",
    "AssertFree1ElementMultiple",
    "AssertSummationElementMultiple",
)


def _is_origami_library(record: LibraryRecord) -> bool:
    """Return True if the library is stored under Origami or is Prediction type.

    Args:
        record: Parsed library record.

    Returns:
        True when Origami validation rules apply.
    """
    if record.library_type == "Prediction":
        return True
    return path_library_folder(record.path) == "Origami"


def _emit(sink: ErrorSink, path: Path, code: str, message: str) -> None:
    sink(str(path), code, message)


def validate_format(record: LibraryRecord, sink: ErrorSink) -> None:
    """Check list vs dict format matches architecture expectations.

    Args:
        record: Parsed library record.
        sink: Callback ``(path, code, message)`` for each error.
    """
    arch = record.architecture
    if arch in DICT_BASED_ARCHITECTURES and record.format != "dict":
        _emit(
            sink,
            record.path,
            "format_dict_expected",
            f"Architecture {arch} requires dict-based library format, found {record.format}",
        )
    elif arch not in DICT_BASED_ARCHITECTURES and record.format != "list":
        _emit(
            sink,
            record.path,
            "format_list_expected",
            f"Architecture {arch} requires list-based library format, found {record.format}",
        )


def validate_activation_func_call(record: LibraryRecord, sink: ErrorSink) -> None:
    """Require ActivationFuncCall=false on every solution.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    for idx, sol in enumerate(record.solutions):
        if not isinstance(sol, dict):
            continue
        value = sol.get("ActivationFuncCall")
        if value is not False:
            _emit(
                sink,
                record.path,
                "activation_func_call",
                f"Solution[{idx}] ActivationFuncCall={value!r}, expected false",
            )


def validate_epilogue_consistency(record: LibraryRecord, sink: ErrorSink) -> None:
    """For list-based libraries, all solutions must share the same epilogue key.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    if record.format != "list" or not record.solutions:
        return
    keys = solution_epilogue_keys(record.solutions)
    if len(keys) > 1:
        _emit(
            sink,
            record.path,
            "epilogue_mismatch",
            f"Solutions have differing epilogue keys: {sorted(keys)}",
        )


def validate_performance_zero(record: LibraryRecord, sink: ErrorSink) -> None:
    """ExactLogic performance values must be zero when numeric.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    for perf in iter_performance_values(record.exact_logic):
        if perf == 0.0:
            continue
        if math.isinf(perf):
            _emit(
                sink,
                record.path,
                "performance_nonzero",
                f"ExactLogic performance is {perf!r}, expected 0.0",
            )
            continue
        if abs(perf) > 1e-12:
            _emit(
                sink,
                record.path,
                "performance_nonzero",
                f"ExactLogic performance is {perf!r}, expected 0.0",
            )


def validate_path_device(
    record: LibraryRecord,
    device_folders: dict[str, set[str]],
    sink: ErrorSink,
) -> None:
    """Validate file path against device id and library type conventions.

    Args:
        record: Parsed library record.
        device_folders: Per-architecture device-specific folder ids.
        sink: Error callback.
    """
    arch = record.architecture
    path = record.path
    folder = path_library_folder(path)
    path_dev = path_device_id(path, arch)
    yaml_devices = record.device_ids
    default_dev = DEFAULT_DEVICE_BY_ARCH.get(arch)
    scoped_ids = device_folders.get(arch, set())

    if not yaml_devices:
        _emit(sink, path, "missing_device", "Library has no DeviceNames entries")
        return

    primary = yaml_devices[0]

    # Device-specific Equality (and similar) libraries belong under gfx*_id<dev>/.
    if (
        folder in DEVICE_SCOPED_LIBRARY_TYPES
        and primary in scoped_ids
        and primary != default_dev
    ):
        expected_segment = f"{arch}_id{primary}"
        if expected_segment not in [p.lower() for p in path.parts]:
            _emit(
                sink,
                path,
                "path_device_folder",
                f"Device {primary} + {folder} should live under {expected_segment}/",
            )
        if path_dev and path_dev != primary:
            _emit(
                sink,
                path,
                "path_device_mismatch",
                f"Path device id {path_dev} does not match primary yaml device {primary}",
            )
        return

    # Generic / default device libraries should not sit under a device-id folder.
    if path_dev and primary == default_dev:
        _emit(
            sink,
            path,
            "path_default_in_id_folder",
            f"Default device {primary} library should not be under {arch}_id{path_dev}",
        )

    # Prediction on gfx950 uses Origami folder (may be nested gfx950/gfx950/Origami).
    if record.library_type == "Prediction" and arch == "gfx950":
        if "Origami" not in path.parts:
            _emit(
                sink,
                path,
                "path_prediction_origami",
                "Prediction libraries for gfx950 should be under an Origami folder",
            )


def validate_filename_consistency(record: LibraryRecord, sink: ErrorSink) -> None:
    """Match filename layouts and datatype token to ProblemType fields.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    parsed = parse_filename(record.path.stem)
    if parsed is None:
        _emit(sink, record.path, "filename_parse", "Could not parse library filename")
        return

    if parsed["architecture"] != record.architecture:
        _emit(
            sink,
            record.path,
            "arch_filename_mismatch",
            f"Filename arch {parsed['architecture']} != yaml {record.architecture}",
        )

    pt = record.problem_type
    ta = pt.get("TransposeA")
    tb = pt.get("TransposeB")
    if ta is not None and bool(ta) != parsed["transpose_a"]:
        _emit(
            sink,
            record.path,
            "transpose_a_mismatch",
            f"Filename expects TransposeA={parsed['transpose_a']}, yaml has {ta!r}",
        )
    if tb is not None and bool(tb) != parsed["transpose_b"]:
        _emit(
            sink,
            record.path,
            "transpose_b_mismatch",
            f"Filename expects TransposeB={parsed['transpose_b']}, yaml has {tb!r}",
        )

    types = parsed.get("types")
    if types is None:
        return
    exp_a, exp_b, exp_c, exp_d, exp_comp = types
    checks = [
        ("DataType", exp_a),
        ("DataTypeA", exp_a),
        ("DataTypeB", exp_b),
        ("DestDataType", exp_d),
        ("DataTypeE", exp_d),
        ("ComputeDataType", exp_comp),
    ]
    for field, expected in checks:
        actual = pt.get(field)
        if actual is not None and int(actual) != expected:
            _emit(
                sink,
                record.path,
                "datatype_mismatch",
                f"Filename token {parsed['type_token']}: {field}={actual}, expected {expected}",
            )


def validate_origami_assert_multiples(record: LibraryRecord, sink: ErrorSink) -> None:
    """For Origami libraries, assert element multiples must be 1 on every solution.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    if not _is_origami_library(record):
        return
    for idx, sol in enumerate(record.solutions):
        if not isinstance(sol, dict):
            continue
        for field in ORIGAMI_ASSERT_FIELDS:
            value = sol.get(field)
            if value is None:
                _emit(
                    sink,
                    record.path,
                    "origami_assert_multiple_missing",
                    f"Solution[{idx}] missing {field}, expected 1",
                )
            elif int(value) != 1:
                _emit(
                    sink,
                    record.path,
                    "origami_assert_multiple",
                    f"Solution[{idx}] {field}={value!r}, expected 1",
                )


def validate_library_type_folder(record: LibraryRecord, sink: ErrorSink) -> None:
    """Library type in YAML should match the parent folder when present.

    Args:
        record: Parsed library record.
        sink: Error callback.
    """
    folder = path_library_folder(record.path)
    if not folder:
        return
    lib_type = record.library_type
    if lib_type == "Matching":
        if folder not in ("GridBased", "Equality", "Range"):
            _emit(
                sink,
                record.path,
                "library_type_folder",
                f"Matching library in unexpected folder {folder}",
            )
        return
    if lib_type != folder and not (lib_type == "Prediction" and folder == "Origami"):
        _emit(
            sink,
            record.path,
            "library_type_folder",
            f"YAML LibraryType {lib_type} does not match folder {folder}",
        )


def run_all_validators(
    record: LibraryRecord,
    device_folders: dict[str, set[str]],
    sink: ErrorSink,
) -> None:
    """Run every validation rule on one library record.

    Args:
        record: Parsed library.
        device_folders: Device-specific subfolder map.
        sink: Error callback.
    """
    validate_format(record, sink)
    validate_activation_func_call(record, sink)
    validate_epilogue_consistency(record, sink)
    validate_performance_zero(record, sink)
    validate_path_device(record, device_folders, sink)
    validate_filename_consistency(record, sink)
    validate_library_type_folder(record, sink)
    validate_origami_assert_multiples(record, sink)

# ========================================================================
# report_html
# ========================================================================

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    """HTML-escape a value for safe insertion into templates.

    Args:
        value: Value to escape.

    Returns:
        Escaped string.
    """
    return html.escape("" if value is None else str(value))


def _page(
    title: str,
    body: str,
    *,
    report_dir_name: str = "reports",
) -> str:
    """Wrap body content in a styled HTML page shell.

    Args:
        title: Page title.
        body: Inner HTML fragment.
        report_dir_name: Label for breadcrumb context.

    Returns:
        Complete HTML document string.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d8bfd;
      --ok: #3dd68c;
      --warn: #f5a524;
      --err: #f2555a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1.5rem 2rem 3rem;
      line-height: 1.5;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 0.5rem; }}
    h2 {{ font-size: 1.1rem; margin: 2rem 0 0.75rem; color: var(--muted); }}
    .meta {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }}
    nav.links {{ display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; margin-bottom: 2rem; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(10rem, 1fr));
      gap: 1rem;
      margin: 1rem 0 2rem;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
    }}
    .card .label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .card .value {{ font-size: 1.75rem; font-weight: 600; margin-top: 0.25rem; }}
    .card.ok .value {{ color: var(--ok); }}
    .card.warn .value {{ color: var(--warn); }}
    .card.err .value {{ color: var(--err); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 0.5rem 0.75rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      background: #243044;
      color: var(--muted);
      font-weight: 600;
      position: sticky;
      top: 0;
    }}
    tr:hover td {{ background: #1f2a3d; }}
    .path {{ font-family: ui-monospace, monospace; font-size: 0.8rem; word-break: break-all; }}
    .code {{
      display: inline-block;
      padding: 0.1rem 0.4rem;
      border-radius: 4px;
      background: #2a3548;
      font-family: ui-monospace, monospace;
      font-size: 0.8rem;
    }}
    .filter {{
      margin: 1rem 0;
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      align-items: center;
    }}
    .filter input, .filter select {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.6rem;
      border-radius: 6px;
      font-size: 0.875rem;
    }}
    .tiles {{ color: var(--muted); font-size: 0.8rem; max-width: 28rem; }}
    details summary {{ cursor: pointer; color: var(--accent); }}
    .bar-list {{ list-style: none; padding: 0; margin: 0; }}
    .bar-list li {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin: 0.35rem 0;
      font-size: 0.875rem;
    }}
    .bar-list .bar {{
      height: 8px;
      background: var(--accent);
      border-radius: 4px;
      min-width: 2px;
    }}
    .empty {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <nav class="links">
    <a href="index.html">Index</a>
    <a href="summary.html">Summary</a>
    <a href="statistics.html">Statistics</a>
    <a href="errors.html">Errors</a>
    <a href="parse_failures.html">Parse failures</a>
  </nav>
  {body}
</body>
</html>
"""


def write_index_html(
    report_dir: Path,
    summary: dict[str, Any],
    *,
    generated_at: str,
) -> None:
    """Write the main index HTML page.

    Args:
        report_dir: Report output directory.
        summary: Summary statistics dict.
        generated_at: ISO timestamp string for display.
    """
    errors = int(summary.get("validation_errors", 0))
    failures = int(summary.get("parse_failures", 0))
    err_class = "err" if errors else "ok"
    fail_class = "err" if failures else "ok"

    body = f"""
  <h1>hipBLASLt Library Audit</h1>
  <p class="meta">Generated {_esc(generated_at)}</p>
  <div class="cards">
    <div class="card ok"><div class="label">Files processed</div><div class="value">{_esc(summary.get('files_processed', 0))}</div></div>
    <div class="card"><div class="label">Groups</div><div class="value">{_esc(summary.get('groups', 0))}</div></div>
    <div class="card {err_class}"><div class="label">Validation errors</div><div class="value">{_esc(errors)}</div></div>
    <div class="card {fail_class}"><div class="label">Parse failures</div><div class="value">{_esc(failures)}</div></div>
    <div class="card"><div class="label">Cache hits</div><div class="value">{_esc(summary.get('cache_hits', 0))}</div></div>
    <div class="card"><div class="label">Cache misses</div><div class="value">{_esc(summary.get('cache_misses', 0))}</div></div>
  </div>
  <h2>Reports</h2>
  <ul>
    <li><a href="summary.html">summary.html</a> — overview metrics</li>
    <li><a href="statistics.html">statistics.html</a> — per arch / device / datatype / library type</li>
    <li><a href="errors.html">errors.html</a> — validation errors ({_esc(errors)})</li>
    <li><a href="parse_failures.html">parse_failures.html</a> — YAML parse failures ({_esc(failures)})</li>
  </ul>
  <h2>Machine-readable</h2>
  <ul>
    <li><a href="summary.json">summary.json</a></li>
    <li><a href="statistics.json">statistics.json</a></li>
    <li><a href="statistics.csv">statistics.csv</a></li>
    <li><a href="errors.json">errors.json</a></li>
    <li><a href="parse_failures.json">parse_failures.json</a></li>
  </ul>
"""
    (report_dir / "index.html").write_text(
        _page("hipBLASLt Audit", body, report_dir_name=report_dir.name),
        encoding="utf-8",
    )


def write_summary_html(report_dir: Path, summary: dict[str, Any], generated_at: str) -> None:
    """Write summary.html.

    Args:
        report_dir: Report output directory.
        summary: Summary dict.
        generated_at: ISO timestamp.
    """
    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"
        for k, v in summary.items()
    )
    body = f"""
  <h1>Summary</h1>
  <p class="meta">Generated {_esc(generated_at)}</p>
  <table><tbody>{rows}</tbody></table>
"""
    (report_dir / "summary.html").write_text(
        _page("Summary", body, report_dir_name=report_dir.name),
        encoding="utf-8",
    )


def _format_tiles(tiles: list[Any], max_show: int = 8) -> str:
    if not tiles:
        return '<span class="empty">—</span>'
    formatted = [f"{t[0]}×{t[1]}×{t[2]}" for t in tiles if isinstance(t, (list, tuple)) and len(t) >= 3]
    if len(formatted) <= max_show:
        return _esc(", ".join(formatted))
    shown = ", ".join(formatted[:max_show])
    rest = len(formatted) - max_show
    all_tiles = _esc(", ".join(formatted))
    return (
        f'<span class="tiles" title="{all_tiles}">{_esc(shown)} '
        f'<em>(+{rest} more)</em></span>'
    )


def write_statistics_html(
    report_dir: Path,
    stats_rows: list[dict[str, Any]],
    generated_at: str,
) -> None:
    """Write statistics.html with a sortable table.

    Args:
        report_dir: Report output directory.
        stats_rows: Statistics rows.
        generated_at: ISO timestamp.
    """
    if not stats_rows:
        body = """
  <h1>Statistics</h1>
  <p class="empty">No statistics rows.</p>
"""
        (report_dir / "statistics.html").write_text(
            _page("Statistics", body, report_dir_name=report_dir.name),
            encoding="utf-8",
        )
        return

    header = """
  <tr>
    <th>Architecture</th>
    <th>Device</th>
    <th>Datatype</th>
    <th>Library type</th>
    <th>Files</th>
    <th>GEMM sizes</th>
    <th>Kernels</th>
    <th>Unique tiles</th>
    <th>Tiles (sample)</th>
  </tr>
"""
    body_rows = []
    for row in stats_rows:
        body_rows.append(
            f"""<tr>
  <td>{_esc(row.get('architecture'))}</td>
  <td>{_esc(row.get('device_id'))}</td>
  <td>{_esc(row.get('datatype'))}</td>
  <td>{_esc(row.get('library_type'))}</td>
  <td>{_esc(row.get('files'))}</td>
  <td>{_esc(row.get('gemm_sizes'))}</td>
  <td>{_esc(row.get('kernels'))}</td>
  <td>{_esc(row.get('unique_tiles'))}</td>
  <td>{_format_tiles(row.get('tiles', []))}</td>
</tr>"""
        )

    body = f"""
  <h1>Statistics</h1>
  <p class="meta">Generated {_esc(generated_at)} · {_esc(len(stats_rows))} groups</p>
  <div class="filter">
    <label>Filter <input type="search" id="table-filter" placeholder="architecture, device, datatype…"></label>
  </div>
  <table id="stats-table">
    <thead>{header}</thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
  <script>
    const input = document.getElementById('table-filter');
    const rows = document.querySelectorAll('#stats-table tbody tr');
    input?.addEventListener('input', () => {{
      const q = input.value.toLowerCase();
      rows.forEach(r => {{
        r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }});
  </script>
"""
    (report_dir / "statistics.html").write_text(
        _page("Statistics", body, report_dir_name=report_dir.name),
        encoding="utf-8",
    )


def write_errors_html(
    report_dir: Path,
    errors: list[dict[str, str]],
    generated_at: str,
) -> None:
    """Write errors.html with code breakdown and filterable table.

    Args:
        report_dir: Report output directory.
        errors: Validation error list.
        generated_at: ISO timestamp.
    """
    counts = Counter(e.get("code", "unknown") for e in errors)
    max_count = max(counts.values()) if counts else 1

    bars = []
    for code, count in counts.most_common():
        width = int(100 * count / max_count)
        bars.append(
            f'<li><span class="code">{_esc(code)}</span> '
            f'<span>{_esc(count)}</span>'
            f'<span class="bar" style="width:{width}px"></span></li>'
        )
    bar_section = (
        f'<ul class="bar-list">{"".join(bars)}</ul>' if bars else '<p class="empty">No errors.</p>'
    )

    codes = sorted(counts.keys())
    options = '<option value="">All codes</option>' + "".join(
        f'<option value="{_esc(c)}">{_esc(c)} ({_esc(counts[c])})</option>' for c in codes
    )

    rows = []
    for err in errors:
        code = err.get("code", "")
        rows.append(
            f"""<tr data-code="{_esc(code)}">
  <td><span class="code">{_esc(code)}</span></td>
  <td class="path">{_esc(err.get('path'))}</td>
  <td>{_esc(err.get('message'))}</td>
</tr>"""
        )

    body = f"""
  <h1>Validation errors</h1>
  <p class="meta">Generated {_esc(generated_at)} · {_esc(len(errors))} errors</p>
  <h2>By error code</h2>
  {bar_section}
  <h2>All errors</h2>
  <div class="filter">
    <label>Code <select id="code-filter">{options}</select></label>
    <label>Search <input type="search" id="err-filter" placeholder="path or message…"></label>
  </div>
  <table id="errors-table">
    <thead><tr><th>Code</th><th>Path</th><th>Message</th></tr></thead>
    <tbody>{''.join(rows) if rows else '<tr><td colspan="3" class="empty">No validation errors.</td></tr>'}</tbody>
  </table>
  <script>
    const codeSel = document.getElementById('code-filter');
    const search = document.getElementById('err-filter');
    const errRows = document.querySelectorAll('#errors-table tbody tr[data-code]');
    function applyFilters() {{
      const code = (codeSel?.value || '').toLowerCase();
      const q = (search?.value || '').toLowerCase();
      errRows.forEach(r => {{
        const matchCode = !code || r.dataset.code.toLowerCase() === code;
        const matchText = !q || r.textContent.toLowerCase().includes(q);
        r.style.display = matchCode && matchText ? '' : 'none';
      }});
    }}
    codeSel?.addEventListener('change', applyFilters);
    search?.addEventListener('input', applyFilters);
  </script>
"""
    (report_dir / "errors.html").write_text(
        _page("Errors", body, report_dir_name=report_dir.name),
        encoding="utf-8",
    )


def write_parse_failures_html(
    report_dir: Path,
    parse_failures: list[dict[str, str]],
    generated_at: str,
) -> None:
    """Write parse_failures.html.

    Args:
        report_dir: Report output directory.
        parse_failures: Parse failure list.
        generated_at: ISO timestamp.
    """
    rows = "".join(
        f"""<tr>
  <td class="path">{_esc(item.get('path'))}</td>
  <td>{_esc(item.get('error'))}</td>
</tr>"""
        for item in parse_failures
    )
    body = f"""
  <h1>Parse failures</h1>
  <p class="meta">Generated {_esc(generated_at)} · {_esc(len(parse_failures))} failures</p>
  <table>
    <thead><tr><th>Path</th><th>Error</th></tr></thead>
    <tbody>
      {rows if rows else '<tr><td colspan="2" class="empty">No parse failures.</td></tr>'}
    </tbody>
  </table>
"""
    (report_dir / "parse_failures.html").write_text(
        _page("Parse failures", body, report_dir_name=report_dir.name),
        encoding="utf-8",
    )


def write_html_reports(
    report_dir: Path,
    summary: dict[str, Any],
    stats_rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    parse_failures: list[dict[str, str]],
) -> None:
    """Write all HTML report pages.

    Args:
        report_dir: Report output directory.
        summary: Summary statistics.
        stats_rows: Statistics table rows.
        errors: Validation errors.
        parse_failures: YAML parse failures.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    write_index_html(report_dir, summary, generated_at=generated_at)
    write_summary_html(report_dir, summary, generated_at)
    write_statistics_html(report_dir, stats_rows, generated_at)
    write_errors_html(report_dir, errors, generated_at)
    write_parse_failures_html(report_dir, parse_failures, generated_at)

# ========================================================================
# audit entrypoint
# ========================================================================

MAX_WORKERS = 32


@dataclass
class GroupStats:
    """Aggregated counters for one statistics group."""

    files: int = 0
    gemm_sizes: int = 0
    kernels: int = 0
    tiles: set[tuple[int, int, int]] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "files": self.files,
            "gemm_sizes": self.gemm_sizes,
            "kernels": self.kernels,
            "unique_tiles": len(self.tiles),
            "tiles": sorted(self.tiles),
        }


@dataclass
class AuditResult:
    """Full audit output."""

    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    parse_failures: list[dict[str, str]] = field(default_factory=list)
    files_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


def _group_key(record: LibraryRecord) -> tuple[str, str, str, str]:
    type_token = ""
    parsed = parse_filename(record.path.stem)
    if parsed:
        type_token = str(parsed.get("type_token", ""))
    device = record.device_ids[0] if record.device_ids else "unknown"
    return (record.architecture, device, type_token, display_library_type(record.library_type))


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=YAML_LOADER)


def _audit_file_worker(args: tuple[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Process-pool entry point.

    Args:
        args: ``(path_str, device_folders_serializable)`` where device folder
            values are lists (sets are not picklable on all platforms).

    Returns:
        Per-file audit result without non-picklable objects.
    """
    path_str, folders_lists = args
    path = Path(path_str)
    device_folders = {k: set(v) for k, v in folders_lists.items()}

    try:
        data = _load_yaml(path)
        record = normalize_library(data, path)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path_str,
            "parse_failure": f"{type(exc).__name__}: {exc}",
            "group_key": None,
            "gemm_sizes": 0,
            "kernels": 0,
            "tiles": [],
            "errors": [],
        }

    errors: list[dict[str, str]] = []

    def sink(file_path: str, code: str, message: str) -> None:
        errors.append({"path": file_path, "code": code, "message": message})

    run_all_validators(record, device_folders, sink)
    key = _group_key(record)
    return {
        "path": path_str,
        "parse_failure": None,
        "group_key": list(key),
        "gemm_sizes": sum(1 for _ in iter_gemm_sizes(record.exact_logic)),
        "kernels": len(record.solutions),
        "tiles": [list(t) for t in sorted(solution_tiles(record.solutions))],
        "errors": errors,
    }


def _merge_file_result(
    file_result: dict[str, Any],
    groups: dict[tuple[str, str, str, str], GroupStats],
    result: AuditResult,
) -> None:
    """Merge one file result into aggregate statistics and error lists.

    Args:
        file_result: Output from ``_audit_file_worker`` or cache.
        groups: Mutable group statistics map.
        result: Mutable audit result container.
    """
    if file_result.get("parse_failure"):
        result.parse_failures.append(
            {"path": file_result["path"], "error": file_result["parse_failure"]}
        )
        return

    result.files_processed += 1
    key_list = file_result.get("group_key")
    if not key_list:
        return
    key = tuple(key_list)
    stats = groups[key]
    stats.files += 1
    stats.kernels += int(file_result.get("kernels", 0))
    stats.gemm_sizes += int(file_result.get("gemm_sizes", 0))
    for tile in file_result.get("tiles", []):
        stats.tiles.add(tuple(tile))
    result.errors.extend(file_result.get("errors", []))


def _collect_yaml_files(
    logic_dir: Path,
    subset: Optional[str],
    limit: Optional[int],
) -> list[Path]:
    """Build the list of YAML files to audit.

    Args:
        logic_dir: hipBLASLt logic library root directory.
        subset: Optional relative subdirectory under the logic root.
        limit: Optional maximum number of files.

    Returns:
        Sorted list of paths to audit.
    """
    if subset:
        root = (logic_dir / subset).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Subset directory not found: {root}")
        yaml_files = sorted(root.rglob("*.yaml"))
    else:
        yaml_files = sorted(logic_dir.rglob("*.yaml"))

    if limit is not None and limit > 0:
        yaml_files = yaml_files[:limit]

    return yaml_files


def write_reports(
    result: AuditResult,
    report_dir: Path,
    groups: dict[tuple[str, str, str, str], GroupStats],
) -> None:
    """Write JSON and CSV reports.

    Args:
        result: Completed audit result.
        report_dir: Output directory (created if missing).
        groups: Aggregated statistics keyed by (arch, device, datatype, library_type).
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: list[dict[str, Any]] = []
    for (arch, device, datatype, library_type), group in sorted(groups.items()):
        stats_rows.append(
            {
                "architecture": arch,
                "device_id": device,
                "datatype": datatype,
                "library_type": library_type,
                **group.to_dict(),
            }
        )

    summary = {
        "files_processed": result.files_processed,
        "parse_failures": len(result.parse_failures),
        "validation_errors": len(result.errors),
        "groups": len(stats_rows),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
    }

    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (report_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(stats_rows, handle, indent=2)

    with (report_dir / "errors.json").open("w", encoding="utf-8") as handle:
        json.dump(result.errors, handle, indent=2)

    with (report_dir / "parse_failures.json").open("w", encoding="utf-8") as handle:
        json.dump(result.parse_failures, handle, indent=2)

    if stats_rows:
        fieldnames = list(stats_rows[0].keys())
        with (report_dir / "statistics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats_rows:
                row_copy = dict(row)
                row_copy["tiles"] = ";".join(
                    f"{a}x{b}x{c}" for a, b, c in row_copy.get("tiles", [])
                )
                writer.writerow(row_copy)

    write_html_reports(
        report_dir,
        summary=summary,
        stats_rows=stats_rows,
        errors=result.errors,
        parse_failures=result.parse_failures,
    )


def run_audit(
    logic_dir: Path,
    report_dir: Path,
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    workers: int = MAX_WORKERS,
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
) -> AuditResult:
    """Run the audit over hipBLASLt logic libraries (parallel, with optional cache).

    Args:
        logic_dir: Root directory containing architecture subfolders.
        report_dir: Where to write reports.
        subset: Optional relative path under the logic root to limit scope.
        limit: Optional max file count (for testing).
        workers: Parallel worker count (capped at 32).
        cache_dir: Cache directory (default: report_dir/.cache).
        no_cache: Disable disk cache.

    Returns:
        Completed audit result.

    Raises:
        FileNotFoundError: If subset path does not exist.
    """
    logic_dir = logic_dir.resolve()
    workers = max(1, min(workers, MAX_WORKERS, os.cpu_count() or 1))
    device_folders = discover_device_folders(logic_dir)
    folders_serializable = {k: sorted(v) for k, v in device_folders.items()}

    yaml_files = _collect_yaml_files(logic_dir, subset, limit)
    groups: dict[tuple[str, str, str, str], GroupStats] = defaultdict(GroupStats)
    result = AuditResult()

    cache_path = (cache_dir or (report_dir / ".cache")).resolve()
    cache = AuditCache(cache_path, logic_dir, enabled=not no_cache)

    to_process: list[Path] = []
    for path in yaml_files:
        cached = cache.lookup(path)
        if cached is not None:
            _merge_file_result(cached, groups, result)
        else:
            to_process.append(path)

    result.cache_hits = cache.stats["hits"]
    result.cache_misses = cache.stats["misses"]

    if to_process:
        t0 = time.perf_counter()
        print(f"Cache: {result.cache_hits} hits, auditing {len(to_process)} files...", file=sys.stderr)
        job_args = [(str(p), folders_serializable) for p in to_process]

        if workers == 1:
            results_by_path = {}
            for a in job_args:
                file_result = _audit_file_worker(a)
                results_by_path[file_result["path"]] = file_result
        else:
            results_by_path = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_audit_file_worker, a): a[0] for a in job_args}
                for future in as_completed(futures):
                    file_result = future.result()
                    results_by_path[file_result["path"]] = file_result

        for path in to_process:
            file_result = results_by_path[str(path)]
            cache.store(path, file_result)
            _merge_file_result(file_result, groups, result)

        elapsed = time.perf_counter() - t0
        print(
            f"Audited {len(to_process)} files in {elapsed:.1f}s "
            f"({workers} workers, {len(to_process) / max(elapsed, 0.001):.1f} files/s)",
            file=sys.stderr,
        )

    cache.save()
    result.cache_hits = cache.stats["hits"]
    result.cache_misses = cache.stats["misses"]

    result.stats = {
        f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v.to_dict() for k, v in sorted(groups.items())
    }
    write_reports(result, report_dir, groups)
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hipblaslt-path",
        type=Path,
        default=None,
        help=(
            "Path to rocm-libraries/projects/hipblaslt; "
            "logic directory is derived as library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for JSON/CSV reports",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Relative path under hipBLASLt logic root (e.g. gfx950/gfx950/Equality)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of YAML files to process (for quick tests)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(MAX_WORKERS, os.cpu_count() or 1),
        help=f"Parallel workers (1..{MAX_WORKERS}, default: min(32, cpu_count))",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: <report-dir>/.cache)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code 0 on success, 1 if errors or parse failures exist.
    """
    args = parse_args(argv)
    logic_dir = resolve_logic_dir(args.hipblaslt_path)
    if not logic_dir.is_dir():
        print(f"hipBLASLt logic directory not found: {logic_dir}", file=sys.stderr)
        if args.hipblaslt_path:
            print(f"  (from --hipblaslt-path {args.hipblaslt_path.resolve()})", file=sys.stderr)
        return 2

    result = run_audit(
        logic_dir,
        args.report_dir.resolve(),
        subset=args.subset,
        limit=args.limit,
        workers=args.workers,
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        no_cache=args.no_cache,
    )
    print(
        f"Processed {result.files_processed} files, "
        f"{len(result.parse_failures)} parse failures, "
        f"{len(result.errors)} validation errors, "
        f"cache {result.cache_hits} hits / {result.cache_misses} misses"
    )
    report_path = args.report_dir.resolve()
    print(f"Reports written to {report_path}")
    print(f"HTML dashboard: {report_path / 'index.html'}")

    if result.parse_failures or result.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
