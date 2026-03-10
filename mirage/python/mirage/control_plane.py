"""Filesystem-backed control-plane helpers for simulator-native Mirage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_INSTANCE_NAME = "default"
DEFAULT_PROFILE_SOURCE = "builtin:single-gfx950"
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_SOURCE
STATE_RUNNING = "running"
STATE_STOPPED = "stopped"
STATE_BOOTING = "booting"
INSTANCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ControlPlaneError(RuntimeError):
    """Base control-plane error."""


class InvalidClusterProfileError(ControlPlaneError):
    """Raised when a cluster profile payload is invalid."""


class InstanceAlreadyRunningError(ControlPlaneError):
    """Raised when attempting to boot an already-running instance."""


class InstanceNotFoundError(ControlPlaneError):
    """Raised when runtime state for an instance does not exist."""


@dataclass(frozen=True)
class GpuProfile:
    gpu_id: str
    arch_name: str
    gfx_target: str
    compute_units: int = 0
    wavefront_size: int = 64
    hbm_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_id": self.gpu_id,
            "arch_name": self.arch_name,
            "gfx_target": self.gfx_target,
            "compute_units": self.compute_units,
            "wavefront_size": self.wavefront_size,
            "hbm_bytes": self.hbm_bytes,
        }


@dataclass(frozen=True)
class NodeProfile:
    node_id: str
    gpus: tuple[GpuProfile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }


@dataclass(frozen=True)
class FabricLinkProfile:
    source_node_id: str
    source_gpu_id: str
    target_node_id: str
    target_gpu_id: str
    link_kind: str
    bandwidth_bytes_per_second: int = 0
    latency_ns: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_node_id": self.source_node_id,
            "source_gpu_id": self.source_gpu_id,
            "target_node_id": self.target_node_id,
            "target_gpu_id": self.target_gpu_id,
            "link_kind": self.link_kind,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "latency_ns": self.latency_ns,
        }


@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: str
    nodes: tuple[NodeProfile, ...]
    links: tuple[FabricLinkProfile, ...] = ()

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def gpu_count(self) -> int:
        return sum(len(node.gpus) for node in self.nodes)

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "links": [link.to_dict() for link in self.links],
        }


@dataclass(frozen=True)
class RuntimeInstanceStatus:
    instance_name: str
    cluster_id: str
    current_state: str
    desired_state: str
    startup_mode: str
    profile_source: str
    created_at: str
    updated_at: str
    started_at: str | None
    stopped_at: str | None
    pid: int | None
    node_count: int
    gpu_count: int
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_name": self.instance_name,
            "cluster_id": self.cluster_id,
            "current_state": self.current_state,
            "desired_state": self.desired_state,
            "startup_mode": self.startup_mode,
            "profile_source": self.profile_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "pid": self.pid,
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeInstanceStatus:
        return cls(
            instance_name=_require_string(payload, "instance_name", "runtime state"),
            cluster_id=_require_string(payload, "cluster_id", "runtime state"),
            current_state=_require_string(payload, "current_state", "runtime state"),
            desired_state=_require_string(payload, "desired_state", "runtime state"),
            startup_mode=_require_string(payload, "startup_mode", "runtime state"),
            profile_source=_require_string(payload, "profile_source", "runtime state"),
            created_at=_require_string(payload, "created_at", "runtime state"),
            updated_at=_require_string(payload, "updated_at", "runtime state"),
            started_at=_optional_string(payload.get("started_at"), "runtime state"),
            stopped_at=_optional_string(payload.get("stopped_at"), "runtime state"),
            pid=_optional_int(payload.get("pid"), "runtime state"),
            node_count=_require_nonnegative_int(payload, "node_count", "runtime state"),
            gpu_count=_require_nonnegative_int(payload, "gpu_count", "runtime state"),
            last_error=_optional_string(payload.get("last_error"), "runtime state"),
        )


class LifecycleState(str, Enum):
    LOADED = "loaded"
    BOOTED = "booted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RuntimeGpu:
    gpu_id: str
    arch_name: str
    gfx_target: str
    compute_units: int
    wavefront_size: int
    hbm_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_id": self.gpu_id,
            "arch_name": self.arch_name,
            "gfx_target": self.gfx_target,
            "compute_units": self.compute_units,
            "wavefront_size": self.wavefront_size,
            "hbm_bytes": self.hbm_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeGpu":
        return cls(
            gpu_id=_require_string(payload, "gpu_id", "runtime gpu"),
            arch_name=_require_string(payload, "arch_name", "runtime gpu"),
            gfx_target=_require_string(payload, "gfx_target", "runtime gpu"),
            compute_units=_require_nonnegative_int(
                payload, "compute_units", "runtime gpu"
            ),
            wavefront_size=_require_nonnegative_int(
                payload, "wavefront_size", "runtime gpu"
            ),
            hbm_bytes=_require_nonnegative_int(payload, "hbm_bytes", "runtime gpu"),
        )


@dataclass(frozen=True)
class RuntimeNode:
    node_id: str
    gpus: tuple[RuntimeGpu, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeNode":
        gpus_payload = _require_list(payload, "gpus", "runtime node")
        return cls(
            node_id=_require_string(payload, "node_id", "runtime node"),
            gpus=tuple(
                RuntimeGpu.from_dict(
                    _require_mapping(gpu_payload, "runtime node gpu")
                )
                for gpu_payload in gpus_payload
            ),
        )


@dataclass(frozen=True)
class RuntimeLink:
    source_node_id: str
    source_gpu_id: str
    target_node_id: str
    target_gpu_id: str
    link_kind: str
    bandwidth_bytes_per_second: int
    latency_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_node_id": self.source_node_id,
            "source_gpu_id": self.source_gpu_id,
            "target_node_id": self.target_node_id,
            "target_gpu_id": self.target_gpu_id,
            "link_kind": self.link_kind,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "latency_ns": self.latency_ns,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeLink":
        return cls(
            source_node_id=_require_string(
                payload, "source_node_id", "runtime link"
            ),
            source_gpu_id=_require_string(payload, "source_gpu_id", "runtime link"),
            target_node_id=_require_string(
                payload, "target_node_id", "runtime link"
            ),
            target_gpu_id=_require_string(payload, "target_gpu_id", "runtime link"),
            link_kind=_require_string(payload, "link_kind", "runtime link"),
            bandwidth_bytes_per_second=_require_nonnegative_int(
                payload, "bandwidth_bytes_per_second", "runtime link"
            ),
            latency_ns=_require_nonnegative_int(payload, "latency_ns", "runtime link"),
        )


@dataclass(frozen=True)
class RuntimeView:
    cluster_id: str
    nodes: tuple[RuntimeNode, ...]
    links: tuple[RuntimeLink, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def gpu_count(self) -> int:
        return sum(len(node.gpus) for node in self.nodes)

    @property
    def link_count(self) -> int:
        return len(self.links)

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "link_count": self.link_count,
            "nodes": [node.to_dict() for node in self.nodes],
            "links": [link.to_dict() for link in self.links],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RuntimeView":
        nodes_payload = _require_list(payload, "nodes", "runtime view")
        links_payload = _require_list(payload, "links", "runtime view")
        return cls(
            cluster_id=_require_string(payload, "cluster_id", "runtime view"),
            nodes=tuple(
                RuntimeNode.from_dict(
                    _require_mapping(node_payload, "runtime view node")
                )
                for node_payload in nodes_payload
            ),
            links=tuple(
                RuntimeLink.from_dict(
                    _require_mapping(link_payload, "runtime view link")
                )
                for link_payload in links_payload
            ),
        )


@dataclass(frozen=True)
class SimulatorSnapshot:
    instance_name: str
    state: LifecycleState
    profile_source: str
    runtime: RuntimeView
    created_at: str
    updated_at: str
    started_at: str | None
    stopped_at: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_name": self.instance_name,
            "state": self.state.value,
            "profile_source": self.profile_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SimulatorSnapshot":
        state = _require_string(payload, "state", "simulator snapshot")
        return cls(
            instance_name=_require_string(
                payload, "instance_name", "simulator snapshot"
            ),
            state=LifecycleState(state),
            profile_source=_require_string(
                payload, "profile_source", "simulator snapshot"
            ),
            created_at=_require_string(payload, "created_at", "simulator snapshot"),
            updated_at=_require_string(payload, "updated_at", "simulator snapshot"),
            started_at=_optional_string(
                payload.get("started_at"), "simulator snapshot"
            ),
            stopped_at=_optional_string(
                payload.get("stopped_at"), "simulator snapshot"
            ),
            runtime=RuntimeView.from_dict(
                _require_mapping(payload.get("runtime"), "simulator runtime")
            ),
        )


def materialize_runtime_view(profile: ClusterProfile) -> RuntimeView:
    return RuntimeView(
        cluster_id=profile.cluster_id,
        nodes=tuple(
            RuntimeNode(
                node_id=node.node_id,
                gpus=tuple(
                    RuntimeGpu(
                        gpu_id=gpu.gpu_id,
                        arch_name=gpu.arch_name,
                        gfx_target=gpu.gfx_target,
                        compute_units=gpu.compute_units,
                        wavefront_size=gpu.wavefront_size,
                        hbm_bytes=gpu.hbm_bytes,
                    )
                    for gpu in node.gpus
                ),
            )
            for node in profile.nodes
        ),
        links=tuple(
            RuntimeLink(
                source_node_id=link.source_node_id,
                source_gpu_id=link.source_gpu_id,
                target_node_id=link.target_node_id,
                target_gpu_id=link.target_gpu_id,
                link_kind=link.link_kind,
                bandwidth_bytes_per_second=link.bandwidth_bytes_per_second,
                latency_ns=link.latency_ns,
            )
            for link in profile.links
        ),
    )


class SimulatorInstance:
    def __init__(
        self,
        profile: ClusterProfile,
        *,
        profile_source: str,
        instance_name: str = DEFAULT_INSTANCE_NAME,
    ) -> None:
        self.profile = profile
        self.profile_source = profile_source
        self.instance_name = normalize_instance_name(instance_name)
        self.state = LifecycleState.LOADED
        now = utc_now()
        self.created_at = now
        self.updated_at = now
        self.started_at: str | None = None
        self.stopped_at: str | None = None

    @classmethod
    def from_profile(
        cls,
        profile_source: Mapping[str, object] | str | Path | None,
        *,
        instance_name: str = DEFAULT_INSTANCE_NAME,
    ) -> "SimulatorInstance":
        if isinstance(profile_source, Mapping):
            profile = parse_cluster_profile(profile_source)
            source = "<inline-profile>"
        else:
            profile, source = load_cluster_profile(profile_source)
        return cls(profile, profile_source=source, instance_name=instance_name)

    def snapshot(self) -> SimulatorSnapshot:
        return SimulatorSnapshot(
            instance_name=self.instance_name,
            state=self.state,
            profile_source=self.profile_source,
            runtime=materialize_runtime_view(self.profile),
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            stopped_at=self.stopped_at,
        )

    def boot(self) -> SimulatorSnapshot:
        self.state = LifecycleState.BOOTED
        self.started_at = self.started_at or utc_now()
        self.stopped_at = None
        self.updated_at = utc_now()
        return self.snapshot()

    def shutdown(self) -> SimulatorSnapshot:
        self.state = LifecycleState.STOPPED
        self.stopped_at = utc_now()
        self.updated_at = self.stopped_at
        return self.snapshot()


def default_cluster_profile() -> ClusterProfile:
    return ClusterProfile(
        cluster_id="single-node",
        nodes=(
            NodeProfile(
                node_id="node0",
                gpus=(
                    GpuProfile(
                        gpu_id="gpu0",
                        arch_name="cdna4",
                        gfx_target="gfx950",
                        compute_units=64,
                        wavefront_size=64,
                        hbm_bytes=16 * 1024 * 1024 * 1024,
                    ),
                ),
            ),
        ),
    )


def load_cluster_profile(profile_path: str | Path | None) -> tuple[ClusterProfile, str]:
    if profile_path is None:
        return default_cluster_profile(), DEFAULT_PROFILE_SOURCE

    path = Path(profile_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_cluster_profile(payload), str(path)


def parse_cluster_profile(payload: Mapping[str, object]) -> ClusterProfile:
    mapping = _require_mapping(payload, "cluster profile")
    cluster_id = _require_string(mapping, "cluster_id", "cluster profile")
    nodes_payload = _require_list(mapping, "nodes", "cluster profile")
    if not nodes_payload:
        raise InvalidClusterProfileError("cluster profile requires at least one node")

    nodes: list[NodeProfile] = []
    for index, node_payload in enumerate(nodes_payload):
        node_mapping = _require_mapping(node_payload, f"cluster profile node[{index}]")
        node_id = _require_string(node_mapping, "node_id", f"node[{index}]")
        gpus_payload = _require_list(node_mapping, "gpus", f"node[{index}]")
        if not gpus_payload:
            raise InvalidClusterProfileError(
                f"cluster profile node[{index}] requires at least one gpu"
            )
        gpus: list[GpuProfile] = []
        for gpu_index, gpu_payload in enumerate(gpus_payload):
            gpu_mapping = _require_mapping(
                gpu_payload, f"cluster profile node[{index}] gpu[{gpu_index}]"
            )
            gpus.append(
                GpuProfile(
                    gpu_id=_require_string(
                        gpu_mapping, "gpu_id", f"node[{index}] gpu[{gpu_index}]"
                    ),
                    arch_name=_require_string(
                        gpu_mapping, "arch_name", f"node[{index}] gpu[{gpu_index}]"
                    ),
                    gfx_target=_require_string(
                        gpu_mapping, "gfx_target", f"node[{index}] gpu[{gpu_index}]"
                    ),
                    compute_units=_optional_nonnegative_int(
                        gpu_mapping.get("compute_units"), f"node[{index}] gpu[{gpu_index}]"
                    ),
                    wavefront_size=_optional_nonnegative_int(
                        gpu_mapping.get("wavefront_size"),
                        f"node[{index}] gpu[{gpu_index}]",
                        default=64,
                    ),
                    hbm_bytes=_optional_nonnegative_int(
                        gpu_mapping.get("hbm_bytes"), f"node[{index}] gpu[{gpu_index}]"
                    ),
                )
            )
        nodes.append(NodeProfile(node_id=node_id, gpus=tuple(gpus)))

    links: list[FabricLinkProfile] = []
    for index, link_payload in enumerate(mapping.get("links", [])):
        link_mapping = _require_mapping(link_payload, f"cluster profile link[{index}]")
        links.append(
            FabricLinkProfile(
                source_node_id=_require_string(
                    link_mapping, "source_node_id", f"link[{index}]"
                ),
                source_gpu_id=_require_string(
                    link_mapping, "source_gpu_id", f"link[{index}]"
                ),
                target_node_id=_require_string(
                    link_mapping, "target_node_id", f"link[{index}]"
                ),
                target_gpu_id=_require_string(
                    link_mapping, "target_gpu_id", f"link[{index}]"
                ),
                link_kind=_require_string(link_mapping, "link_kind", f"link[{index}]"),
                bandwidth_bytes_per_second=_optional_nonnegative_int(
                    link_mapping.get("bandwidth_bytes_per_second"), f"link[{index}]"
                ),
                latency_ns=_optional_nonnegative_int(
                    link_mapping.get("latency_ns"), f"link[{index}]"
                ),
            )
        )

    return ClusterProfile(cluster_id=cluster_id, nodes=tuple(nodes), links=tuple(links))


class RuntimeStore:
    """Filesystem-backed runtime state store."""

    def __init__(self, state_root: str | Path | None = None) -> None:
        self.state_root = resolve_state_root(state_root)

    @property
    def instances_dir(self) -> Path:
        return self.state_root / "instances"

    def instance_dir(self, instance_name: str) -> Path:
        normalized_name = normalize_instance_name(instance_name)
        return self.instances_dir / normalized_name

    def state_path(self, instance_name: str) -> Path:
        return self.instance_dir(instance_name) / "state.json"

    def control_path(self, instance_name: str) -> Path:
        return self.instance_dir(instance_name) / "control.json"

    def profile_snapshot_path(self, instance_name: str) -> Path:
        return self.instance_dir(instance_name) / "profile.json"

    def boot_instance(
        self,
        instance_name: str,
        profile: ClusterProfile,
        *,
        profile_source: str,
        startup_mode: str,
        force: bool = False,
        pid: int | None = None,
    ) -> RuntimeInstanceStatus:
        normalized_name = normalize_instance_name(instance_name)
        previous = self.try_get_instance(normalized_name)
        if (
            previous is not None
            and previous.current_state == STATE_RUNNING
            and not force
        ):
            raise InstanceAlreadyRunningError(
                f"Mirage instance '{normalized_name}' is already running"
            )

        started_at = utc_now()
        created_at = previous.created_at if previous is not None else started_at
        self._write_control(normalized_name, desired_state=STATE_RUNNING, updated_at=started_at)
        self._write_profile_snapshot(
            normalized_name,
            profile,
            profile_source=profile_source,
            loaded_at=started_at,
        )

        booting = RuntimeInstanceStatus(
            instance_name=normalized_name,
            cluster_id=profile.cluster_id,
            current_state=STATE_BOOTING,
            desired_state=STATE_RUNNING,
            startup_mode=startup_mode,
            profile_source=profile_source,
            created_at=created_at,
            updated_at=started_at,
            started_at=started_at,
            stopped_at=None,
            pid=pid,
            node_count=profile.node_count,
            gpu_count=profile.gpu_count,
        )
        self._write_state(booting)

        running = replace(booting, current_state=STATE_RUNNING, updated_at=utc_now())
        self._write_state(running)
        return running

    def stop_instance(self, instance_name: str) -> RuntimeInstanceStatus:
        status = self.get_instance(instance_name)
        stopped_at = utc_now()
        self._write_control(
            status.instance_name, desired_state=STATE_STOPPED, updated_at=stopped_at
        )
        if status.current_state == STATE_STOPPED:
            stopped = replace(
                status,
                desired_state=STATE_STOPPED,
                updated_at=stopped_at,
                stopped_at=status.stopped_at or stopped_at,
                pid=None,
            )
            self._write_state(stopped)
            return stopped

        stopped = replace(
            status,
            current_state=STATE_STOPPED,
            desired_state=STATE_STOPPED,
            updated_at=stopped_at,
            stopped_at=stopped_at,
            pid=None,
        )
        self._write_state(stopped)
        return stopped

    def get_instance(self, instance_name: str) -> RuntimeInstanceStatus:
        normalized_name = normalize_instance_name(instance_name)
        state_path = self.state_path(normalized_name)
        if not state_path.exists():
            raise InstanceNotFoundError(
                f"Mirage instance '{normalized_name}' has not been booted"
            )
        state_payload = _read_json_file(state_path)
        status = RuntimeInstanceStatus.from_dict(state_payload)
        desired_state = self._read_desired_state(
            normalized_name, default=status.desired_state
        )
        if desired_state != status.desired_state:
            status = replace(status, desired_state=desired_state)
        return status

    def try_get_instance(self, instance_name: str) -> RuntimeInstanceStatus | None:
        try:
            return self.get_instance(instance_name)
        except InstanceNotFoundError:
            return None

    def list_instances(self) -> list[RuntimeInstanceStatus]:
        if not self.instances_dir.exists():
            return []

        instances: list[RuntimeInstanceStatus] = []
        for instance_dir in sorted(self.instances_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            state_path = instance_dir / "state.json"
            if not state_path.exists():
                continue
            instances.append(self.get_instance(instance_dir.name))
        return instances

    def _write_state(self, status: RuntimeInstanceStatus) -> None:
        _write_json_file(self.state_path(status.instance_name), status.to_dict())

    def _write_control(
        self, instance_name: str, *, desired_state: str, updated_at: str
    ) -> None:
        _write_json_file(
            self.control_path(instance_name),
            {
                "schema_version": SCHEMA_VERSION,
                "instance_name": instance_name,
                "desired_state": desired_state,
                "updated_at": updated_at,
            },
        )

    def _write_profile_snapshot(
        self,
        instance_name: str,
        profile: ClusterProfile,
        *,
        profile_source: str,
        loaded_at: str,
    ) -> None:
        _write_json_file(
            self.profile_snapshot_path(instance_name),
            {
                "schema_version": SCHEMA_VERSION,
                "profile_source": profile_source,
                "loaded_at": loaded_at,
                "profile": profile.to_dict(),
            },
        )

    def _read_desired_state(self, instance_name: str, *, default: str) -> str:
        control_path = self.control_path(instance_name)
        if not control_path.exists():
            return default
        control_payload = _read_json_file(control_path)
        desired_state = control_payload.get("desired_state")
        if isinstance(desired_state, str):
            return desired_state
        return default


def resolve_state_root(state_root: str | Path | None) -> Path:
    if state_root is not None:
        return Path(state_root).expanduser().resolve()

    env_override = os.environ.get("MIRAGE_STATE_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return (Path(xdg_state_home).expanduser() / "mirage").resolve()

    return (Path.home() / ".local" / "state" / "mirage").resolve()


def normalize_instance_name(instance_name: str) -> str:
    normalized_name = instance_name.strip()
    if not normalized_name or not INSTANCE_NAME_PATTERN.match(normalized_name):
        raise ControlPlaneError(
            "instance names must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return normalized_name


def render_instance_status(
    status: RuntimeInstanceStatus, *, state_root: str | Path | None = None
) -> dict[str, object]:
    runtime_root = resolve_state_root(state_root)
    return {
        **status.to_dict(),
        "state_root": str(runtime_root),
        "instance_dir": str(runtime_root / "instances" / status.instance_name),
    }


def format_instance_status(status: RuntimeInstanceStatus) -> str:
    return (
        f"{status.instance_name}: state={status.current_state} "
        f"desired={status.desired_state} cluster={status.cluster_id} "
        f"nodes={status.node_count} gpus={status.gpu_count} "
        f"mode={status.startup_mode}"
    )


def format_snapshot(
    snapshot: SimulatorSnapshot, *, output_format: str = "text"
) -> str:
    if output_format == "json":
        return json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
    runtime = snapshot.runtime
    return (
        f"{snapshot.instance_name}: state={snapshot.state.value} "
        f"cluster={runtime.cluster_id} nodes={runtime.node_count} "
        f"gpus={runtime.gpu_count} links={runtime.link_count}"
    )


def write_status_snapshot(
    snapshot: SimulatorSnapshot, path: str | Path
) -> None:
    _write_json_file(Path(path).expanduser().resolve(), snapshot.to_dict())


def load_status_snapshot(path: str | Path) -> SimulatorSnapshot:
    payload = _read_json_file(Path(path).expanduser().resolve())
    return SimulatorSnapshot.from_dict(payload)


def utc_now() -> str:
    return f"{__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00', 'Z')}"


def _read_json_file(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ControlPlaneError(f"expected JSON object in {path}")
    return payload


def _write_json_file(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp_path.replace(path)


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidClusterProfileError(f"{context} must be a JSON object")
    return value


def _require_list(
    payload: Mapping[str, object], key: str, context: str
) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise InvalidClusterProfileError(f"{context} field '{key}' must be a list")
    return value


def _require_string(payload: Mapping[str, object], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidClusterProfileError(
            f"{context} field '{key}' must be a non-empty string"
        )
    return value.strip()


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidClusterProfileError(f"{context} optional string field is invalid")
    return value


def _optional_int(value: object, context: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise InvalidClusterProfileError(f"{context} optional integer field is invalid")
    return value


def _require_nonnegative_int(
    payload: Mapping[str, object], key: str, context: str
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise InvalidClusterProfileError(
            f"{context} field '{key}' must be a non-negative integer"
        )
    return value


def _optional_nonnegative_int(
    value: object, context: str, default: int = 0
) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or value < 0:
        raise InvalidClusterProfileError(
            f"{context} optional integer field must be non-negative"
        )
    return value


__all__ = [
    "ClusterProfile",
    "ControlPlaneError",
    "DEFAULT_INSTANCE_NAME",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_PROFILE_SOURCE",
    "FabricLinkProfile",
    "GpuProfile",
    "InstanceAlreadyRunningError",
    "InstanceNotFoundError",
    "InvalidClusterProfileError",
    "LifecycleState",
    "NodeProfile",
    "RuntimeInstanceStatus",
    "RuntimeGpu",
    "RuntimeLink",
    "RuntimeNode",
    "RuntimeView",
    "RuntimeStore",
    "SimulatorInstance",
    "SimulatorSnapshot",
    "default_cluster_profile",
    "format_snapshot",
    "format_instance_status",
    "load_cluster_profile",
    "load_status_snapshot",
    "materialize_runtime_view",
    "normalize_instance_name",
    "parse_cluster_profile",
    "render_instance_status",
    "resolve_state_root",
    "write_status_snapshot",
]
