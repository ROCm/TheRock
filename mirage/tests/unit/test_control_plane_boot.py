from __future__ import annotations

import json

from mirage.cli import miragectl, miraged
from mirage.control_plane import (
    LifecycleState,
    SimulatorInstance,
    load_status_snapshot,
    materialize_runtime_view,
    write_status_snapshot,
)


SAMPLE_PROFILE = {
    "cluster_id": "lab-cluster",
    "nodes": [
        {
            "node_id": "node-a",
            "gpus": [
                {
                    "gpu_id": "gpu-0",
                    "arch_name": "cdna4",
                    "gfx_target": "gfx950",
                    "compute_units": 128,
                    "wavefront_size": 64,
                    "hbm_bytes": 17179869184,
                },
                {
                    "gpu_id": "gpu-1",
                    "arch_name": "cdna4",
                    "gfx_target": "gfx950",
                    "compute_units": 120,
                    "wavefront_size": 64,
                    "hbm_bytes": 17179869184,
                },
            ],
        },
        {
            "node_id": "node-b",
            "gpus": [
                {
                    "gpu_id": "gpu-0",
                    "arch_name": "cdna4",
                    "gfx_target": "gfx950",
                    "compute_units": 96,
                    "wavefront_size": 64,
                    "hbm_bytes": 8589934592,
                }
            ],
        },
    ],
    "links": [
        {
            "source_node_id": "node-a",
            "source_gpu_id": "gpu-0",
            "target_node_id": "node-b",
            "target_gpu_id": "gpu-0",
            "link_kind": "xgmi",
            "bandwidth_bytes_per_second": 32000000000,
            "latency_ns": 250,
        }
    ],
}


def test_materialize_runtime_view_from_profile():
    instance = SimulatorInstance.from_profile(SAMPLE_PROFILE)

    assert instance.state == LifecycleState.LOADED

    runtime = materialize_runtime_view(instance.profile)
    assert runtime.cluster_id == "lab-cluster"
    assert runtime.node_count == 2
    assert runtime.gpu_count == 3
    assert runtime.link_count == 1
    assert runtime.nodes[0].gpus[0].gfx_target == "gfx950"
    assert runtime.nodes[1].gpus[0].compute_units == 96


def test_simulator_instance_lifecycle_snapshot_round_trip(tmp_path):
    profile_path = _write_profile(tmp_path)
    snapshot_path = tmp_path / "runtime-state.json"

    instance = SimulatorInstance.from_profile(profile_path)
    booted = instance.boot()
    assert booted.state == LifecycleState.BOOTED
    assert booted.started_at is not None

    write_status_snapshot(booted, snapshot_path)
    loaded_snapshot = load_status_snapshot(snapshot_path)
    assert loaded_snapshot.state == LifecycleState.BOOTED
    assert loaded_snapshot.runtime.gpu_count == 3

    stopped = instance.shutdown()
    assert stopped.state == LifecycleState.STOPPED
    assert stopped.stopped_at is not None


def test_miraged_boot_persists_booted_status(tmp_path, capsys):
    profile_path = _write_profile(tmp_path)
    snapshot_path = tmp_path / "miraged-state.json"

    exit_code = miraged.main(
        [
            "boot",
            "--profile",
            str(profile_path),
            "--state-path",
            str(snapshot_path),
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "booted"
    assert output["runtime"]["node_count"] == 2
    assert json.loads(snapshot_path.read_text(encoding="utf-8"))["state"] == "booted"


def test_miragectl_boot_and_status_flow(tmp_path, capsys):
    profile_path = _write_profile(tmp_path)
    snapshot_path = tmp_path / "miragectl-state.json"

    boot_exit = miragectl.main(
        [
            "boot",
            "--profile",
            str(profile_path),
            "--state-path",
            str(snapshot_path),
            "--format",
            "json",
        ]
    )

    assert boot_exit == 0
    boot_output = json.loads(capsys.readouterr().out)
    assert boot_output["state"] == "booted"
    assert boot_output["runtime"]["gpu_count"] == 3

    status_exit = miragectl.main(
        [
            "status",
            "--state-path",
            str(snapshot_path),
            "--format",
            "json",
        ]
    )

    assert status_exit == 0
    status_output = json.loads(capsys.readouterr().out)
    assert status_output["state"] == "booted"
    assert status_output["runtime"]["nodes"][1]["node_id"] == "node-b"

    loaded_exit = miragectl.main(["--profile", str(profile_path), "--format", "json"])
    assert loaded_exit == 0
    loaded_output = json.loads(capsys.readouterr().out)
    assert loaded_output["state"] == "loaded"
    assert loaded_output["runtime"]["links"][0]["link_kind"] == "xgmi"


def _write_profile(tmp_path):
    profile_path = tmp_path / "cluster-profile.json"
    profile_path.write_text(json.dumps(SAMPLE_PROFILE), encoding="utf-8")
    return profile_path
