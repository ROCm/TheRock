# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the consumer-graph generator CLI and its drift check."""

import json
from pathlib import Path

import pytest

import generate_consumer_graph as gcg

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _skip_without_checkout() -> None:
    if not (
        (_REPO_ROOT / ".git").exists()
        and (_REPO_ROOT / "CMakeLists.txt").exists()
        and gcg.CONSUMER_GRAPH_PATH.exists()
    ):
        pytest.skip("not a TheRock git checkout with committed consumer-graph files")


# --- _map_delta: pure diff of two "name -> list" maps ---


def test_map_delta_empty_when_equal() -> None:
    assert gcg._map_delta({"a": ["x"]}, {"a": ["x"]}, "subtree") == []


def test_map_delta_reports_generated_only() -> None:
    assert gcg._map_delta({"a": ["x"]}, {}, "subtree") == [
        "    generated-only subtree: a -> ['x']"
    ]


def test_map_delta_reports_committed_only() -> None:
    assert gcg._map_delta({}, {"a": ["x"]}, "subtree") == [
        "    committed-only subtree: a -> ['x']"
    ]


def test_map_delta_reports_changed_value() -> None:
    assert gcg._map_delta({"a": ["x", "y"]}, {"a": ["x"]}, "artifact") == [
        "    artifact a: generated ['x', 'y'] != committed ['x']"
    ]


def test_map_delta_sorts_keys() -> None:
    lines = gcg._map_delta({"b": ["2"], "a": ["1"]}, {}, "subtree")
    assert lines == [
        "    generated-only subtree: a -> ['1']",
        "    generated-only subtree: b -> ['2']",
    ]


# --- _read_committed_map: missing file reads as empty ---


def test_read_committed_map_missing_returns_empty(tmp_path: Path) -> None:
    assert gcg._read_committed_map(tmp_path / "absent.json") == {}


def test_read_committed_map_present_parses(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text('{"a": ["x"]}', encoding="utf-8")
    assert gcg._read_committed_map(path) == {"a": ["x"]}


# --- main(): --check gate and write mode ---

# main() formats its paths with relative_to(THEROCK_DIR), so the write-mode and
# drift tests redirect THEROCK_DIR + the output paths into a temp tree and feed a
# fixed analysis result, keeping them off the real committed files.
_FAKE_GRAPH = {"foo": {"consumers": ["bar"]}}
_FAKE_SUBTREE = {"projects/foo": ["foo"]}
_FAKE_SOURCE_DIR = {"art": ["projects/foo"]}


class _FakeResult:
    def build_consumer_graph(self) -> dict:
        return _FAKE_GRAPH

    def build_subtree_map(self) -> dict:
        return _FAKE_SUBTREE

    def build_source_dir_map(self) -> dict:
        return _FAKE_SOURCE_DIR


class _FakeAnalyzer:
    def __init__(self, root: Path) -> None:
        pass

    def analyze(self) -> _FakeResult:
        return _FakeResult()


def _redirect_into(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(gcg, "RepositoryAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(gcg, "THEROCK_DIR", tmp_path)
    paths = {
        "CONSUMER_GRAPH_PATH": tmp_path / "test_tools" / "therock_consumer_graph.json",
        "SUBTREE_MAP_PATH": tmp_path / "test_tools" / "therock_subtree_map.json",
        "SOURCE_DIR_MAP_PATH": tmp_path / "test_tools" / "therock_source_dir_map.json",
    }
    for attr, path in paths.items():
        monkeypatch.setattr(gcg, attr, path)
    return paths


def test_check_passes_on_committed_tree(capsys: pytest.CaptureFixture[str]) -> None:
    # Real-tree integration: the committed files must match the live parser output.
    _skip_without_checkout()
    assert gcg.main(["--check"]) == 0
    assert "match the committed files" in capsys.readouterr().out


def test_write_mode_creates_all_three_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _redirect_into(tmp_path, monkeypatch)

    assert gcg.main([]) == 0
    assert json.loads(paths["CONSUMER_GRAPH_PATH"].read_text()) == _FAKE_GRAPH
    assert json.loads(paths["SUBTREE_MAP_PATH"].read_text()) == _FAKE_SUBTREE
    assert json.loads(paths["SOURCE_DIR_MAP_PATH"].read_text()) == _FAKE_SOURCE_DIR
    assert "Wrote consumer graph" in capsys.readouterr().out


def test_check_returns_one_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _redirect_into(tmp_path, monkeypatch)
    # Committed copies empty, so every parser-derived entry reads as generated-only.
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    assert gcg.main(["--check"]) == 1
    assert "differs from" in capsys.readouterr().out


def test_check_reports_drift_when_committed_map_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A missing committed subtree/source-dir map reads as full drift, not a
    # FileNotFoundError crash. Write all three first so the graph round-trips and
    # only the deleted maps register as drift.
    paths = _redirect_into(tmp_path, monkeypatch)
    assert gcg.main([]) == 0
    capsys.readouterr()
    paths["SUBTREE_MAP_PATH"].unlink()
    paths["SOURCE_DIR_MAP_PATH"].unlink()

    assert gcg.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "subtree map differs from" in out
    assert "source-dir map differs from" in out
