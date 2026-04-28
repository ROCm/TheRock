import importlib.util
from pathlib import Path


_REPO_MANAGEMENT_PATH = (
    Path(__file__).resolve().parents[2]
    / "external-builds"
    / "pytorch"
    / "repo_management.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "pytorch_repo_management", _REPO_MANAGEMENT_PATH
)
assert _SPEC is not None
repo_management = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repo_management)


def test_read_pytorch_rocm_pins_uses_exact_os_match(tmp_path: Path) -> None:
    (tmp_path / "related_commits").write_text(
        "ubuntu|pytorch|torchaudio|main|ubuntu-commit|https://github.com/pytorch/audio\n"
        "centos|pytorch|torchaudio|main|centos-commit|https://github.com/pytorch/audio\n"
    )

    origin, commit, found = repo_management.read_pytorch_rocm_pins(
        tmp_path,
        os="centos",
        project="torchaudio",
        default_origin="default-origin",
        default_hashtag="default-commit",
    )

    assert origin == "https://github.com/pytorch/audio"
    assert commit == "centos-commit"
    assert found


def test_read_pytorch_rocm_pins_falls_back_to_unique_project_match(
    tmp_path: Path,
) -> None:
    (tmp_path / "related_commits").write_text(
        "ubuntu|pytorch|torchaudio|main|shared-commit|https://github.com/pytorch/audio\n"
    )

    origin, commit, found = repo_management.read_pytorch_rocm_pins(
        tmp_path,
        os="centos",
        project="torchaudio",
        default_origin="default-origin",
        default_hashtag="default-commit",
    )

    assert origin == "https://github.com/pytorch/audio"
    assert commit == "shared-commit"
    assert found


def test_read_pytorch_rocm_pins_keeps_defaults_for_ambiguous_project_match(
    tmp_path: Path,
) -> None:
    (tmp_path / "related_commits").write_text(
        "ubuntu|pytorch|torchaudio|main|ubuntu-commit|https://github.com/pytorch/audio\n"
        "windows|pytorch|torchaudio|main|windows-commit|https://github.com/pytorch/audio\n"
    )

    origin, commit, found = repo_management.read_pytorch_rocm_pins(
        tmp_path,
        os="centos",
        project="torchaudio",
        default_origin="default-origin",
        default_hashtag="default-commit",
    )

    assert origin == "default-origin"
    assert commit == "default-commit"
    assert not found
