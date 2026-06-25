"""Render tests for the post-strip PHDR-normalize hooks in the rpm/deb templates.

These assert the normalize command is injected after the strip step (and skipped
when stripping is disabled), without needing rpmbuild/dh. The end-to-end behavior
(real llvm-strip then normalize) is covered in CI's package build.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = Path(__file__).resolve().parents[1]  # build_tools/packaging/linux


def _render(template_name: str, **ctx) -> str:
    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    return env.get_template(template_name).render(**ctx)


def test_rpm_spec_injects_normalize_after_strip():
    out = _render(
        "template/rpm_specfile.j2",
        phdr_normalize_cmd="NORMALIZE_CMD",
        disable_rpm_strip=False,
        install_prefix="/opt/rocm-7.14",
        rpm_scripts={"%pre": "", "%post": "", "%preun": "", "%postun": ""},
    )
    assert "%global __os_install_post %{__os_install_post}" in out
    assert "NORMALIZE_CMD %{buildroot}/opt/rocm-7.14" in out


def test_rpm_spec_skips_normalize_when_strip_disabled():
    out = _render(
        "template/rpm_specfile.j2",
        phdr_normalize_cmd="NORMALIZE_CMD",
        disable_rpm_strip=True,
        install_prefix="/opt/rocm-7.14",
        rpm_scripts={"%pre": "", "%post": "", "%preun": "", "%postun": ""},
    )
    assert "NORMALIZE_CMD" not in out


def test_debian_rules_injects_normalize_after_dh_strip():
    out = _render(
        "template/debian_rules.j2",
        phdr_normalize_cmd="NORMALIZE_CMD",
        disable_dh_strip=False,
        install_prefix="/opt/rocm-7.14",
        pkg_name="amdrocm-fft-test",
    )
    assert "dh_strip" in out
    assert out.index("dh_strip") < out.index("NORMALIZE_CMD"), "normalize must run after dh_strip"
    assert 'NORMALIZE_CMD "$(CURDIR)/debian/amdrocm-fft-test/opt/rocm-7.14"' in out


def test_debian_rules_skips_normalize_when_strip_disabled():
    out = _render(
        "template/debian_rules.j2",
        phdr_normalize_cmd="NORMALIZE_CMD",
        disable_dh_strip=True,
        install_prefix="/opt/rocm-7.14",
        pkg_name="amdrocm-fft-test",
    )
    assert "NORMALIZE_CMD" not in out
