# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_repo_package.py.

build_repo_package.py lives in build_tools/packaging/linux/, which is not on the
path that conftest.py sets up (it only adds build_tools/). Add it explicitly.
Importing the module is side-effect-free thanks to its ``__main__`` guard.
"""

import json
import sys
from pathlib import Path

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import build_repo_package as brp  # noqa: E402

PRERELEASE_BASE = "https://rocm.prereleases.amd.com/packages-multi-arch"
RELEASE_BASE = "https://repo.amd.com/rocm/packages-multi-arch"
NIGHTLY_BASE = "https://rocm.nightlies.amd.com/packages-multi-arch"
NIGHTLY_SUB = "20260716-12345"


def _args(*extra: str):
    """Parse args with sensible defaults, overridable by appending flags."""
    base = [
        "--os-profile",
        "ubuntu2404",
        "--release-type",
        "prerelease",
        "--repo-base-url",
        PRERELEASE_BASE,
        "--rocm-version",
        "7.14.0",
        "--dest-dir",
        "/tmp/out",
    ]
    return brp.parse_args(base + list(extra))


def _render(template: str, context: dict) -> str:
    return brp.get_jinja_env().get_template(template).render(context)


# --- URL construction ---------------------------------------------------------
#
# The release lines do not share a layout, so each one's exact URLs are pinned
# here: the public signed lines publish one repository per distro, and the
# nightly line publishes one per package type under a dated sub-folder. A URL
# that does not serve a repository only fails at the user's first metadata
# refresh, so these are pinned rather than derived.

RELEASE_EXPECTED = {
    "ubuntu2404": f"{RELEASE_BASE}/ubuntu2404/",
    "rhel8": f"{RELEASE_BASE}/rhel8/x86_64/",
    "rhel10": f"{RELEASE_BASE}/rhel10/x86_64/",
    "sles16": f"{RELEASE_BASE}/sles16/x86_64/",
}

PRERELEASE_EXPECTED = {
    "ubuntu2404": f"{PRERELEASE_BASE}/ubuntu2404/",
    "rhel8": f"{PRERELEASE_BASE}/rhel8/x86_64/",
    "rhel10": f"{PRERELEASE_BASE}/rhel10/x86_64/",
    "sles16": f"{PRERELEASE_BASE}/sles16/x86_64/",
}

NIGHTLY_EXPECTED = {
    "ubuntu2404": f"{NIGHTLY_BASE}/deb/{NIGHTLY_SUB}/",
    "rhel8": f"{NIGHTLY_BASE}/rpm/{NIGHTLY_SUB}/x86_64/",
    "rhel10": f"{NIGHTLY_BASE}/rpm/{NIGHTLY_SUB}/x86_64/",
    "sles16": f"{NIGHTLY_BASE}/rpm/{NIGHTLY_SUB}/x86_64/",
}


def _baseurl(profile: str, release_type: str, base: str, sub: str = "") -> str:
    return brp.repo_baseurl(
        base, brp.OS_PROFILES[profile]["pkg_type"], profile, release_type, sub
    )


@pytest.mark.parametrize("profile,expected", sorted(RELEASE_EXPECTED.items()))
def test_release_baseurl_is_per_distro(profile, expected):
    assert _baseurl(profile, "release", RELEASE_BASE) == expected


@pytest.mark.parametrize("profile,expected", sorted(PRERELEASE_EXPECTED.items()))
def test_prerelease_baseurl_is_per_distro(profile, expected):
    assert _baseurl(profile, "prerelease", PRERELEASE_BASE) == expected


@pytest.mark.parametrize("profile,expected", sorted(NIGHTLY_EXPECTED.items()))
def test_nightly_baseurl_is_flat_and_dated(profile, expected):
    assert _baseurl(profile, "nightly", NIGHTLY_BASE, NIGHTLY_SUB) == expected


def test_gpg_key_url_under_base():
    assert brp.gpg_key_url(PRERELEASE_BASE) == f"{PRERELEASE_BASE}/gpg/rocm.gpg"


def test_baseurl_tolerates_trailing_slash():
    assert (
        _baseurl("ubuntu2404", "release", RELEASE_BASE + "/")
        == RELEASE_EXPECTED["ubuntu2404"]
    )


def test_signed_lines_are_per_distro_never_flat():
    # A flat <base>/deb/ or <base>/rpm/x86_64/ is not served on the release CDN,
    # so a signed line must always carry the distro segment.
    for release_type, base in (
        ("release", RELEASE_BASE),
        ("prerelease", PRERELEASE_BASE),
    ):
        for profile in brp.OS_PROFILES:
            url = _baseurl(profile, release_type, base)
            assert f"/{profile}/" in url, f"{release_type} must be per-distro"
            assert not url.endswith("/deb/")
            assert "/rpm/x86_64/" not in url


def test_nightly_is_flat_and_carries_no_distro_segment():
    # The nightly repository has no per-distro tree, so it must stay flat.
    for profile in brp.OS_PROFILES:
        url = _baseurl(profile, "nightly", NIGHTLY_BASE, NIGHTLY_SUB)
        assert f"/{profile}/" not in url
        assert NIGHTLY_SUB in url


def test_gpg_key_url_is_not_the_domain_root():
    gpg = brp.gpg_key_url(PRERELEASE_BASE)
    assert "/packages-multi-arch/gpg/rocm.gpg" in gpg
    assert gpg != "https://rocm.prereleases.amd.com/gpg/rocm.gpg"


# --- repository reachability check --------------------------------------------


def test_repo_metadata_url_targets_the_index():
    deb = brp.repo_metadata_url(RELEASE_EXPECTED["ubuntu2404"], "deb")
    assert deb == f"{RELEASE_EXPECTED['ubuntu2404']}dists/{brp.DEB_SUITE}/Release"
    rpm = brp.repo_metadata_url(RELEASE_EXPECTED["rhel10"], "rpm")
    assert rpm == f"{RELEASE_EXPECTED['rhel10']}repodata/repomd.xml"


def test_verify_repo_url_accepts_a_served_repo(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"", RELEASE_EXPECTED["rhel10"], status=200),
    )
    brp.verify_repo_url(RELEASE_EXPECTED["rhel10"], "rpm")  # must not raise


def test_verify_repo_url_rejects_an_unserved_repo(monkeypatch):
    # The flat shape on the release line 404/403s; the guard must fail the build
    # rather than ship a package that cannot refresh.
    def not_found(*a, **k):
        raise brp.urllib.error.HTTPError(
            f"{RELEASE_BASE}/rpm/x86_64/", 403, "Forbidden", {}, None
        )

    monkeypatch.setattr(brp.urllib.request, "urlopen", not_found)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        brp.verify_repo_url(f"{RELEASE_BASE}/rpm/x86_64/", "rpm")


def test_verify_repo_url_does_not_retry_a_missing_repo(monkeypatch):
    # A wrong URL will not become right, so it must fail on the first attempt
    # rather than spending the full retry budget.
    calls = {"n": 0}

    def forbidden(*a, **k):
        calls["n"] += 1
        raise brp.urllib.error.HTTPError(
            f"{RELEASE_BASE}/rpm/x86_64/", 403, "Forbidden", {}, None
        )

    monkeypatch.setattr(brp.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(brp.time, "sleep", lambda s: pytest.fail("must not back off"))
    with pytest.raises(RuntimeError):
        brp.verify_repo_url(f"{RELEASE_BASE}/rpm/x86_64/", "rpm")
    assert calls["n"] == 1


def test_verify_repo_url_retries_rate_limiting(monkeypatch):
    calls = {"n": 0}

    def throttled(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise brp.urllib.error.HTTPError(
                RELEASE_EXPECTED["rhel10"], 429, "Too Many Requests", {}, None
            )
        return _FakeResponse(b"", RELEASE_EXPECTED["rhel10"], status=200)

    monkeypatch.setattr(brp.urllib.request, "urlopen", throttled)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    brp.verify_repo_url(RELEASE_EXPECTED["rhel10"], "rpm")
    assert calls["n"] == 2


def test_verify_repo_url_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise brp.urllib.error.URLError("transient")
        return _FakeResponse(b"", RELEASE_EXPECTED["rhel10"], status=200)

    monkeypatch.setattr(brp.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    brp.verify_repo_url(RELEASE_EXPECTED["rhel10"], "rpm")
    assert calls["n"] == 3


# --- reachability check wiring -------------------------------------------------
#
# A guard that is never invoked is indistinguishable from no guard, so pin the
# call from main() rather than only testing the function in isolation.


def _main_argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--os-profile",
        "rhel10",
        "--release-type",
        "release",
        "--repo-base-url",
        RELEASE_BASE,
        "--rocm-version",
        "7.14.0",
        "--dest-dir",
        str(tmp_path),
        *extra,
    ]


def test_verify_repo_url_is_off_by_default():
    assert _args().verify_repo_url is False


def test_main_verifies_the_repository_when_asked(monkeypatch, tmp_path):
    checked = []
    monkeypatch.setattr(
        brp, "verify_repo_url", lambda url, pkg_type: checked.append((url, pkg_type))
    )
    monkeypatch.setattr(brp, "build_rpm_package", lambda *a, **k: None)
    brp.main(_main_argv(tmp_path, "--verify-repo-url"))
    # The URL actually built for this profile is what gets checked.
    assert checked == [(RELEASE_EXPECTED["rhel10"], "rpm")]


def test_main_skips_verification_for_nightly(monkeypatch, tmp_path):
    # CI passes the flag unconditionally; the nightly repository is published by
    # the same run, so its dated sub-folder cannot be checked at build time.
    monkeypatch.setattr(
        brp,
        "verify_repo_url",
        lambda *a: pytest.fail("nightly has nothing to verify yet"),
    )
    monkeypatch.setattr(brp, "build_rpm_package", lambda *a, **k: None)
    brp.main(
        [
            "--os-profile",
            "rhel10",
            "--release-type",
            "nightly",
            "--repo-base-url",
            NIGHTLY_BASE,
            "--repo-sub-folder",
            NIGHTLY_SUB,
            "--rocm-version",
            "7.14.0",
            "--dest-dir",
            str(tmp_path),
            "--verify-repo-url",
        ]
    )


def test_main_does_not_verify_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        brp,
        "verify_repo_url",
        lambda *a: pytest.fail("must not reach the network unless asked"),
    )
    monkeypatch.setattr(brp, "build_rpm_package", lambda *a, **k: None)
    brp.main(_main_argv(tmp_path))


# --- context wiring -----------------------------------------------------------


@pytest.mark.parametrize("profile", sorted(RELEASE_EXPECTED))
def test_build_context_threads_profile_and_release_type(profile):
    # The baseurl depends on both the target distro and the release line, so
    # build_context must pass both through rather than defaulting either.
    args = _args(
        "--os-profile",
        profile,
        "--release-type",
        "release",
        "--repo-base-url",
        RELEASE_BASE,
    )
    ctx = brp.build_context(args, brp.OS_PROFILES[profile])
    assert ctx["baseurl"] == RELEASE_EXPECTED[profile]


# --- signing ------------------------------------------------------------------


def test_signed_lines():
    assert brp.is_signed("prerelease") is True
    assert brp.is_signed("release") is True
    assert brp.is_signed("nightly") is False


# --- signing-key fetch hardening ----------------------------------------------


class _FakeResponse:
    """Minimal stand-in for a urlopen() result (context manager + read)."""

    def __init__(self, body: bytes, url: str, status: int = 200):
        self._body = body
        self._url = url
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


def test_require_https_accepts_https():
    brp._require_https("https://example.com/gpg/rocm.gpg", "key URL")


def test_require_https_rejects_http():
    with pytest.raises(ValueError):
        brp._require_https("http://example.com/gpg/rocm.gpg", "key URL")


def test_load_signing_key_rejects_non_https_base(monkeypatch):
    # A non-https base must fail before any network fetch is attempted.
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not fetch over a non-https URL"),
    )
    args = _args()
    args.repo_base_url = "http://rocm.example.com/packages-multi-arch"
    with pytest.raises(ValueError):
        brp.load_signing_key(args)


def test_load_signing_key_rejects_post_redirect_downgrade(monkeypatch):
    # An https URL that redirects to http must be rejected after the fetch.
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"KEY", "http://evil.example/gpg/rocm.gpg"),
    )
    with pytest.raises(ValueError):
        brp.load_signing_key(_args())


def test_load_signing_key_rejects_oversize_key(monkeypatch):
    big = b"A" * (brp.MAX_GPG_KEY_BYTES + 1)
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(big, f"{PRERELEASE_BASE}/gpg/rocm.gpg"),
    )
    with pytest.raises(ValueError):
        brp.load_signing_key(_args())


def test_load_signing_key_returns_fetched_key(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(
            b"ARMORED KEY", f"{PRERELEASE_BASE}/gpg/rocm.gpg"
        ),
    )
    # Fingerprint verification runs gpg on real key bytes; stub it here.
    monkeypatch.setattr(brp, "_verify_key_fingerprint", lambda key: None)
    assert brp.load_signing_key(_args()) == b"ARMORED KEY"


def test_load_signing_key_prefers_key_file(monkeypatch, tmp_path):
    key = tmp_path / "rocm.gpg"
    key.write_bytes(b"FILE KEY")
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("must not fetch when --gpg-key-file is given"),
    )
    args = _args()
    args.gpg_key_file = key
    assert brp.load_signing_key(args) == b"FILE KEY"


def test_load_signing_key_rejects_wrong_fingerprint(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"KEY", f"{PRERELEASE_BASE}/gpg/rocm.gpg"),
    )
    monkeypatch.setattr(brp, "_key_fingerprint", lambda armored: "0" * 40)
    with pytest.raises(ValueError):
        brp.load_signing_key(_args())


def test_verify_key_fingerprint_accepts_pinned(monkeypatch):
    monkeypatch.setattr(
        brp, "_key_fingerprint", lambda armored: brp.EXPECTED_KEY_FINGERPRINT
    )
    brp._verify_key_fingerprint(b"whatever")  # must not raise


def test_fetch_signing_key_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise brp.urllib.error.URLError("transient")
        return _FakeResponse(b"KEY", f"{PRERELEASE_BASE}/gpg/rocm.gpg")

    monkeypatch.setattr(brp.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)  # no real backoff
    assert brp._fetch_signing_key(f"{PRERELEASE_BASE}/gpg/rocm.gpg") == b"KEY"
    assert calls["n"] == 3


def test_fetch_signing_key_gives_up_after_attempts(monkeypatch):
    def always_fail(*a, **k):
        raise brp.urllib.error.URLError("down")

    monkeypatch.setattr(brp.urllib.request, "urlopen", always_fail)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        brp._fetch_signing_key(f"{PRERELEASE_BASE}/gpg/rocm.gpg")


# --- versioning ---------------------------------------------------------------


def test_rpm_version_rolling_for_prerelease():
    assert brp.rpm_version_release("prerelease", "7.14.0", "") == (
        "7.14.0",
        "1.prerelease",
    )


def test_streams_never_share_a_package_version():
    # Two lines at the same ROCm version must not produce the same package:
    # installing one over the other would otherwise report success and silently
    # leave the original repository configured.
    version = "7.14.0"
    rpm_pre = brp.rpm_version_release("prerelease", version, "")
    rpm_rel = brp.rpm_version_release("release", version, "")
    assert rpm_pre != rpm_rel
    assert brp.deb_version("prerelease", version, "") != brp.deb_version(
        "release", version, ""
    )


def test_prerelease_sorts_before_release():
    # The release package must upgrade over the prerelease one, not be seen as a
    # downgrade. deb: "~" sorts before the plain version. rpm: the release field
    # differs and "prerelease" precedes "release".
    assert brp.deb_version("prerelease", "7.14.0", "") == "7.14.0~prerelease"
    assert brp.deb_version("release", "7.14.0", "") == "7.14.0"
    assert brp.rpm_version_release("prerelease", "7.14.0", "")[1] < (
        brp.rpm_version_release("release", "7.14.0", "")[1]
    )


def test_rpm_version_nightly_splits_date_and_id():
    version, release = brp.rpm_version_release("nightly", "7.14.0", NIGHTLY_SUB)
    assert version == "20260716"
    assert release == "12345.nightly"
    assert "-" not in version  # rpm Version cannot contain a hyphen


def test_deb_version_nightly_has_no_hyphen():
    assert brp.deb_version("nightly", "7.14.0", NIGHTLY_SUB) == "20260716.12345"


def test_deb_version_rolling():
    assert brp.deb_version("release", "7.14.0", "") == "7.14.0"


# --- CLI validation -----------------------------------------------------------


def test_release_type_rejects_unknown():
    with pytest.raises(SystemExit):
        _args("--release-type", "dev")


def test_repo_base_url_required():
    with pytest.raises(SystemExit):
        brp.parse_args(
            [
                "--os-profile",
                "ubuntu2404",
                "--release-type",
                "prerelease",
                "--rocm-version",
                "7.14.0",
                "--dest-dir",
                "/tmp/out",
            ]
        )


def test_nightly_requires_sub_folder():
    with pytest.raises(SystemExit):
        _args("--release-type", "nightly", "--repo-base-url", NIGHTLY_BASE)


def test_nightly_with_sub_folder_ok():
    args = _args(
        "--release-type",
        "nightly",
        "--repo-base-url",
        NIGHTLY_BASE,
        "--repo-sub-folder",
        NIGHTLY_SUB,
    )
    assert args.repo_sub_folder == NIGHTLY_SUB


def test_sub_folder_rejected_for_non_nightly():
    with pytest.raises(SystemExit):
        _args("--repo-sub-folder", NIGHTLY_SUB)  # default release-type is prerelease


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad_version",
    [
        '7.14.0"; rm -rf /; echo "',  # shell-injection payload
        "7.14.0 evil",  # whitespace
        "7.14.0$(id)",  # command substitution
        "7.14.0\ngpgcheck=0",  # embedded newline
        "7.14.0\n",  # trailing newline (regex must use \\Z, not $)
        "",  # empty
        "-7.14.0",  # must start with a digit
    ],
)
def test_rocm_version_rejects_bad_format(bad_version):
    with pytest.raises(SystemExit):
        _args("--rocm-version", bad_version)


@pytest.mark.parametrize("good_version", ["7.14.0", "7.14.0~rc3", "8.0.1rc1", "7.0"])
def test_rocm_version_accepts_valid(good_version):
    assert _args("--rocm-version", good_version).rocm_version == good_version


@pytest.mark.parametrize(
    "bad_url",
    [
        f"{PRERELEASE_BASE}\ngpgcheck=0",  # newline directive injection
        f"{PRERELEASE_BASE} extra",  # whitespace
        "not a url",  # no scheme/netloc
        "ftp://example.com/repo",  # unsupported scheme
    ],
)
def test_repo_base_url_rejects_bad(bad_url):
    with pytest.raises(SystemExit):
        _args("--repo-base-url", bad_url)


def test_repo_base_url_accepts_https():
    assert _args("--repo-base-url", PRERELEASE_BASE).repo_base_url == PRERELEASE_BASE


def test_repo_sub_folder_rejects_bad_format():
    with pytest.raises(SystemExit):
        _args(
            "--release-type",
            "nightly",
            "--repo-base-url",
            NIGHTLY_BASE,
            "--repo-sub-folder",
            "20260722-99\nX",  # newline in the dated sub-folder
        )


def test_postinst_does_not_interpolate_version():
    # The postinst runs as root at install time; the version must not be rendered
    # into it (defense-in-depth beyond input validation).
    ctx = brp.build_context(
        _args("--rocm-version", "9.9.9"), brp.OS_PROFILES["ubuntu2404"]
    )
    out = _render("template/repo/deb/postinst.j2", ctx)
    assert "9.9.9" not in out
    assert "AMD ROCm repository configured." in out


# --- rpm .repo rendering ------------------------------------------------------


def _rpm_repo(release_type: str, base: str, sub: str = "") -> str:
    extra = [
        "--os-profile",
        "rhel10",
        "--release-type",
        release_type,
        "--repo-base-url",
        base,
    ]
    if sub:
        extra += ["--repo-sub-folder", sub]
    args = _args(*extra)
    return _render(
        "template/repo/rpm/amdrocm.repo.j2",
        brp.build_context(args, brp.OS_PROFILES["rhel10"]),
    )


def test_rpm_repo_signed_has_gpgcheck_and_gpgkey():
    out = _rpm_repo("prerelease", PRERELEASE_BASE)
    assert "[amdrocm]" in out
    assert "enabled=1" in out
    assert "gpgcheck=1" in out
    assert "gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rocm" in out
    assert f"baseurl={PRERELEASE_EXPECTED['rhel10']}" in out


def test_rpm_repo_nightly_unsigned_no_gpgkey():
    out = _rpm_repo("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "gpgcheck=0" in out
    assert "gpgkey" not in out


def test_rpm_repo_ends_with_newline():
    assert _rpm_repo("prerelease", PRERELEASE_BASE).endswith("\n")


# --- deb .sources rendering ---------------------------------------------------


def _deb_sources(release_type: str, base: str, sub: str = "") -> str:
    extra = [
        "--os-profile",
        "ubuntu2404",
        "--release-type",
        release_type,
        "--repo-base-url",
        base,
    ]
    if sub:
        extra += ["--repo-sub-folder", sub]
    args = _args(*extra)
    return _render(
        "template/repo/deb/amdrocm.sources.j2",
        brp.build_context(args, brp.OS_PROFILES["ubuntu2404"]),
    )


def test_deb_sources_signed_fields():
    out = _deb_sources("prerelease", PRERELEASE_BASE)
    assert "Types: deb" in out
    assert f"URIs: {PRERELEASE_EXPECTED['ubuntu2404']}" in out
    # The suite is rendered from the constant the reachability check also uses,
    # so the repo file and the verified index path cannot drift apart.
    assert f"Suites: {brp.DEB_SUITE}" in out
    assert "Suites: stable" in out
    assert "Components: main" in out
    assert "Architectures: amd64" in out
    assert "Signed-By: /usr/share/keyrings/rocm.gpg" in out
    assert "Trusted:" not in out


def test_deb_sources_nightly_trusted_not_signed():
    out = _deb_sources("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "Trusted: yes" in out
    assert "Signed-By:" not in out


# --- rpm spec rendering -------------------------------------------------------


def _spec(release_type: str, base: str, sub: str = "") -> str:
    extra = [
        "--os-profile",
        "rhel10",
        "--release-type",
        release_type,
        "--repo-base-url",
        base,
    ]
    if sub:
        extra += ["--repo-sub-folder", sub]
    args = _args(*extra)
    ctx = brp.build_context(args, brp.OS_PROFILES["rhel10"])
    version, release = brp.rpm_version_release(release_type, args.rocm_version, sub)
    return _render(
        "template/repo/rpm/amdrocm-repo.spec.j2",
        {
            **ctx,
            "version": version,
            "release": release,
            "changelog_date": "Mon Jul 20 2026",
        },
    )


def test_spec_name_and_repo_install():
    out = _spec("prerelease", PRERELEASE_BASE)
    assert "Name: amdrocm-repo" in out
    assert "Version: 7.14.0" in out
    assert "/etc/yum.repos.d/amdrocm.repo" in out


def test_spec_signed_installs_key_nightly_does_not():
    signed = _spec("prerelease", PRERELEASE_BASE)
    assert "/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm" in signed
    unsigned = _spec("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "RPM-GPG-KEY-rocm" not in unsigned


def test_spec_has_no_conflicts():
    assert "Conflicts" not in _spec("prerelease", PRERELEASE_BASE)


# --- deb control / install rendering ------------------------------------------


def test_control_has_no_key_runtime_deps():
    args = _args()
    ctx = brp.build_context(args, brp.OS_PROFILES["ubuntu2404"])
    out = _render("template/repo/deb/control.j2", ctx)
    assert "Package: amdrocm-repo" in out
    assert "gnupg" not in out
    assert "wget" not in out
    assert "Conflicts" not in out


def test_install_maps_keyring_only_when_signed():
    signed = brp.build_context(_args(), brp.OS_PROFILES["ubuntu2404"])
    out_signed = _render("template/repo/deb/install.j2", signed)
    assert "amdrocm.sources /etc/apt/sources.list.d/" in out_signed
    assert "rocm.gpg /usr/share/keyrings/" in out_signed

    args_n = _args(
        "--release-type",
        "nightly",
        "--repo-base-url",
        NIGHTLY_BASE,
        "--repo-sub-folder",
        NIGHTLY_SUB,
    )
    out_nightly = _render(
        "template/repo/deb/install.j2",
        brp.build_context(args_n, brp.OS_PROFILES["ubuntu2404"]),
    )
    assert "rocm.gpg" not in out_nightly


# --- list-profiles --------------------------------------------------------


def test_list_profiles_rpm():
    assert brp.list_profiles("rpm") == ["rhel8", "rhel10", "sles16"]


def test_list_profiles_deb():
    assert brp.list_profiles("deb") == ["ubuntu2404"]


def test_list_profiles_cli_emits_json(capsys):
    brp.run_list_profiles(["--pkg-type", "rpm"])
    out = capsys.readouterr().out
    assert json.loads(out) == ["rhel8", "rhel10", "sles16"]


def test_list_profiles_cli_rejects_invalid_pkg_type(capsys):
    with pytest.raises(SystemExit):
        brp.run_list_profiles(["--pkg-type", "foo"])


def test_main_dispatches_list_profiles(capsys):
    brp.main(["list-profiles", "--pkg-type", "deb"])
    out = capsys.readouterr().out
    assert json.loads(out) == ["ubuntu2404"]


# --- dest-dir normalization -----------------------------------------------


def test_relative_dest_dir_is_resolved_to_absolute():
    # rpmbuild needs an absolute _topdir, so a relative --dest-dir must be
    # normalized to absolute at parse time.
    args = brp.parse_args(
        [
            "--os-profile",
            "rhel10",
            "--release-type",
            "prerelease",
            "--repo-base-url",
            PRERELEASE_BASE,
            "--rocm-version",
            "7.14.0",
            "--dest-dir",
            "relout",
        ]
    )
    assert args.dest_dir.is_absolute()
    assert args.dest_dir.name == "relout"


def test_absolute_dest_dir_stays_absolute():
    args = _args("--dest-dir", "/tmp/somewhere")
    assert args.dest_dir.is_absolute()
    assert args.dest_dir.name == "somewhere"
