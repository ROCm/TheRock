# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_repo_package.py.

build_repo_package.py lives in build_tools/packaging/linux/, which is not on the
path that conftest.py sets up (it only adds build_tools/). Add it explicitly.
Importing the module is side-effect-free thanks to its ``__main__`` guard.
"""

import json
import locale
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import build_repo_package as brp  # noqa: E402

STABLE_BASE = "https://stable.repo.amd.com/rocm/core/packages"
NIGHTLY_BASE = "https://nightly.repo.amd.com/rocm/core/packages"
# The signing key, supplied whole. It sits beside core/ rather than under
# STABLE_BASE, which is why it is a separate input rather than derived from it.
STABLE_KEY_URL = "https://stable.repo.amd.com/rocm/gpg/packages.gpg"
NIGHTLY_SUB = "20260716-12345"


def _args(*extra: str):
    """Parse args with sensible defaults, overridable by appending flags."""
    base = [
        "--os-profile",
        "ubuntu2404",
        "--stream",
        "stable",
        "--repo-base-url",
        STABLE_BASE,
        "--gpg-key-url",
        STABLE_KEY_URL,
        "--rocm-version",
        "10.0.0",
        "--dest-dir",
        "/tmp/out",
    ]
    return brp.parse_args(base + list(extra))


def _render(template: str, context: dict) -> str:
    return brp.get_jinja_env().get_template(template).render(context)


# --- URL construction ---------------------------------------------------------
#
# Every stream is served per distro. The two shapes differ only by a build-id
# segment:
#
#   flat      <base>/<distro>/[x86_64/]
#   build_id  <base>/<distro>/<YYYYMMDD-id>/[x86_64/]
#
# These are pinned rather than derived, and the values match the shapes the
# published install instructions use (scriptgen's unit-test/test_stream_urls.py,
# checked against the live repos). A URL that does not serve a repository only
# fails at the user's first metadata refresh, so a derived expectation would
# reproduce a builder bug instead of catching it.

STABLE_EXPECTED = {
    "ubuntu2404": f"{STABLE_BASE}/ubuntu2404/",
    "rhel8": f"{STABLE_BASE}/rhel8/x86_64/",
    "rhel10": f"{STABLE_BASE}/rhel10/x86_64/",
    "sles16": f"{STABLE_BASE}/sles16/x86_64/",
}

NIGHTLY_EXPECTED = {
    "ubuntu2404": f"{NIGHTLY_BASE}/ubuntu2404/{NIGHTLY_SUB}/",
    "rhel8": f"{NIGHTLY_BASE}/rhel8/{NIGHTLY_SUB}/x86_64/",
    "rhel10": f"{NIGHTLY_BASE}/rhel10/{NIGHTLY_SUB}/x86_64/",
    "sles16": f"{NIGHTLY_BASE}/sles16/{NIGHTLY_SUB}/x86_64/",
}


def _baseurl(profile: str, stream: str, base: str, sub: str = "") -> str:
    return brp.repo_baseurl(
        base, brp.OS_PROFILES[profile]["pkg_type"], profile, stream, sub
    )


@pytest.mark.parametrize("profile,expected", sorted(STABLE_EXPECTED.items()))
def test_stable_baseurl_is_per_distro(profile, expected):
    assert _baseurl(profile, "stable", STABLE_BASE) == expected


@pytest.mark.parametrize("profile,expected", sorted(NIGHTLY_EXPECTED.items()))
def test_nightly_baseurl_is_per_distro_and_carries_the_build_id(profile, expected):
    assert _baseurl(profile, "nightly", NIGHTLY_BASE, NIGHTLY_SUB) == expected


def test_gpg_key_url_is_outside_the_packages_base():
    # The key sits beside core/ while packages live under core/packages/, so it
    # cannot be reached by appending to the base. This is the whole reason the
    # key URL is its own argument, so pin the relationship rather than trusting
    # the comment that says so.
    assert not STABLE_KEY_URL.startswith(STABLE_BASE)
    # Nor by trimming: the key is not under the base's parent either.
    assert not STABLE_KEY_URL.startswith(STABLE_BASE.rsplit("/", 1)[0])


def test_key_url_is_used_verbatim(monkeypatch):
    # Taking the URL whole -- rather than appending a hardcoded gpg/packages.gpg
    # to a root -- is what keeps the key's location a publisher decision. A
    # builder that still derived the tail would fetch a different URL here.
    seen = []
    monkeypatch.setattr(brp, "_fetch_signing_key", lambda url: seen.append(url) or b"k")
    monkeypatch.setattr(brp, "_verify_key_fingerprint", lambda key: None)
    args = _args()
    args.gpg_key_url = "https://example.invalid/somewhere/else/custom-key.gpg"
    brp.load_signing_key(args)
    assert seen == ["https://example.invalid/somewhere/else/custom-key.gpg"]


def test_baseurl_tolerates_trailing_slash():
    assert (
        _baseurl("ubuntu2404", "stable", STABLE_BASE + "/")
        == STABLE_EXPECTED["ubuntu2404"]
    )


def test_every_stream_is_per_distro_never_per_format():
    # Both the per-distro and the per-format (<base>/deb/, <base>/rpm/) trees are
    # served on every populated stream, so a wrong choice cannot be caught by
    # fetching one URL. Pin the per-distro shape: it is what the published
    # install instructions configure, and a stream that ever drops the
    # per-format alias would break in the field rather than in CI.
    for stream, base, sub in (
        ("stable", STABLE_BASE, ""),
        ("nightly", NIGHTLY_BASE, NIGHTLY_SUB),
    ):
        for profile in brp.OS_PROFILES:
            url = _baseurl(profile, stream, base, sub)
            assert f"/{profile}/" in url, f"{stream} must be per-distro"
            assert not url.endswith("/deb/")
            assert "/rpm/x86_64/" not in url


def test_only_the_build_id_shape_carries_a_sub_folder():
    for profile in brp.OS_PROFILES:
        assert NIGHTLY_SUB in _baseurl(profile, "nightly", NIGHTLY_BASE, NIGHTLY_SUB)
        assert NIGHTLY_SUB not in _baseurl(profile, "stable", STABLE_BASE)


def test_signedness_follows_the_shape():
    # Not an independent axis: the flat streams are the ones publishing
    # InRelease/Release.gpg and repomd.xml.asc.
    for stream in brp.STREAMS:
        assert brp.is_signed(stream) is (brp.stream_shape(stream) == brp._FLAT)
    assert brp.is_signed("stable") is True
    assert brp.is_signed("nightly") is False


def test_unknown_stream_is_rejected_before_it_reaches_a_url():
    with pytest.raises(ValueError):
        brp.repo_baseurl(STABLE_BASE, "deb", "ubuntu2404", "bogus", "")
    with pytest.raises(ValueError):
        brp.repo_id("bogus")


def test_a_sub_folder_mismatched_to_the_shape_is_rejected():
    # Dropping a missing sub-folder instead of raising would emit
    # <base>/<distro>/ for nightly -- identical in shape to a correct stable URL
    # and serving nothing, so the package would install and fail on first
    # refresh. parse_args blocks this today; the guard belongs here too, because
    # the flat-looking URL is only wrong once it reaches a user.
    with pytest.raises(ValueError):
        brp.repo_baseurl(NIGHTLY_BASE, "deb", "ubuntu2404", "nightly", "")
    with pytest.raises(ValueError):
        brp.repo_baseurl(STABLE_BASE, "deb", "ubuntu2404", "stable", NIGHTLY_SUB)


# --- repository reachability check --------------------------------------------


def test_repo_metadata_url_targets_the_index():
    deb = brp.repo_metadata_url(STABLE_EXPECTED["ubuntu2404"], "deb")
    assert deb == f"{STABLE_EXPECTED['ubuntu2404']}dists/{brp.DEB_SUITE}/Release"
    rpm = brp.repo_metadata_url(STABLE_EXPECTED["rhel10"], "rpm")
    assert rpm == f"{STABLE_EXPECTED['rhel10']}repodata/repomd.xml"


def test_verify_repo_url_accepts_a_served_repo(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"", STABLE_EXPECTED["rhel10"], status=200),
    )
    brp.verify_repo_url(STABLE_EXPECTED["rhel10"], "rpm")  # must not raise


def test_verify_repo_url_rejects_an_unserved_repo(monkeypatch):
    # A base that serves no repository 404/403s; the guard must fail the build
    # rather than ship a package that cannot refresh.
    def not_found(*a, **k):
        raise brp.urllib.error.HTTPError(
            f"{STABLE_BASE}/rpm/x86_64/", 403, "Forbidden", {}, None
        )

    monkeypatch.setattr(brp.urllib.request, "urlopen", not_found)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        brp.verify_repo_url(f"{STABLE_BASE}/rpm/x86_64/", "rpm")


def test_verify_repo_url_does_not_retry_a_missing_repo(monkeypatch):
    # A wrong URL will not become right, so it must fail on the first attempt
    # rather than spending the full retry budget.
    calls = {"n": 0}

    def forbidden(*a, **k):
        calls["n"] += 1
        raise brp.urllib.error.HTTPError(
            f"{STABLE_BASE}/rpm/x86_64/", 403, "Forbidden", {}, None
        )

    monkeypatch.setattr(brp.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(brp.time, "sleep", lambda s: pytest.fail("must not back off"))
    with pytest.raises(RuntimeError):
        brp.verify_repo_url(f"{STABLE_BASE}/rpm/x86_64/", "rpm")
    assert calls["n"] == 1


def test_verify_repo_url_does_not_retry_a_non_200_response(monkeypatch):
    # urlopen raises for the error codes, so a status that arrives here is a URL
    # that answers but serves no repository index. Retrying cannot change that.
    # Before this was fixed the status was recorded and the loop slept through
    # every remaining attempt before failing anyway.
    calls = {"n": 0}

    def created(*a, **k):
        calls["n"] += 1
        return _FakeResponse(b"", STABLE_EXPECTED["rhel10"], status=201)

    monkeypatch.setattr(brp.urllib.request, "urlopen", created)
    monkeypatch.setattr(brp.time, "sleep", lambda s: pytest.fail("must not back off"))
    with pytest.raises(RuntimeError, match="201"):
        brp.verify_repo_url(STABLE_EXPECTED["rhel10"], "rpm")
    assert calls["n"] == 1


def test_verify_repo_url_retries_rate_limiting(monkeypatch):
    calls = {"n": 0}

    def throttled(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise brp.urllib.error.HTTPError(
                STABLE_EXPECTED["rhel10"], 429, "Too Many Requests", {}, None
            )
        return _FakeResponse(b"", STABLE_EXPECTED["rhel10"], status=200)

    monkeypatch.setattr(brp.urllib.request, "urlopen", throttled)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    brp.verify_repo_url(STABLE_EXPECTED["rhel10"], "rpm")
    assert calls["n"] == 2


def test_verify_repo_url_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise brp.urllib.error.URLError("transient")
        return _FakeResponse(b"", STABLE_EXPECTED["rhel10"], status=200)

    monkeypatch.setattr(brp.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    brp.verify_repo_url(STABLE_EXPECTED["rhel10"], "rpm")
    assert calls["n"] == 3


# --- reachability check wiring -------------------------------------------------
#
# A guard that is never invoked is indistinguishable from no guard, so pin the
# call from main() rather than only testing the function in isolation.


def _main_argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--os-profile",
        "rhel10",
        "--stream",
        "stable",
        "--repo-base-url",
        STABLE_BASE,
        "--gpg-key-url",
        STABLE_KEY_URL,
        "--rocm-version",
        "10.0.0",
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
    assert checked == [(STABLE_EXPECTED["rhel10"], "rpm")]


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
            "--stream",
            "nightly",
            "--repo-base-url",
            NIGHTLY_BASE,
            "--repo-sub-folder",
            NIGHTLY_SUB,
            "--rocm-version",
            "10.0.0",
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


# The other main() tests all use an rpm profile, so without this one the deb
# half of the builder dispatch is never taken and could be deleted outright
# without failing the suite.
@pytest.mark.parametrize(
    "os_profile,expected",
    [("ubuntu2404", "deb"), ("rhel10", "rpm")],
)
def test_main_dispatches_on_the_profile_package_type(
    monkeypatch, tmp_path, os_profile, expected
):
    built = []
    monkeypatch.setattr(brp, "build_deb_package", lambda *a, **k: built.append("deb"))
    monkeypatch.setattr(brp, "build_rpm_package", lambda *a, **k: built.append("rpm"))
    brp.main(
        [
            "--os-profile",
            os_profile,
            "--stream",
            "stable",
            "--repo-base-url",
            STABLE_BASE,
            "--gpg-key-url",
            STABLE_KEY_URL,
            "--rocm-version",
            "10.0.0",
            "--dest-dir",
            str(tmp_path),
        ]
    )
    assert built == [expected]


# --- context wiring -----------------------------------------------------------


@pytest.mark.parametrize("profile", sorted(STABLE_EXPECTED))
def test_build_context_threads_profile_and_stream(profile):
    # The baseurl depends on both the target distro and the stream, so
    # build_context must pass both through rather than defaulting either.
    args = _args(
        "--os-profile",
        profile,
        "--stream",
        "stable",
        "--repo-base-url",
        STABLE_BASE,
    )
    ctx = brp.build_context(args, brp.OS_PROFILES[profile])
    assert ctx["baseurl"] == STABLE_EXPECTED[profile]


# --- signing ------------------------------------------------------------------


def test_signed_lines():
    assert brp.is_signed("stable") is True
    assert brp.is_signed("stable") is True
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
    args.gpg_key_url = "http://rocm.example.com/rocm/gpg/packages.gpg"
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
        lambda *a, **k: _FakeResponse(big, f"{STABLE_BASE}/gpg/rocm.gpg"),
    )
    with pytest.raises(ValueError):
        brp.load_signing_key(_args())


def test_load_signing_key_returns_fetched_key(monkeypatch):
    monkeypatch.setattr(
        brp.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b"ARMORED KEY", f"{STABLE_BASE}/gpg/rocm.gpg"),
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
        lambda *a, **k: _FakeResponse(b"KEY", f"{STABLE_BASE}/gpg/rocm.gpg"),
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
        return _FakeResponse(b"KEY", f"{STABLE_BASE}/gpg/rocm.gpg")

    monkeypatch.setattr(brp.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)  # no real backoff
    assert brp._fetch_signing_key(f"{STABLE_BASE}/gpg/rocm.gpg") == b"KEY"
    assert calls["n"] == 3


def test_fetch_signing_key_gives_up_after_attempts(monkeypatch):
    def always_fail(*a, **k):
        raise brp.urllib.error.URLError("down")

    monkeypatch.setattr(brp.urllib.request, "urlopen", always_fail)
    monkeypatch.setattr(brp.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        brp._fetch_signing_key(f"{STABLE_BASE}/gpg/rocm.gpg")


# --- versioning ---------------------------------------------------------------


def test_rpm_version_rolling_for_a_flat_stream():
    assert brp.rpm_version_release("stable", "10.0.0", "") == ("10.0.0", "1.stable")


def test_streams_never_share_a_package_version():
    # Two streams at the same ROCm version must not produce the same package:
    # installing one over the other would otherwise report success and silently
    # leave the original repository configured.
    version = "10.0.0"
    seen_rpm = set()
    seen_deb = set()
    for stream in brp.STREAMS:
        sub = NIGHTLY_SUB if brp.stream_shape(stream) == brp._BUILD_ID else ""
        seen_rpm.add(brp.rpm_version_release(stream, version, sub))
        seen_deb.add(brp.deb_version(stream, version, sub))
    assert len(seen_rpm) == len(brp.STREAMS)
    assert len(seen_deb) == len(brp.STREAMS)


def test_stable_carries_the_plain_version_and_the_stream_in_the_rpm_release():
    # stable is the GA stream, so its deb version is the bare ROCm version; any
    # other flat stream would carry a "~<stream>" suffix, which sorts before it.
    assert brp.deb_version("stable", "10.0.0", "") == "10.0.0"
    assert brp.rpm_version_release("stable", "10.0.0", "")[1] == "1.stable"


def test_a_build_id_stream_outranks_a_flat_one():
    # Documents a known consequence rather than asserting a fix. rpm compares
    # Version before Release, and a build_id Version is a date (20260716) while
    # a flat one is a semantic version (10.0.0), so nightly always sorts above
    # stable and switching nightly -> stable is a downgrade. This predates the
    # streams and resolves itself if the streams ever become separate packages.
    nightly_v, _ = brp.rpm_version_release("nightly", "10.0.0", NIGHTLY_SUB)
    stable_v, _ = brp.rpm_version_release("stable", "10.0.0", "")
    assert nightly_v == "20260716"
    assert stable_v == "10.0.0"
    # Compare the way rpm does -- leading numeric segments, numerically.
    assert int(nightly_v.split(".")[0]) > int(stable_v.split(".")[0])


def test_rpm_version_nightly_splits_date_and_id():
    version, release = brp.rpm_version_release("nightly", "10.0.0", NIGHTLY_SUB)
    assert version == "20260716"
    assert release == "12345.nightly"
    assert "-" not in version  # rpm Version cannot contain a hyphen


def test_deb_version_nightly_has_no_hyphen():
    assert brp.deb_version("nightly", "10.0.0", NIGHTLY_SUB) == "20260716.12345"


def test_deb_version_rolling():
    assert brp.deb_version("stable", "10.0.0", "") == "10.0.0"


# The rpm %changelog date must stay English on any build machine. strftime and
# calendar.day_abbr both follow the process locale, so the names are pinned in
# module constants; these two tests are what keep anyone from reverting to
# either of those.


def test_rpm_changelog_date_reads_the_pinned_names(monkeypatch):
    # Runs everywhere, unlike the locale test below. Swapping the constants for
    # sentinels proves the value comes from them and not from the C library: a
    # strftime-based implementation would ignore the patch and still say "Thu".
    monkeypatch.setattr(brp, "_CHANGELOG_DAY_ABBR", ("D0",) * 7)
    monkeypatch.setattr(brp, "_CHANGELOG_MONTH_ABBR", ("M0",) * 12)
    assert (
        brp.rpm_changelog_date(datetime(2026, 7, 30, tzinfo=timezone.utc))
        == "D0 M0 30 2026"
    )


def test_rpm_changelog_date_is_english_under_a_foreign_locale():
    # The real end-to-end check, but it only runs where a German locale has been
    # generated. CI has not: a stock ubuntu image carries C, C.utf8 and POSIX
    # only. Under C the pinned and strftime implementations agree, which is why
    # the sentinel test above carries the actual regression guard.
    saved = locale.setlocale(locale.LC_ALL)
    try:
        try:
            locale.setlocale(locale.LC_ALL, "de_DE.UTF-8")
        except locale.Error:
            pytest.skip("de_DE.UTF-8 not generated on this machine")
        assert (
            brp.rpm_changelog_date(datetime(2026, 7, 30, tzinfo=timezone.utc))
            == "Thu Jul 30 2026"
        )
    finally:
        locale.setlocale(locale.LC_ALL, saved)


# --- CLI validation -----------------------------------------------------------


def test_stream_rejects_unknown():
    with pytest.raises(SystemExit):
        _args("--stream", "dev")


def test_repo_base_url_required():
    with pytest.raises(SystemExit):
        brp.parse_args(
            [
                "--os-profile",
                "ubuntu2404",
                "--stream",
                "stable",
                "--rocm-version",
                "10.0.0",
                "--dest-dir",
                "/tmp/out",
            ]
        )


def test_nightly_requires_sub_folder():
    with pytest.raises(SystemExit):
        _args("--stream", "nightly", "--repo-base-url", NIGHTLY_BASE)


def test_nightly_with_sub_folder_ok():
    args = _args(
        "--stream",
        "nightly",
        "--repo-base-url",
        NIGHTLY_BASE,
        "--repo-sub-folder",
        NIGHTLY_SUB,
    )
    assert args.repo_sub_folder == NIGHTLY_SUB


def test_sub_folder_rejected_for_non_nightly():
    with pytest.raises(SystemExit):
        _args("--repo-sub-folder", NIGHTLY_SUB)  # default stream is stable (flat)


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad_version",
    [
        '10.0.0"; rm -rf /; echo "',  # shell-injection payload
        "10.0.0 evil",  # whitespace
        "10.0.0$(id)",  # command substitution
        "10.0.0\ngpgcheck=0",  # embedded newline
        "10.0.0\n",  # trailing newline (regex must use \\Z, not $)
        "",  # empty
        "-10.0.0",  # must start with a digit
        # An rpm Version cannot contain a hyphen: rpm reads it as the boundary
        # between version and release. rpm_version_release() passes this value
        # straight into the spec's Version: field, so accepting one here would
        # produce a spec rpmbuild rejects. "~rc1" is the form to use instead.
        "10.0.0-rc1",
    ],
)
def test_rocm_version_rejects_bad_format(bad_version):
    with pytest.raises(SystemExit):
        _args("--rocm-version", bad_version)


@pytest.mark.parametrize("good_version", ["10.0.0", "10.0.0~rc3", "8.0.1rc1", "7.0"])
def test_rocm_version_accepts_valid(good_version):
    assert _args("--rocm-version", good_version).rocm_version == good_version


@pytest.mark.parametrize(
    "bad_url",
    [
        f"{STABLE_BASE}\ngpgcheck=0",  # newline directive injection
        f"{STABLE_BASE} extra",  # whitespace
        "not a url",  # no scheme/netloc
        "ftp://example.com/repo",  # unsupported scheme
    ],
)
def test_repo_base_url_rejects_bad(bad_url):
    with pytest.raises(SystemExit):
        _args("--repo-base-url", bad_url)


def test_repo_base_url_accepts_https():
    assert _args("--repo-base-url", STABLE_BASE).repo_base_url == STABLE_BASE


def test_repo_sub_folder_rejects_bad_format():
    with pytest.raises(SystemExit):
        _args(
            "--stream",
            "nightly",
            "--repo-base-url",
            NIGHTLY_BASE,
            "--repo-sub-folder",
            "20260722-99\nX",  # newline in the dated sub-folder
        )


@pytest.mark.parametrize("sub", ["20260722-99-1", "20260722-run-42"])
def test_repo_sub_folder_rejects_a_second_hyphen(sub):
    # rpm_version_release splits on the first "-" and the remainder becomes the
    # rpm Release, which cannot contain a "-" either: rpmbuild rejects the spec
    # with "Illegal char '-' (0x2d) in: Release". The deb path replaces every
    # "-" and would still build, so without this guard one input yields a
    # package on one format and a build failure on the other.
    with pytest.raises(SystemExit):
        _args(
            "--stream",
            "nightly",
            "--repo-base-url",
            NIGHTLY_BASE,
            "--repo-sub-folder",
            sub,
        )


def test_rpm_release_never_contains_a_hyphen():
    # The invariant the sub-folder pattern exists to protect: anything that
    # passes validation must produce a Release rpmbuild will accept.
    _, release = brp.rpm_version_release("nightly", "10.0.0", NIGHTLY_SUB)
    assert "-" not in release


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


def _rpm_repo(stream: str, base: str, sub: str = "", profile: str = "rhel10") -> str:
    extra = [
        "--os-profile",
        profile,
        "--stream",
        stream,
        "--repo-base-url",
        base,
    ]
    if sub:
        extra += ["--repo-sub-folder", sub]
    args = _args(*extra)
    return _render(
        "template/repo/rpm/amdrocm.repo.j2",
        brp.build_context(args, brp.OS_PROFILES[profile]),
    )


def test_rpm_repo_signed_has_gpgcheck_and_gpgkey():
    out = _rpm_repo("stable", STABLE_BASE)
    assert "[amdrocm-stable]" in out
    assert "enabled=1" in out
    assert "gpgcheck=1" in out
    assert "gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm" in out
    assert f"baseurl={STABLE_EXPECTED['rhel10']}" in out


def test_rpm_repo_nightly_unsigned_no_gpgkey():
    out = _rpm_repo("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "gpgcheck=0" in out
    assert "gpgkey" not in out


def test_rpm_repo_ends_with_newline():
    assert _rpm_repo("stable", STABLE_BASE).endswith("\n")


# --- deb .sources rendering ---------------------------------------------------


def _deb_sources(stream: str, base: str, sub: str = "") -> str:
    extra = [
        "--os-profile",
        "ubuntu2404",
        "--stream",
        stream,
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
    out = _deb_sources("stable", STABLE_BASE)
    assert "X-Repo-Id: amdrocm-stable" in out
    assert "Types: deb" in out
    assert f"URIs: {STABLE_EXPECTED['ubuntu2404']}" in out
    # The suite is rendered from the constant the reachability check also uses,
    # so the repo file and the verified index path cannot drift apart.
    assert f"Suites: {brp.DEB_SUITE}" in out
    assert "Suites: stable" in out
    assert "Components: main" in out
    assert "Architectures: amd64" in out
    assert f"Signed-By: {brp.DEB_KEYRING_PATH}" in out
    # Pinned literally as well: the keyring must not be named rocm.gpg, which
    # the amdgpu driver setup from repo.radeon.com already owns.
    assert "Signed-By: /usr/share/keyrings/amdrocm.gpg" in out
    assert "Trusted:" not in out


def test_deb_sources_nightly_trusted_not_signed():
    out = _deb_sources("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "Trusted: yes" in out
    assert "Signed-By:" not in out


# --- rpm spec rendering -------------------------------------------------------


def _spec(stream: str, base: str, sub: str = "") -> str:
    extra = [
        "--os-profile",
        "rhel10",
        "--stream",
        stream,
        "--repo-base-url",
        base,
    ]
    if sub:
        extra += ["--repo-sub-folder", sub]
    args = _args(*extra)
    ctx = brp.build_context(args, brp.OS_PROFILES["rhel10"])
    version, release = brp.rpm_version_release(stream, args.rocm_version, sub)
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
    out = _spec("stable", STABLE_BASE)
    assert "Name: amdrocm-repo" in out
    assert "Version: 10.0.0" in out
    assert "/etc/yum.repos.d/amdrocm-stable.repo" in out


def test_spec_signed_installs_key_nightly_does_not():
    signed = _spec("stable", STABLE_BASE)
    assert "/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm" in signed
    unsigned = _spec("nightly", NIGHTLY_BASE, NIGHTLY_SUB)
    assert "RPM-GPG-KEY-amdrocm" not in unsigned


@pytest.mark.parametrize(
    "stream,sub",
    [("stable", ""), ("nightly", NIGHTLY_SUB)],
)
def test_spec_conflicts_with_the_legacy_installer_on_every_stream(stream, sub):
    # amdgpu-install ships its own enabled ROCm repository, so the two packages
    # must not be co-installed. Their repo files are named differently from
    # ours, so nothing stops rpm accepting both -- this declaration is the only
    # thing that makes the overlap visible. Asserted on every stream
    # because that package configures a ROCm repository whichever stream we
    # point at.
    base = NIGHTLY_BASE if stream == "nightly" else STABLE_BASE
    assert f"Conflicts: {brp.LEGACY_INSTALLER_PACKAGE}" in _spec(stream, base, sub)


@pytest.mark.parametrize(
    "stream,sub",
    [("stable", ""), ("nightly", NIGHTLY_SUB)],
)
def test_spec_has_no_install_scriptlet(stream, sub):
    # The package ships the key as a file and points gpgkey= at it; it must not
    # try to import it into the rpm keyring from a scriptlet.
    #
    # This looks like an easy improvement -- zypper otherwise stops to ask
    # before its first refresh, and --non-interactive answers "reject", so the
    # repository is skipped and nothing installs. It does not work: rpm holds
    # the database lock for the whole transaction, so "rpm --import" fails with
    # "can't create transaction lock" from %post and from %posttrans alike.
    # Both were tried in ubi10 and bci-base:16.0 containers, and the failure is
    # only a scriptlet warning, so the package still installs looking healthy.
    #
    # The supported answer is the package manager's own flag -- "dnf -y" or
    # "zypper --gpg-auto-import-keys" -- which is what the published ROCm
    # install instructions use. See docs/packaging/rocm_repo_setup.md.
    base = NIGHTLY_BASE if stream == "nightly" else STABLE_BASE
    out = _spec(stream, base, sub)
    assert "%post" not in out
    assert "rpm --import" not in out


def test_spec_files_sets_root_ownership_before_listing_anything():
    # %defattr only governs the entries that follow it, so position is the point
    # and a bare "is it present" assertion would not catch a misplaced one.
    # Without it, files packaged by a non-root rpmbuild keep the builder's
    # uid/gid.
    out = _spec("stable", STABLE_BASE)
    body = out.split("%files", 1)[1].split("%changelog", 1)[0]
    entries = [ln.strip() for ln in body.splitlines() if ln.strip()]
    assert entries[0] == "%defattr(-, root, root, -)"
    assert any(ln.startswith("%config(noreplace)") for ln in entries[1:])


# --- deb control / install rendering ------------------------------------------


def test_control_has_no_key_runtime_deps():
    args = _args()
    ctx = brp.build_context(args, brp.OS_PROFILES["ubuntu2404"])
    out = _render("template/repo/deb/control.j2", ctx)
    assert "Package: amdrocm-repo" in out
    assert "gnupg" not in out
    assert "wget" not in out


@pytest.mark.parametrize("stream", sorted(brp.STREAMS))
def test_control_conflicts_with_the_legacy_installer_on_every_stream(stream):
    # Conflicts only, not Conflicts + Breaks. Conflicts is the stronger field --
    # it blocks unpacking, where Breaks only blocks configuration -- and adding
    # Breaks alongside it was measured to change nothing: dpkg and apt behave
    # identically with either field alone or both. Breaks would also imply some
    # version of amdgpu-install exists that is not affected, which is not the
    # case. Declared for every stream, since that package configures a
    # ROCm repository regardless of which stream this one points at.
    extra = ["--stream", stream]
    if stream == "nightly":
        extra += ["--repo-base-url", NIGHTLY_BASE, "--repo-sub-folder", NIGHTLY_SUB]
    ctx = brp.build_context(_args(*extra), brp.OS_PROFILES["ubuntu2404"])
    out = _render("template/repo/deb/control.j2", ctx)
    assert f"Conflicts: {brp.LEGACY_INSTALLER_PACKAGE}" in out
    assert "Breaks:" not in out
    # The field belongs to the binary package, not the source stanza; a
    # Conflicts above the "Package:" line would silently do nothing.
    binary_stanza = out.split("Package: ", 1)[1]
    assert "Conflicts:" in binary_stanza


def test_postinst_warns_only_when_the_legacy_installer_is_unpurged():
    # apt resolves the Conflicts by removing amdgpu-install, whose repo file
    # and priority pin are conffiles and survive. The probe keys on that exact
    # dpkg state rather than scanning apt's configuration: a scan cannot tell
    # a leftover ROCm repository from the AMDGPU driver repository the same
    # vendor ships, and would tell the user to remove one they still need.
    out = _render(
        "template/repo/deb/postinst.j2",
        brp.build_context(_args(), brp.OS_PROFILES["ubuntu2404"]),
    )
    assert f"dpkg-query -W -f='${{Status}}' {brp.LEGACY_INSTALLER_PACKAGE}" in out
    assert "deinstall ok config-files" in out
    assert f"apt purge {brp.LEGACY_INSTALLER_PACKAGE}" in out
    # The probe must not name individual apt configuration files, which is
    # what made the previous wording point at the driver repository.
    assert "/etc/apt/sources.list.d" not in out
    assert "/etc/apt/preferences.d" not in out


def test_postinst_cannot_fail_the_install():
    # postinst runs under "set -e" and before #DEBHELPER#, so any command that
    # can report non-zero would abort the package install. dpkg-query exits
    # non-zero for an unknown package, which is the ordinary case on a system
    # that never had the legacy installer, so it must be guarded.
    out = _render(
        "template/repo/deb/postinst.j2",
        brp.build_context(_args(), brp.OS_PROFILES["ubuntu2404"]),
    )
    assert "set -e" in out
    assert "dpkg-query" in out, "expected the legacy-installer probe"
    # The guard may sit on a continuation line, so check the whole command.
    probe = out.split("legacy_state=$(", 1)[1].split(")", 1)[0]
    assert "|| true" in probe


def test_rules_forces_root_ownership_in_the_archive():
    # Building as a non-root uid already yields root/root today, because
    # dh_builddeb passes --root-owner-group itself when a package sets no
    # Rules-Requires-Root. The override states it anyway, so the archive does
    # not start carrying the build account's uid/gid if that field is ever
    # added, and to match template/debian_rules.j2.
    out = _render(
        "template/repo/deb/rules.j2",
        brp.build_context(_args(), brp.OS_PROFILES["ubuntu2404"]),
    )
    assert "override_dh_builddeb:" in out
    assert "--root-owner-group" in out
    # The flag has to reach dpkg-deb itself, past the "--" separator, or
    # dh_builddeb treats it as one of its own options and fails.
    assert "dh_builddeb -- -Zxz --root-owner-group" in out


def test_install_maps_keyring_only_when_signed():
    signed = brp.build_context(_args(), brp.OS_PROFILES["ubuntu2404"])
    out_signed = _render("template/repo/deb/install.j2", signed)
    assert "amdrocm-stable.sources /etc/apt/sources.list.d/" in out_signed
    assert "amdrocm.gpg /usr/share/keyrings/" in out_signed
    # The mapped destination must be exactly where Signed-By points, or apt
    # silently treats the repository as unsigned. PurePosixPath, not Path: on
    # Windows the local flavour would give both sides "\usr\share\keyrings" and
    # this would agree with a rendered value that apt could never use.
    keyring = PurePosixPath(brp.DEB_KEYRING_PATH)
    assert f"{keyring.name} {keyring.parent}/" in out_signed

    args_n = _args(
        "--stream",
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
    # Assert the whole mapping line is absent. Substring-matching "rocm.gpg"
    # alone would be satisfied by "amdrocm.gpg" and prove nothing.
    assert "amdrocm.gpg" not in out_nightly
    assert "/usr/share/keyrings/" not in out_nightly


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
            "--stream",
            "stable",
            "--repo-base-url",
            STABLE_BASE,
            "--gpg-key-url",
            STABLE_KEY_URL,
            "--rocm-version",
            "10.0.0",
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


# --- installed-file naming ----------------------------------------------------
#
# The repo file's stem and its rpm section id both come from repo_id(), so the
# stream name reaches disk through a single value. Two streams' packages can be
# installed in turn, and if they wrote the same filename the second would
# silently replace the first's configuration.


@pytest.mark.parametrize("stream", sorted(brp.STREAMS))
def test_installed_filenames_carry_the_stream(stream):
    assert brp.repo_id(stream) == f"amdrocm-{stream}"


def test_no_two_streams_share_an_installed_filename():
    ids = {brp.repo_id(s) for s in brp.STREAMS}
    assert len(ids) == len(brp.STREAMS)


@pytest.mark.parametrize("stream,sub", [("stable", ""), ("nightly", NIGHTLY_SUB)])
def test_rpm_section_id_matches_the_repo_file_stem(stream, sub):
    # Assert the whole header, not a substring: "[amdrocm-stable]" contains
    # "amdrocm", so a containment check would pass on the unscoped id and prove
    # nothing. Same trap as the amdrocm.gpg / rocm.gpg keyring rename.
    base = NIGHTLY_BASE if stream == "nightly" else STABLE_BASE
    out = _rpm_repo(stream, base, sub)
    assert f"[{brp.repo_id(stream)}]" in out
    assert "[amdrocm]\n" not in out


def test_deb_install_maps_the_stream_scoped_sources_file():
    # debian/install maps the staged file to /etc/apt/sources.list.d, so the
    # stem here is the name that lands on the user's disk.
    args = _args()
    ctx = brp.build_context(args, brp.OS_PROFILES["ubuntu2404"])
    out = _render("template/repo/deb/install.j2", ctx)
    assert out.startswith("amdrocm-stable.sources /etc/apt/sources.list.d/")


# --- parity with the published install instructions ---------------------------
#
# These stanzas are copied verbatim from the ROCm 10.0.0 install scripts that
# rocm-install-utils generates and ships (commit 2c40b326f, 2026-08-25), which
# are what users are told to run. They are the closest thing to an external
# oracle available here: a golden file we wrote ourselves only proves the
# renderer is self-consistent, whereas these prove it agrees with the
# configuration AMD actually publishes for the same repositories.
#
# Vendored rather than read from the sibling repository: rocm-install-utils is
# not a dependency of TheRock and must not become one. Refresh them by hand when
# that project changes shape, which is rare -- and a mismatch here is the signal
# that it did.

SCRIPTGEN_DEB_UBUNTU2404 = """\
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
"""

# The rpm stanzas differ only by the distro slug, so one template covers all
# three rpm profiles we build. scriptgen ships RHEL 8.x/10.x and SLES 16.0,
# which map onto rhel8/rhel10/sles16 exactly.
SCRIPTGEN_RPM = """\
[amdrocm-stable]
name=ROCm 10.0.0
baseurl=https://stable.repo.amd.com/rocm/core/packages/{slug}/x86_64
enabled=1
gpgcheck=1
gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
"""

# Fields where a package must diverge from an inline setup script, each with the
# reason. Anything NOT listed here has to match, and a new entry is a decision
# that belongs in the PR description rather than a quiet edit to this table.
#
#   Signed-By   scriptgen downloads the key itself to /etc/apt/keyrings; the
#               package owns its keyring, and the amdrocm name (not rocm) is
#               what keeps dpkg from refusing to unpack it alongside the amdgpu
#               driver setup, which already owns .../keyrings/rocm.gpg.
#   gpgkey      the package ships the key, so it references it locally with
#               file:// rather than refetching it over the network at install.
#   baseurl     trailing slash; both forms resolve, cosmetic only.
#   name        scriptgen bakes the version ("ROCm 10.0.0") because it generates
#               a script per release. This package configures a rolling repo
#               that serves every retained version, so a version in the display
#               name would be wrong the moment the next one ships.
DEVIATIONS = {"Signed-By", "gpgkey", "baseurl", "name"}


def _fields(text: str, sep: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        if sep in line and not line.startswith("["):
            k, _, v = line.partition(sep)
            out[k.strip()] = v.strip()
    return out


def test_deb_sources_match_the_published_install_instructions():
    ours = _fields(_deb_sources("stable", STABLE_BASE), ":")
    theirs = _fields(SCRIPTGEN_DEB_UBUNTU2404, ":")
    assert set(ours) == set(theirs), "field set diverged from the published config"
    for key in theirs:
        if key in DEVIATIONS:
            continue
        assert ours[key] == theirs[key], f"{key} diverged from the published config"


def test_deb_deviations_are_the_expected_ones():
    # Assert the deviations too, so a silent change to one shows up here rather
    # than being waved through by the skip list above.
    ours = _fields(_deb_sources("stable", STABLE_BASE), ":")
    assert ours["Signed-By"] == "/usr/share/keyrings/amdrocm.gpg"


def test_rpm_name_is_not_version_pinned():
    # The repo serves every retained version, so its display name must not name
    # one. Guards the "name" deviation above against being quietly reverted to
    # match scriptgen.
    ours = _fields(_rpm_repo("stable", STABLE_BASE), "=")
    assert ours["name"] == brp.REPO_NAME
    assert "10.0.0" not in ours["name"]


@pytest.mark.parametrize("profile", ["rhel8", "rhel10", "sles16"])
def test_rpm_repo_matches_the_published_install_instructions(profile):
    ours = _fields(_rpm_repo("stable", STABLE_BASE, profile=profile), "=")
    theirs = _fields(SCRIPTGEN_RPM.format(slug=profile), "=")
    assert set(ours) == set(theirs), "field set diverged from the published config"
    for key in theirs:
        if key in DEVIATIONS:
            continue
        assert ours[key] == theirs[key], f"{key} diverged from the published config"
    # baseurl differs only by the trailing slash.
    assert ours["baseurl"].rstrip("/") == theirs["baseurl"].rstrip("/")
    # gpgkey is local because the package ships the key.
    assert ours["gpgkey"] == f"file://{brp.RPM_GPG_KEY_PATH}"


@pytest.mark.parametrize(
    "profile,expected_dir",
    [
        ("rhel8", "/etc/yum.repos.d"),
        ("rhel10", "/etc/yum.repos.d"),
        ("sles16", "/etc/zypp/repos.d"),
    ],
)
def test_rpm_repo_dir_matches_the_published_install_instructions(profile, expected_dir):
    # scriptgen writes SLES to /etc/zypp/repos.d and the RHEL family to
    # /etc/yum.repos.d. Easy to get wrong and invisible until install time.
    assert brp.OS_PROFILES[profile]["rpm_repo_dir"] == expected_dir
