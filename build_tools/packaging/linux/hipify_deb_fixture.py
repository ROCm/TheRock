#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""TheRock CI functional fixture for actual dh_perl and DEB alias metadata.

No build-performance experiment: controls only File::Find visitation order in
the installed, unchanged helper. Never installs or executes the payload script.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

import deb_package as deb

ORIGINAL = b"#!/usr/bin/env perl\nprint qq(fixture\\n);\n"
PREPARED = b"#! /usr/bin/perl\nprint qq(fixture\\n);\n"
LINK = "../libexec/hipify/hipify-perl"
PREFIX = "opt/rocm/core-10.3"
FIND_MODULE = r"""package TherockFixtureFind;
use File::Find ();
my $original = \&File::Find::find;
no warnings 'redefine';
*File::Find::find = sub {
  my ($wanted, @paths) = @_;
  my %options = ref($wanted) eq 'HASH' ? %$wanted : (wanted => $wanted);
  $options{preprocess} = sub {
    return $ENV{THEROCK_FIXTURE_ORDER} eq 'ascending' ? sort(@_) : reverse(sort(@_));
  };
  return $original->(\%options, @paths);
};
1;
"""


def snapshot(root: Path) -> tuple:
    alias, target = root / "bin/hipify-perl", root / "libexec/hipify/hipify-perl"
    return (
        alias.is_symlink(),
        str(alias.readlink()),
        stat.S_IMODE(alias.lstat().st_mode),
        target.read_bytes(),
        stat.S_IMODE(target.stat().st_mode),
        target.stat().st_mtime_ns,
        alias.lstat().st_mtime_ns,
    )


def run(command: list[str], cwd: Path, env: dict) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Fixture command failed: {command}\n{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}"
        )
    return result


def generate_control(case: Path, tools: dict, env: dict) -> subprocess.CompletedProcess:
    """Match dpkg-buildpackage's fakeroot metadata context without real root."""
    return run([tools["fakeroot"], tools["dh_gencontrol"], "-pfixture"], case, env)


def run_case(base: Path, tools: dict, fixed: bool, order: str) -> dict:
    case = base / f"{fixed}-{order}"
    source = case / "input"
    real = source / "libexec/hipify/hipify-perl"
    link = source / "bin/hipify-perl"
    real.parent.mkdir(parents=True)
    link.parent.mkdir(parents=True)
    real.write_bytes(ORIGINAL)
    real.chmod(0o755)
    link.symlink_to(LINK)
    original = snapshot(source)
    staged = case / "debian/fixture" / PREFIX
    deb.copy_package_contents(source, staged)
    target = staged / "libexec/hipify/hipify-perl"
    target.unlink()
    os.link(real, target)
    assert target.samefile(real)
    if fixed:
        assert deb.prepare_hipify_perl_shebang(staged)
        assert not target.samefile(real)
    (case / "debian/control").write_text(
        "Source: fixture\nSection: devel\nPriority: optional\n"
        "Maintainer: Fixture <fixture@example.invalid>\n\n"
        "Package: fixture\nArchitecture: all\nDepends: ${perl:Depends}\n"
        "Description: HIPIFY alias source fixture\n"
    )
    (case / "debian/compat").write_text("13\n")
    (case / "debian/changelog").write_text(
        "fixture (1.0) unstable; urgency=medium\n\n  * Fixture.\n\n"
        " -- Fixture <fixture@example.invalid>  Sat, 31 Oct 2026 00:00:00 +0000\n"
    )
    env = dict(
        os.environ,
        THEROCK_FIXTURE_ORDER=order,
        DH_VERBOSE="1",
        DH_COMPAT="13",
        SOURCE_DATE_EPOCH="1793404800",
    )
    ordered = dict(env, PERL5OPT=f"-I{base} -MTherockFixtureFind")
    command = [tools["dh_perl"], "-pfixture"]
    helper = run(command, case, ordered)
    substvars = (case / "debian/fixture.substvars").read_text()
    assert "perl:Depends=" in substvars
    alias = staged / "bin/hipify-perl"
    if fixed:
        assert alias.is_symlink(), helper.stdout.decode(errors="replace")
    assert target.read_bytes() == PREPARED
    assert snapshot(source) == original
    control = generate_control(case, tools, env)
    archive = case / "fixture.deb"
    run(
        [
            tools["dpkg-deb"],
            "--root-owner-group",
            "--build",
            "debian/fixture",
            str(archive),
        ],
        case,
        env,
    )
    payload = run([tools["dpkg-deb"], "--fsys-tarfile", str(archive)], case, env).stdout
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        member = tar.getmember(f"./{PREFIX}/bin/hipify-perl")
        assert member.issym() == alias.is_symlink()
        if member.issym():
            assert (member.linkname, member.mode) == (LINK, 0o777)
        else:
            assert (member.mode, tar.extractfile(member).read()) == (0o755, PREPARED)
        actual = tar.getmember(f"./{PREFIX}/libexec/hipify/hipify-perl")
        assert (actual.mode, tar.extractfile(actual).read()) == (0o755, PREPARED)
    depends = run(
        [tools["dpkg-deb"], "--field", str(archive), "Depends"], case, env
    ).stdout.decode()
    assert "perl" in depends
    return dict(
        prepared=fixed,
        traversal=order,
        alias_symlink=alias.is_symlink(),
        input_unchanged=snapshot(source) == original,
        alias_mode=member.mode,
        alias_target=member.linkname,
        real_target_mode=actual.mode,
        target_sha256=hashlib.sha256(PREPARED).hexdigest(),
        depends=depends,
        substvars=substvars,
        helper_argv=command,
        control_argv=control.args,
        helper_stdout=helper.stdout.decode(),
        helper_stderr=helper.stderr.decode(),
    )


def main() -> None:
    if not __debug__:
        raise RuntimeError("Fixture requires enabled Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not (
        sys.platform == "linux"
        and os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("GITHUB_REPOSITORY") == "ROCm/TheRock"
    ):
        raise RuntimeError("Actual Debhelper fixture executes only in ROCm/TheRock CI")
    tools = {
        name: shutil.which(name)
        for name in ["dh_perl", "dh_gencontrol", "dpkg-deb", "dpkg-query", "fakeroot"]
    }
    if not all(tools.values()):
        raise RuntimeError(
            "Fixture requires installed debhelper, dpkg-dev, perl, and fakeroot"
        )
    helper_sha = hashlib.sha256(Path(tools["dh_perl"]).read_bytes()).hexdigest()
    fakeroot_sha = hashlib.sha256(Path(tools["fakeroot"]).read_bytes()).hexdigest()
    versions = subprocess.check_output(
        [
            tools["dpkg-query"],
            "-W",
            "-f=${binary:Package}\t${Version}\n",
            "debhelper",
            "perl",
            "fakeroot",
            "dpkg",
        ],
        text=True,
    )
    records = []
    validated = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "TherockFixtureFind.pm").write_text(FIND_MODULE)
            for fixed in [False, True]:
                for order in ["ascending", "descending"]:
                    records.append(run_case(base, tools, fixed, order))
        assert {r["alias_symlink"] for r in records if not r["prepared"]} == {
            False,
            True,
        }
        assert all(
            r["alias_symlink"] and r["input_unchanged"]
            for r in records
            if r["prepared"]
        )
        assert (
            hashlib.sha256(Path(tools["dh_perl"]).read_bytes()).hexdigest()
            == helper_sha
        )
        assert (
            hashlib.sha256(Path(tools["fakeroot"]).read_bytes()).hexdigest()
            == fakeroot_sha
        )
        validated = True
    finally:
        report = dict(
            validated=validated,
            repository=os.getenv("GITHUB_REPOSITORY"),
            sha=os.getenv("GITHUB_SHA"),
            dh_perl_sha256=helper_sha,
            fakeroot_path=tools["fakeroot"],
            fakeroot_sha256=fakeroot_sha,
            installed_packages=versions,
            visitation_module_sha256=hashlib.sha256(FIND_MODULE.encode()).hexdigest(),
            cases=records,
            note="Unchanged installed helper; only File::Find visitation order is controlled",
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
