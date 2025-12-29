"""Main ROCm SDK CLI for managing the Python installation."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from . import _dist_info as di


def _do_path(args: argparse.Namespace):
    from . import _devel

    try:
        root_path = _devel.get_devel_root()
    except ModuleNotFoundError as e:
        print(
            "ERROR: Could not load the `rocm[devel]` package, which is required "
            "to access runtime tools and development files. Please install it with "
            "your package manager (pip, uv, etc)",
            file=sys.stderr,
        )
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    if args.cmake:
        print(root_path / "lib" / "cmake")
    elif args.bin:
        print(root_path / "bin")
    elif args.root:
        print(root_path)
    else:
        print("ERROR: Expected path type flag", file=sys.stderr)
        sys.exit(1)


def _do_init(args: argparse.Namespace):
    from . import _devel

    try:
        # The function `_devel.get_devel_root()` calls into `_expand_devel_contents`
        # if contents for development were not yet unpacked.
        root_path = _devel.get_devel_root()
    except ModuleNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Devel contents expanded to '{root_path}'")


def _do_test(args: argparse.Namespace):
    import unittest

    # Start with required test modules.
    ALL_TEST_MODULES = [
        "rocm_sdk.tests.base_test",
        "rocm_sdk.tests.core_test",
    ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test modules if they are installed.
    target_family = di.determine_target_family()
    if di.ALL_PACKAGES["libraries"].has_py_package(target_family):
        ALL_TEST_MODULES.append("rocm_sdk.tests.libraries_test")
    else:
        print("NOTE: Skipping libraries tests (not installed for this arch)")

    # The devel platform package may not exist yet since it is populated on-demand,
    # so check that the pure package exists.
    if importlib.util.find_spec("rocm_sdk_devel") is not None:
        ALL_TEST_MODULES.append("rocm_sdk.tests.devel_test")

    # Load all tests.
    for test_module_name in ALL_TEST_MODULES:
        suite.addTests(loader.loadTestsFromName(test_module_name))
    if loader.errors:
        print(f"WARNING: Test discovery had errors: {loader.errors}")

    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=3)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def _do_version(args: argparse.Namespace):
    print(di.__version__)


def _do_targets(args: argparse.Namespace):
    core_mod_name = di.ALL_PACKAGES["core"].get_py_package_name()
    core_mod = importlib.import_module(core_mod_name)
    core_path = Path(core_mod.__file__).parent  # Chop __init__.py
    dist_info_path = core_path / "share" / "therock" / "dist_info.json"
    with open(dist_info_path, "rb") as f:
        dist_info_struct = json.load(f)
    print(dist_info_struct["dist_amdgpu_targets"])


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="rocm-sdk",
        usage="rocm-sdk {command} ...",
        description="ROCm SDK Python CLI",
    )
    sub_p = p.add_subparsers(required=True)

    # 'path' subcommand.
    path_p = sub_p.add_parser("path", help="Print various paths to ROCm installation")
    path_p_group = path_p.add_mutually_exclusive_group(required=True)
    path_p_group.add_argument(
        "--cmake",
        action="store_true",
        help="Print the CMAKE_PATH_PREFIX for the ROCm development package",
    )
    path_p_group.add_argument(
        "--bin",
        action="store_true",
        help="Print the ROCm development package binary directory",
    )
    path_p_group.add_argument(
        "--root",
        action="store_true",
        help="Print the ROCm development package root directory",
    )
    path_p.set_defaults(func=_do_path)

    # 'test' subcommand.
    test_p = sub_p.add_parser("test", help="Run installation tests to verify integrity")
    test_p.set_defaults(func=_do_test)

    # 'version' subcommand.
    version_p = sub_p.add_parser("version", help="Print version information")
    version_p.set_defaults(func=_do_version)

    # 'targets' subcommand.
    targets_p = sub_p.add_parser(
        "targets", help="Print information about the GPU targets that are supported"
    )
    targets_p.set_defaults(func=_do_targets)

    # 'init' subcommand.
    init_p = sub_p.add_parser(
        "init", help="Expand devel contents to initialize rocm[devel]"
    )
    init_p.set_defaults(func=_do_init)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
