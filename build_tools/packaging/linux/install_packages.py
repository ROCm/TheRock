#!/usr/bin/env python3
import argparse
import sys
from package_load import LoadPackages, logger


def main():
    parser = argparse.ArgumentParser(description="Install ROCm native build packages.")
    parser.add_argument(
        "--dest-dir", required=True, help="Directory where built packages are located."
    )
    parser.add_argument("--package_json", required=True, help="Path to package.json.")
    parser.add_argument(
        "--version",
        choices=["true", "false"],
        default="false",
        help="If true, install only versioned packages.",
    )
    parser.add_argument(
        "--composite",
        choices=["true", "false"],
        default="false",
        help="Install composite packages only.",
    )
    parser.add_argument(
        "--amdgpu_family",
        type=str,
        required=False,
        help="Specify AMD GPU family (e.g., gfx94x).",
    )
    parser.add_argument(
        "--rocm-version", 
        type=str,
        required=True,
        help="Specify ROCm version (e.g., 7.0.0).",
    )
    parser.add_argument(
        "--upload",
        choices=["pre", "post"],
        required=True,
        help="Specify whether this is a pre-upload or post-upload installation.",
    )

    args = parser.parse_args()

    version_flag = args.version.lower() == "true"
    composite_flag = args.composite.lower() == "true"
    amdgpu_family = args.amdgpu_family
    rocm_version = args.rocm_version

    pm = LoadPackages(args.package_json, version_flag, amdgpu_family,rocm_version)
    non_comp, comp = pm.list_composite_packages()


    # Select package list
    if composite_flag:
        logger.info(f"Count of Composite packages: {len(comp)}")
        sorted_packages = pm.sort_packages_by_dependencies(comp)
    else:
        logger.info(f"Count of non Composite packages: {len(non_comp)}")
        sorted_packages = pm.sort_packages_by_dependencies(non_comp)

    logger.info(f"Version flag: {version_flag}")
    logger.info(f"Upload stage: {args.upload}")

    # -----------------------
    # PRE-UPLOAD INSTALLATION
    # -----------------------
    if args.upload == "pre":
        logger.info("=== Starting pre-upload installation ===")
        try:
            pm.install_packages(args.dest_dir, sorted_packages, version_flag,False)
            logger.info("Pre-upload installation completed successfully.")
        except Exception as e:
            logger.error(f"Pre-upload installation failed: {e}")
            sys.exit(1)

    # ------------------------
    # POST-UPLOAD INSTALLATION
    # ------------------------
    elif args.upload == "post":
        logger.info("=== Starting post-upload installation ===")

        # Step 1: Populate repo file after upload
        try:
            pm.populate_repo_file(args.amdgpu_family+"_"+args.dest_dir)
            logger.info("Repository file populated successfully.")
        except AttributeError:
            logger.warning("populate_repo_file() not implemented in LoadPackages. Add it for repo setup.")
        except Exception as e:
            logger.error(f"Failed to populate repo file: {e}")
            sys.exit(1)

        # Step 2: Install from repo
        try:
            pm.install_packages(args.dest_dir, sorted_packages, version_flag,True)
            logger.info("Post-upload installation completed successfully.")
        except Exception as e:
            logger.error(f"Post-upload installation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

