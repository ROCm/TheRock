from datetime import datetime, timezone
from packaging_utils import *


def get_skipped_pkglist(pkg_list):
    """Determine which packages were skipped during packaging.

    This function reads package.json and builds the full list of packages
    that are eligible for packaging (i.e., not marked with 'Disablepackaging').
    It then compares that list with the list of packages that were actually
    processed and returns the packages that were skipped.

    Parameters: pkg_list: A list of package names that were successfully packaged.

    Returns: A list of package names that were expected to be packaged but were skipped.
    """

    data = read_package_json_file()
    original_pkglist = [
        pkginfo["Package"] for pkginfo in data if not is_packaging_disabled(pkginfo)
    ]
    skipped_pkglist = [item for item in original_pkglist if item not in pkg_list]
    return skipped_pkglist


def write_build_manifest(config: PackageConfig, pkg_list):
    """Write manifest files listing built and skipped packages.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list: List of all packages attempted

    Returns: None
    """
    print_function_name()

    # Write successful packages manifest
    manifest_file = Path(config.dest_dir) / "built_packages.txt"
    skipped_packages = get_skipped_pkglist(pkg_list)
    try:
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(f"# Built Packages Manifest\n")
            f.write(f"# Package Type: {config.pkg_type.upper()}\n")
            f.write(f"# ROCm Version: {config.rocm_version}\n")
            f.write(f"# Graphics Architecture: {config.gfx_arch}\n")
            f.write(
                f"# Build Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            )
            f.write(f"# Total Attempted: {len(pkg_list) + len(skipped_packages)}\n")
            f.write(f"# Successfully Built: {len(pkg_list)}\n")
            f.write(f"# Skipped: {len(skipped_packages)}\n")
            f.write(f"\n")

            for pkg in sorted(pkg_list):
                f.write(f"{pkg}\n")

        print(f"✅ Built packages manifest written to: {manifest_file}")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to write built packages manifest: {e}")

    # Write skipped packages manifest
    if skipped_packages:
        skipped_file = Path(config.dest_dir) / "skipped_packages.txt"
        try:
            with open(skipped_file, "w", encoding="utf-8") as f:
                f.write(f"# Skipped Packages Manifest\n")
                f.write(f"# Package Type: {config.pkg_type.upper()}\n")
                f.write(f"# ROCm Version: {config.rocm_version}\n")
                f.write(f"# Graphics Architecture: {config.gfx_arch}\n")
                f.write(
                    f"# Build Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                )
                f.write(f"# Format: package_name | reason\n")
                f.write(
                    f"# Note: Package names shown are base names from package.json\n"
                )
                f.write(f"\n")
                f.write(f"# Skipped Packages:\n")
                for pkg in sorted(skipped_packages):
                    f.write(f"{pkg}\n")

            print(f"⚠️  Skipped packages manifest written to: {skipped_file}")
        except Exception as e:
            print(f"⚠️  WARNING: Failed to write skipped packages manifest: {e}")


def print_build_status(config: PackageConfig, pkg_list):
    """Print a summary of the build process.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list : List of all packages attempted

    Returns: None
    """
    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)

    skipped_packages = get_skipped_pkglist(pkg_list)
    skipped_count = len(skipped_packages)
    built_count = len(pkg_list)
    total_packages = built_count + skipped_count

    print(f"\nTotal packages attempted: {total_packages}")
    print(f"✅ Successfully built: {built_count}")
    print(f"   (Showing base package names)")
    for pkg in sorted(pkg_list):
        print(f"   - {pkg}")

    if skipped_packages:
        print(f"\n⏭️   Skipped packages ({skipped_count}):")
        print(f"   (Showing base package names from package.json)")
        for pkg in sorted(skipped_packages):
            print(f"   - {pkg}")
        print("\nNote: Skipped packages have been excluded from dependencies")

    print("\n" + "=" * 80)
    print(f"Package type: {config.pkg_type.upper()}")
    print(f"ROCm version: {config.rocm_version}")
    print(f"Output directory: {config.dest_dir}")
    print("=" * 80 + "\n")


def print_build_summary(config: PackageConfig, pkg_list):
    write_build_manifest(config, pkg_list)
    print_build_status(config, pkg_list)
