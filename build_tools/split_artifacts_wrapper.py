#!/usr/bin/env python3
"""
Wrapper for split_artifacts.py that handles stripped binaries with empty .hip_fatbin sections.

When binaries are stripped, the .hip_fatbin section header may remain but the content
is removed. This causes the kpack splitter to fail when trying to extract device code.

This wrapper monkey-patches the kpack code to gracefully skip such binaries instead
of failing the entire build.
"""

import sys


def patch_kpack():
    """Patch kpack to handle empty .hip_fatbin sections gracefully."""
    from rocm_kpack import binutils, artifact_splitter
    from rocm_kpack.kpack_transform import NotFatBinaryError

    # Store original method
    original_get_bundler_input = binutils.BundledBinary._get_bundler_input

    def patched_get_bundler_input(self):
        """Patched version that raises NotFatBinaryError for empty sections."""
        result = original_get_bundler_input(self)

        # Check if the file was actually created and has content
        if not result.exists():
            raise NotFatBinaryError(
                f"Section .hip_fatbin in {self.file_path} has no extractable content "
                f"(possibly stripped). Skipping."
            )

        if result.stat().st_size == 0:
            raise NotFatBinaryError(
                f"Section .hip_fatbin in {self.file_path} is empty "
                f"(possibly stripped). Skipping."
            )

        return result

    binutils.BundledBinary._get_bundler_input = patched_get_bundler_input

    # Store original process_fat_binaries method
    original_process_fat_binaries = artifact_splitter.ArtifactSplitter.process_fat_binaries

    def patched_process_fat_binaries(self, fat_binaries, prefix, prefix_path):
        """Patched version that catches NotFatBinaryError and continues."""
        from collections import defaultdict

        kernels_by_arch = defaultdict(list)

        for binary_path in fat_binaries:
            if self.verbose:
                print(f"Processing fat binary: {binary_path.relative_to(prefix_path)}")

            try:
                # Create a BundledBinary instance with our toolchain
                binary = binutils.BundledBinary(binary_path, toolchain=self.toolchain)

                # Track code object index per (binary, arch) for multi-TU support
                code_object_index = defaultdict(int)

                # Extract kernels using context manager
                with binary.unbundle() as unbundled:
                    for target_name, file_name in unbundled.target_list:
                        arch = artifact_splitter.extract_architecture_from_target(target_name)
                        if arch:
                            if self.gpu_targets is not None:
                                bare_arch = artifact_splitter.strip_target_features(arch)
                                if bare_arch not in self.gpu_targets:
                                    code_object_index[arch] += 1
                                    if self.verbose:
                                        print(
                                            f"    Skipping kernel for {arch}: "
                                            f"{file_name} (not in gpu_targets)"
                                        )
                                    continue

                            kernel_path = unbundled.dest_dir / file_name
                            kernel_data = kernel_path.read_bytes()

                            base_relpath = binary_path.relative_to(prefix_path).as_posix()
                            index = code_object_index[arch]
                            code_object_index[arch] += 1
                            indexed_relpath = f"{base_relpath}#{index}"

                            extracted_kernel = artifact_splitter.ExtractedKernel(
                                target_name=target_name,
                                kernel_data=kernel_data,
                                source_binary_relpath=indexed_relpath,
                                source_prefix=prefix,
                                architecture=arch,
                            )

                            kernels_by_arch[arch].append(extracted_kernel)
                            if self.verbose:
                                print(
                                    f"    Extracted kernel for {arch}: {file_name} -> "
                                    f"{indexed_relpath} ({len(kernel_data)} bytes)"
                                )

            except NotFatBinaryError as e:
                # Binary has section header but no extractable content - skip it
                print(f"    Skipping {binary_path.name}: {e}")
                continue

        return dict(kernels_by_arch)

    artifact_splitter.ArtifactSplitter.process_fat_binaries = patched_process_fat_binaries


def main():
    # Apply patches before importing the main module
    patch_kpack()

    # Now run the original split_artifacts main
    from rocm_kpack.tools import split_artifacts
    sys.exit(split_artifacts.main())


if __name__ == "__main__":
    main()
