#!/usr/bin/env python3
"""Blender HIP Cycles Rendering Tests for TheRock CI.

This script runs Blender scenes using the Cycles render engine with the HIP
backend and optionally compares rendered output against reference images using
SSIM and MSE metrics.

Prerequisites
-------------
- Blender (official release tarball, >= 4.0)
- ROCm installation with HIP support
- Python packages: opencv-python-headless, scikit-image, numpy

Usage Examples
--------------
Smoke test (render only, no image comparison):
    $ python run_blender_tests.py \\
        --blender-dir /path/to/blender-4.5.6-linux-x64 \\
        --scenes-dir /path/to/scenes

With reference image comparison:
    $ python run_blender_tests.py \\
        --blender-dir /path/to/blender-4.5.6-linux-x64 \\
        --scenes-dir /path/to/scenes \\
        --ref-dir /path/to/reference_images

Environment Variables
---------------------
AMDGPU_FAMILY : str, optional
    Target AMDGPU family for informational logging.
HIP_VISIBLE_DEVICES : str, optional
    Restrict which GPU devices are visible to HIP.
"""

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path


def extract_render_summary(blender_output: str) -> dict[str, str]:
    """Extract key information from Blender's captured output.

    Parses GPU device, render time, kernel compilation time, and
    sample count from Blender's debug output.

    Returns:
        Dict with extracted fields (missing fields omitted).
    """
    summary: dict[str, str] = {}

    # GPU device: 'Added device "AMD Radeon RX 6950 XT"'
    match = re.search(r'Added device "([^"]+)"', blender_output)
    if match:
        summary["gpu"] = match.group(1)

    # Total render time: 'Total render time: 12.9499'
    match = re.search(r"Total render time:\s+([\d.]+)", blender_output)
    if match:
        summary["total_time"] = f"{float(match.group(1)):.2f}s"

    # Path tracing time: 'Path Tracing  4.937481  0.019287'
    match = re.search(r"Path Tracing\s+([\d.]+)\s+([\d.]+)", blender_output)
    if match:
        summary["render_time"] = f"{float(match.group(1)):.2f}s"

    # Kernel compilation: 'Kernel compilation finished in 45.3s.'
    match = re.search(r"Kernel compilation finished in ([\d.]+)s", blender_output)
    if match:
        summary["kernel_compile"] = f"{float(match.group(1)):.2f}s"

    # Samples: 'Rendered 256 samples' (use last match for final count)
    sample_matches = re.findall(r"Rendered (\d+) samples", blender_output)
    if sample_matches:
        summary["samples"] = sample_matches[-1]

    return summary


def print_render_summary(summary: dict[str, str]) -> None:
    """Print a brief render summary to the main log."""
    parts = []
    if "gpu" in summary:
        parts.append(f"GPU: {summary['gpu']}")
    if "samples" in summary:
        parts.append(f"Samples: {summary['samples']}")
    if "render_time" in summary:
        parts.append(f"Render: {summary['render_time']}")
    if "total_time" in summary:
        parts.append(f"Total: {summary['total_time']}")
    if "kernel_compile" in summary:
        parts.append(f"Kernel compile: {summary['kernel_compile']}")
    if parts:
        print(f"  {', '.join(parts)}")


def compare_images(
    rendered_path: Path,
    ref_path: Path,
    ssim_thresh: float,
    mse_thresh: float,
) -> tuple[bool, float, float]:
    """Compare a rendered image against a reference using SSIM and MSE.

    Args:
        rendered_path: Path to the rendered image.
        ref_path: Path to the reference image.
        ssim_thresh: Minimum SSIM score to pass (0-1, higher is more similar).
        mse_thresh: Maximum MSE value to pass (lower is more similar).

    Returns:
        Tuple of (passed, ssim_score, mse_value).
    """
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity

    img = cv2.imread(str(rendered_path))
    ref = cv2.imread(str(ref_path))

    if img is None:
        print(f"  ERROR: Could not read rendered image: {rendered_path}")
        return False, 0.0, float("inf")
    if ref is None:
        print(f"  ERROR: Could not read reference image: {ref_path}")
        return False, 0.0, float("inf")

    if img.shape != ref.shape:
        print(
            f"  WARNING: Image dimensions differ "
            f"(rendered={img.shape}, ref={ref.shape}), resizing reference"
        )
        ref = cv2.resize(ref, (img.shape[1], img.shape[0]))

    # SSIM window size must be odd and <= smallest image dimension
    smaller_side = min(img.shape[:2])
    win_size = min(smaller_side, 7)
    win_size = win_size if win_size % 2 == 1 else win_size - 1
    win_size = max(win_size, 3)

    score, _ = structural_similarity(
        img, ref, full=True, channel_axis=2, win_size=win_size
    )
    mse_val = float(np.mean((img.astype(float) - ref.astype(float)) ** 2))

    passed = score >= ssim_thresh and mse_val <= mse_thresh
    return passed, score, mse_val


def parse_scenes_file(scenes_file: Path) -> list[tuple[str, int]]:
    """Parse a scenes manifest file.

    Format: one scene per line as ``<scene_name_without_ext> <frame_number>``.
    Lines starting with ``#`` and blank lines are skipped.

    Returns:
        List of (scene_name, frame_number) tuples.
    """
    scenes: list[tuple[str, int]] = []
    with open(scenes_file) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(
                    f"  WARNING: Skipping malformed line {line_no} "
                    f"in {scenes_file}: {line!r}"
                )
                continue
            scene_name = parts[0]
            try:
                frame = int(parts[1])
            except ValueError:
                print(
                    f"  WARNING: Invalid frame number on line {line_no} "
                    f"in {scenes_file}: {parts[1]!r}"
                )
                continue
            scenes.append((scene_name, frame))
    return scenes


def render_scene(
    blender_bin: Path,
    scene_file: Path,
    output_dir: Path,
    scene_name: str,
    frame: int,
    timeout_seconds: int,
) -> tuple[int, Path, str]:
    """Render a single Blender scene with Cycles HIP.

    Returns:
        Tuple of (return_code, expected_output_path, captured_output).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / f"{scene_name}_"

    cmd = [
        str(blender_bin),
        "-b",
        str(scene_file),
        "--engine",
        "CYCLES",
        "-F",
        "PNG",
        "-o",
        f"{output_prefix}###",
        "-f",
        str(frame),
        "--debug-cycles",
        "--",
        "--cycles-device",
        "HIP",
    ]

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
        return_code = result.returncode
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        print(f"  ERROR: Blender timed out after {timeout_seconds}s")
        return_code = -1
        output = (e.stdout or "") + (e.stderr or "")

    # Blender names output as <prefix><frame_padded>.png
    expected_output = output_dir / f"{scene_name}_{frame:03d}.png"
    return return_code, expected_output, output


def run_tests(args: argparse.Namespace) -> int:
    """Run all Blender test scenes and report results.

    Returns:
        0 if all tests passed, 1 otherwise.
    """
    blender_bin = args.blender_dir / "blender"
    if not blender_bin.exists():
        print(f"ERROR: Blender binary not found at {blender_bin}")
        return 1

    scenes = parse_scenes_file(args.scenes_file)
    if not scenes:
        print(f"ERROR: No scenes found in {args.scenes_file}")
        return 1

    print(f"Blender binary: {blender_bin}")
    print(f"Scenes directory: {args.scenes_dir}")
    print(f"Reference directory: {args.ref_dir or 'None (smoke test mode)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scenes to render: {len(scenes)}")
    print(f"SSIM threshold: {args.ssim_threshold}")
    print(f"MSE threshold: {args.mse_threshold}")
    print(f"Render timeout: {args.timeout}s")
    print()

    results: list[dict] = []

    for scene_name, frame in scenes:
        print(f"--- Scene: {scene_name}, Frame: {frame} ---")

        scene_file = args.scenes_dir / f"{scene_name}.blend"
        if not scene_file.exists():
            print(f"  ERROR: Scene file not found: {scene_file}")
            results.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "render_rc": -1,
                    "passed": False,
                    "ssim": None,
                    "mse": None,
                    "reason": "scene file not found",
                }
            )
            continue

        render_rc, output_path, blender_output = render_scene(
            blender_bin=blender_bin,
            scene_file=scene_file,
            output_dir=args.output_dir,
            scene_name=scene_name,
            frame=frame,
            timeout_seconds=args.timeout,
        )

        # Save per-scene log for all outcomes (success or failure)
        log_path = args.output_dir / f"{scene_name}_{frame:03d}.log"
        log_path.write_text(blender_output)
        print(f"  Log: {log_path}")

        if render_rc != 0:
            print(f"  Blender exited with code {render_rc}")
            print("  --- Blender output ---")
            print(blender_output)
            print("  --- End of Blender output ---")
            results.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "render_rc": render_rc,
                    "passed": False,
                    "ssim": None,
                    "mse": None,
                    "reason": f"render failed (exit code {render_rc})",
                }
            )
            continue

        if not output_path.exists():
            print(f"  ERROR: Expected output not found: {output_path}")
            results.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "render_rc": render_rc,
                    "passed": False,
                    "ssim": None,
                    "mse": None,
                    "reason": "output file not produced",
                }
            )
            continue

        print(f"  Render succeeded: {output_path}")
        render_summary = extract_render_summary(blender_output)
        print_render_summary(render_summary)

        # Check for reference image
        ref_path = None
        if args.ref_dir:
            ref_path = args.ref_dir / output_path.name
            if not ref_path.exists():
                ref_path = None

        if ref_path:
            print(f"  Comparing against reference: {ref_path}")
            passed, ssim_val, mse_val = compare_images(
                output_path, ref_path, args.ssim_threshold, args.mse_threshold
            )
            print(
                f"  SSIM: {ssim_val:.4f}, MSE: {mse_val:.2f} -> {'PASS' if passed else 'FAIL'}"
            )
            results.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "render_rc": render_rc,
                    "passed": passed,
                    "ssim": ssim_val,
                    "mse": mse_val,
                    "reason": "" if passed else "image comparison failed",
                }
            )
        else:
            print("  No reference image found, smoke test only (render succeeded)")
            results.append(
                {
                    "scene": scene_name,
                    "frame": frame,
                    "render_rc": render_rc,
                    "passed": True,
                    "ssim": None,
                    "mse": None,
                    "reason": "smoke test (no reference)",
                }
            )

        print()

    # Print summary table
    print("=" * 80)
    print("BLENDER HIP TEST SUMMARY")
    print("=" * 80)
    header = (
        f"{'Scene':<30} {'Frame':>5} {'RC':>4} {'SSIM':>8} {'MSE':>10} {'Result':>8}"
    )
    print(header)
    print("-" * 80)

    all_passed = True
    for r in results:
        ssim_str = f"{r['ssim']:.4f}" if r["ssim"] is not None else "N/A"
        mse_str = f"{r['mse']:.2f}" if r["mse"] is not None else "N/A"
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(
            f"{r['scene']:<30} {r['frame']:>5} {r['render_rc']:>4} "
            f"{ssim_str:>8} {mse_str:>10} {status:>8}"
        )
        if r["reason"] and not r["passed"]:
            print(f"  -> {r['reason']}")

    print("-" * 80)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print("=" * 80)

    if all_passed:
        print("\nAll Blender HIP tests passed.")
    else:
        print("\nSome Blender HIP tests FAILED.")

    return 0 if all_passed else 1


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Run Blender Cycles HIP rendering tests.

            Renders scenes listed in a manifest file using Blender's Cycles engine
            with the HIP backend. Optionally compares rendered output against
            reference images using SSIM and MSE metrics.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--blender-dir",
        type=Path,
        required=True,
        help="Path to the Blender installation directory (containing the 'blender' binary)",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        required=True,
        help="Directory containing .blend scene files",
    )
    parser.add_argument(
        "--scenes-file",
        type=Path,
        default=None,
        help=("Path to scenes manifest file. " "Defaults to <scenes-dir>/scenes.txt"),
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing reference images for comparison. "
            "If not provided, runs in smoke test mode (render only)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for rendered output. Defaults to <scenes-dir>/out",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.9,
        help="Minimum SSIM score to pass (default: 0.9)",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=1000.0,
        help="Maximum MSE value to pass (default: 1000.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each Blender render (default: 600)",
    )

    args = parser.parse_args(argv)

    # Apply defaults that depend on other args
    if args.scenes_file is None:
        args.scenes_file = args.scenes_dir / "scenes.txt"
    if args.output_dir is None:
        args.output_dir = args.scenes_dir / "out"

    return args


def main() -> int:
    """Main entry point."""
    args = cmd_arguments(sys.argv[1:])
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
