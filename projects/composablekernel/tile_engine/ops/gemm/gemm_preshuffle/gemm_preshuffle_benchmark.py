# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import json
import subprocess
import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class GemmPreshuffleBenchmark:
    def __init__(self, build_dir: str, verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results = []

    def discover_kernels(self) -> List[Path]:
        """Find all benchmark_gemm_preshuffle* executables in the build directory"""
        bin_dir = self.build_dir / "bin"
        if not bin_dir.exists():
            print(f"Error: Binary directory {bin_dir} does not exist")
            return []

        kernels = list(bin_dir.glob("benchmark_gemm_preshuffle*"))
        if self.verbose:
            print(f"Found {len(kernels)} kernel executables")
            for k in kernels:
                print(f"  - {k.name}")
        return kernels

    def extract_kernel_info(self, kernel_path: Path) -> Dict[str, str]:
        """Extract comprehensive kernel information from filename"""
        name = kernel_path.stem

        # Initialize with basic info
        info = {
            "executable": str(kernel_path),
            "name": name,
            "data_type": "unknown",
            "layout": "unknown",
            "pipeline": "unknown",
            "scheduler": "unknown",
            "epilogue": "unknown",
        }

        # Parse the kernel name pattern:
        # benchmark_gemm_preshuffle_fp16_rcr_mem_default_intrawave_False_False_False_False_False_256x256x32_2x2x1_4x64x16
        parts = name.split("_")

        if len(parts) >= 4:
            # Extract data type (4rd part after benchmark_gemm_preshuffle_)
            info["data_type"] = parts[3] if len(parts) > 2 else "unknown"

            # Extract layout (5th part)
            info["layout"] = parts[4] if len(parts) > 3 else "unknown"

            # Extract pipeline (6th part)
            info["pipeline"] = parts[5] if len(parts) > 4 else "unknown"

            # Extract epilogue (7th part)
            info["epilogue"] = parts[6] if len(parts) > 5 else "unknown"

            # Extract scheduler (8th part)
            info["scheduler"] = parts[7] if len(parts) > 6 else "unknown"

        # Extract detailed configuration from the end of the name
        config_info = self.parse_detailed_config(name)
        info.update(config_info)

        # Generate config ID
        info["config_id"] = self.generate_config_id(info)

        return info

    def parse_detailed_config(self, kernel_name: str) -> Dict:
        """Parse detailed configuration from kernel name"""
        config = {
            "tile_sizes": {"tile_m": 0, "tile_n": 0, "tile_k": 0},
            "warp_config": {"warp_m": 0, "warp_n": 0, "warp_k": 0},
            "warp_tile": {"warp_tile_m": 0, "warp_tile_n": 0, "warp_tile_k": 0},
            "optimization_flags": {
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
            },
        }

        # Split by underscore and look for patterns
        parts = kernel_name.split("_")

        # Look for boolean flags (sequence of True/False values)
        bool_sequence = []
        for i, part in enumerate(parts):
            if part in ["True", "False"]:
                bool_sequence.append(part == "True")
                # Continue collecting consecutive boolean values
                j = i + 1
                while j < len(parts) and parts[j] in ["True", "False"]:
                    bool_sequence.append(parts[j] == "True")
                    j += 1
                break

        # Assign boolean flags if we found them
        # Order: pad_m, pad_n, pad_k, persistent (4 flags total)
        if len(bool_sequence) >= 4:
            config["optimization_flags"]["pad_m"] = bool_sequence[0]
            config["optimization_flags"]["pad_n"] = bool_sequence[1]
            config["optimization_flags"]["pad_k"] = bool_sequence[2]
            config["optimization_flags"]["persistent"] = bool_sequence[3]

        # Look for tile size patterns (e.g., 256x256x32_2x2x1_4x64x16)
        # The pattern is: tile_sizes_warp_config_warp_tile
        dimension_groups = []
        for part in parts:
            if "x" in part and len(part.split("x")) == 3:
                try:
                    dims = [int(x) for x in part.split("x")]
                    if all(d > 0 for d in dims):
                        dimension_groups.append(dims)
                except ValueError:
                    continue

        # Assign dimensions based on order and magnitude
        if len(dimension_groups) >= 3:
            # Sort by magnitude to identify: largest=tile_sizes, smallest=warp_config, middle=warp_tile
            sorted_groups = sorted(dimension_groups, key=lambda x: max(x), reverse=True)

            # Largest dimensions = tile sizes
            config["tile_sizes"]["tile_m"] = sorted_groups[0][0]
            config["tile_sizes"]["tile_n"] = sorted_groups[0][1]
            config["tile_sizes"]["tile_k"] = sorted_groups[0][2]

            # Smallest dimensions = warp config
            config["warp_config"]["warp_m"] = sorted_groups[2][0]
            config["warp_config"]["warp_n"] = sorted_groups[2][1]
            config["warp_config"]["warp_k"] = sorted_groups[2][2]

            # Middle dimensions = warp tile
            config["warp_tile"]["warp_tile_m"] = sorted_groups[1][0]
            config["warp_tile"]["warp_tile_n"] = sorted_groups[1][1]
            config["warp_tile"]["warp_tile_k"] = sorted_groups[1][2]
        elif len(dimension_groups) == 2:
            # If only 2 groups, assign based on magnitude
            sorted_groups = sorted(dimension_groups, key=lambda x: max(x), reverse=True)

            # Larger = tile sizes
            config["tile_sizes"]["tile_m"] = sorted_groups[0][0]
            config["tile_sizes"]["tile_n"] = sorted_groups[0][1]
            config["tile_sizes"]["tile_k"] = sorted_groups[0][2]

            # Smaller = warp config
            config["warp_config"]["warp_m"] = sorted_groups[1][0]
            config["warp_config"]["warp_n"] = sorted_groups[1][1]
            config["warp_config"]["warp_k"] = sorted_groups[1][2]
        elif len(dimension_groups) == 1:
            # Only one group - assume it's tile sizes
            config["tile_sizes"]["tile_m"] = dimension_groups[0][0]
            config["tile_sizes"]["tile_n"] = dimension_groups[0][1]
            config["tile_sizes"]["tile_k"] = dimension_groups[0][2]

        return config

    def generate_config_id(self, info: Dict) -> str:
        """Generate a compact config ID from kernel info"""
        # Create a compact identifier
        parts = [
            info.get("data_type", "unk"),
            info.get("layout", "unk"),
            info.get("pipeline", "unk"),
            info.get("scheduler", "unk"),
        ]

        # Add tile configuration if available
        tile_sizes = info.get("tile_sizes", {})
        if tile_sizes.get("tile_m", 0) > 0:
            tile_str = (
                f"{tile_sizes['tile_m']}x{tile_sizes['tile_n']}x{tile_sizes['tile_k']}"
            )
            parts.append(tile_str)

        # Add warp config if available
        warp_config = info.get("warp_config", {})
        if warp_config.get("warp_m", 0) > 0:
            warp_str = f"w{warp_config['warp_m']}x{warp_config['warp_n']}x{warp_config['warp_k']}"
            parts.append(warp_str)

        # Add warp tile if available
        warp_tile = info.get("warp_tile", {})
        if warp_tile.get("warp_tile_m", 0) > 0:
            warp_tile_str = f"wt{warp_tile['warp_tile_m']}x{warp_tile['warp_tile_n']}x{warp_tile['warp_tile_k']}"
            parts.append(warp_tile_str)

        return "_".join(parts)

    def run_kernel(self, kernel_path: Path, params: Dict[str, str]) -> Optional[Dict]:
        """Run a single kernel with given parameters and save output to individual JSON file"""
        # Create results directory
        results_dir = self.build_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Generate unique JSON filename for this kernel
        json_file = results_dir / f"{kernel_path.stem}.json"

        cmd = [str(kernel_path)]

        # Add parameters
        for key, value in params.items():
            cmd.append(f"-{key}={value}")

        # Add JSON output flag for clean JSON output
        cmd.append("-json_output=true")

        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Error running {kernel_path.name}: {result.stderr}")
                return None

            # Save raw output to individual JSON file
            output = result.stdout.strip()

            if output:
                with open(json_file, "w") as f:
                    f.write(output)

                # Parse the JSON file
                return self.parse_json_file(json_file)
            else:
                print(f"No output from {kernel_path.name}")
                return None

        except subprocess.TimeoutExpired:
            print(f"Timeout running {kernel_path.name}")
            return None
        except Exception as e:
            print(f"Error running {kernel_path.name}: {e}")
            return None

    def parse_json_file(self, json_file: Path) -> Optional[Dict]:
        """Parse JSON data from individual kernel output file"""
        try:
            with open(json_file, "r") as f:
                content = f.read().strip()

            # Parse the JSON directly since executables produce clean JSON
            data = json.loads(content)

            # Return the complete JSON data as-is, just add some convenience fields
            result = data.copy()
            if "perf_result" in data:
                perf = data["perf_result"]
                # Add convenience fields for backward compatibility
                result["time_ms"] = perf.get("latency(ms)", 0)
                result["tflops"] = perf.get("tflops(TFlops)", 0)
                result["bandwidth_gb_s"] = perf.get("bandwidth(GB/s)", 0)

            return result

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse JSON from {json_file}: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"Error reading JSON file {json_file}: {e}")
            return None

    def benchmark_problem_size(
        self,
        kernels: List[Path],
        m: int,
        n: int,
        k: int,
        split_k: int = 1,
        verify: int = 0,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> List[Dict]:
        """Benchmark all kernels for a specific problem size"""
        results = []

        params = {
            "m": m,
            "n": n,
            "k": k,
            "split_k": split_k,
            "verify": verify,
            "warmup": warmup,
            "repeat": repeat,
            "flush_cache": str(flush_cache).lower(),
            "rotating_count": rotating_count,
        }

        print(f"\nBenchmarking M={m}, N={n}, K={k}, split_k={split_k}")

        for kernel_path in kernels:
            kernel_info = self.extract_kernel_info(kernel_path)
            result = self.run_kernel(kernel_path, params)

            if result:
                # Create new structured result format
                structured_result = {
                    "name": kernel_info["name"],  # Add name field for compatibility
                    "config_id": kernel_info["config_id"],
                    "problem": result.get("problem", {}),
                    "perf_result": result.get("perf_result", {}),
                    "config": {
                        "data_type": kernel_info["data_type"],
                        "layout": kernel_info["layout"],
                        "pipeline": kernel_info["pipeline"],
                        "scheduler": kernel_info["scheduler"],
                        "epilogue": kernel_info["epilogue"],
                        "tile_sizes": kernel_info.get("tile_sizes", {}),
                        "warp_config": kernel_info.get("warp_config", {}),
                        "warp_tile": kernel_info.get("warp_tile", {}),
                        "optimization_flags": kernel_info.get("optimization_flags", {}),
                    },
                    "executable": kernel_info["executable"],
                    # Keep backward compatibility fields
                    "time_ms": result.get("time_ms", 0),
                    "tflops": result.get("tflops", 0),
                    "bandwidth_gb_s": result.get("bandwidth_gb_s", 0),
                }

                results.append(structured_result)

                if self.verbose:
                    print(
                        f"  {kernel_info['config_id']}: {structured_result['tflops']:.2f} TFLOPS, {structured_result['bandwidth_gb_s']:.2f} GB/s, {structured_result['time_ms']:.2f}ms"
                    )

        return results

    def find_best_kernel(
        self, results: List[Dict], metric: str = "tflops"
    ) -> Optional[Dict]:
        """Find the best performing kernel based on metric"""
        if not results:
            return None

        if metric == "tflops":
            return max(results, key=lambda x: x.get("tflops", 0))
        elif metric == "time_ms":
            return min(results, key=lambda x: x.get("time_ms", float("inf")))
        elif metric == "bandwidth_gb_s":
            return max(results, key=lambda x: x.get("bandwidth_gb_s", 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def benchmark_sweep(
        self,
        problem_sizes: List[Tuple[int, int, int]],
        split_k_values: List[int] = [1],
        verify: bool = False,
        warmup: int = 50,
        repeat: int = 100,
        flush_cache: bool = True,
        rotating_count: int = 1000,
    ) -> Dict:
        """Run comprehensive benchmark sweep"""
        kernels = self.discover_kernels()
        if not kernels:
            print("No kernels found!")
            return {}

        all_results = []
        best_kernels = {}

        for m, n, k in problem_sizes:
            for split_k in split_k_values:
                results = self.benchmark_problem_size(
                    kernels,
                    m,
                    n,
                    k,
                    split_k,
                    verify=2 if verify else 0,
                    warmup=warmup,
                    repeat=repeat,
                    flush_cache=flush_cache,
                    rotating_count=rotating_count,
                )

                all_results.extend(results)

                # Find best kernel for this configuration
                best = self.find_best_kernel(results)
                if best:
                    key = f"m{m}_n{n}_k{k}_splitk{split_k}"
                    best_kernels[key] = best
                    print(
                        f"Best for {key}: {best['name']} ({best['tflops']:.2f} TFLOPS, {best['bandwidth_gb_s']:.2f} GB/s, {best['time_ms']:.2f}ms)"
                    )

        self.results = all_results
        return best_kernels

    def export_csv(self, filename: str):
        """Export all results to CSV"""
        if not self.results:
            print("No results to export")
            return

        # Get all unique keys from results
        all_keys = set()
        for result in self.results:
            all_keys.update(result.keys())

        # Sort keys for consistent output
        fieldnames = sorted(all_keys)

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Results exported to {filename}")

    def export_best_kernels(self, best_kernels: Dict, filename: str):
        """Export best kernel selections to file"""
        with open(filename, "w") as f:
            f.write("# Best kernel selections\n")
            f.write(
                "# Format: problem_size -> kernel_name (TFLOPS, bandwidth, latency)\n\n"
            )

            for key, kernel in sorted(best_kernels.items()):
                f.write(
                    f"{key}: {kernel['name']} ({kernel['tflops']:.2f} TFLOPS, {kernel['bandwidth_gb_s']:.2f} GB/s, {kernel['time_ms']:.2f}ms)\n"
                )

        print(f"Best kernels exported to {filename}")

    def export_json(self, filename: str, best_kernels: Dict = None):
        """Export all results and best kernels to JSON with comprehensive metadata"""
        from datetime import datetime

        # Calculate comprehensive summary statistics for all metrics
        successful_results = [r for r in self.results if r.get("tflops", 0) > 0]

        tflops_values = [r.get("tflops", 0) for r in successful_results]
        bandwidth_values = [r.get("bandwidth_gb_s", 0) for r in successful_results]
        latency_values = [
            r.get("time_ms", 0) for r in successful_results if r.get("time_ms", 0) > 0
        ]

        # Performance breakdown by kernel type
        pipeline_stats = {}
        scheduler_stats = {}
        data_type_stats = {}

        for result in successful_results:
            # Get config info from the new structure
            config = result.get("config", {})

            # Pipeline statistics
            pipeline = config.get("pipeline", "unknown")
            if pipeline not in pipeline_stats:
                pipeline_stats[pipeline] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            pipeline_stats[pipeline]["count"] += 1
            pipeline_stats[pipeline]["best_tflops"] = max(
                pipeline_stats[pipeline]["best_tflops"], result.get("tflops", 0)
            )

            # Scheduler statistics
            scheduler = config.get("scheduler", "unknown")
            if scheduler not in scheduler_stats:
                scheduler_stats[scheduler] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            scheduler_stats[scheduler]["count"] += 1
            scheduler_stats[scheduler]["best_tflops"] = max(
                scheduler_stats[scheduler]["best_tflops"], result.get("tflops", 0)
            )

            # Data type statistics
            data_type = config.get("data_type", "unknown")
            if data_type not in data_type_stats:
                data_type_stats[data_type] = {
                    "count": 0,
                    "avg_tflops": 0,
                    "best_tflops": 0,
                }
            data_type_stats[data_type]["count"] += 1
            data_type_stats[data_type]["best_tflops"] = max(
                data_type_stats[data_type]["best_tflops"], result.get("tflops", 0)
            )

        # Calculate averages for breakdown stats
        for stats_dict, field_name in [
            (pipeline_stats, "pipeline"),
            (scheduler_stats, "scheduler"),
            (data_type_stats, "data_type"),
        ]:
            for key in stats_dict:
                relevant_results = [
                    r
                    for r in successful_results
                    if r.get("config", {}).get(field_name, "unknown") == key
                ]
                if relevant_results:
                    stats_dict[key]["avg_tflops"] = sum(
                        r.get("tflops", 0) for r in relevant_results
                    ) / len(relevant_results)

        output_data = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_kernels_tested": len(self.results),
                "unique_kernels": len(
                    set(r.get("name", "unknown") for r in self.results)
                ),
                "successful_runs": len(successful_results),
                "failed_runs": len(self.results) - len(successful_results),
            },
            "performance_summary": {
                "tflops_stats": {
                    "best": max(tflops_values, default=0),
                    "average": sum(tflops_values) / len(tflops_values)
                    if tflops_values
                    else 0,
                    "min": min(tflops_values, default=0),
                    "median": sorted(tflops_values)[len(tflops_values) // 2]
                    if tflops_values
                    else 0,
                },
                "bandwidth_stats": {
                    "best_gb_s": max(bandwidth_values, default=0),
                    "average_gb_s": sum(bandwidth_values) / len(bandwidth_values)
                    if bandwidth_values
                    else 0,
                    "min_gb_s": min(bandwidth_values, default=0),
                    "median_gb_s": sorted(bandwidth_values)[len(bandwidth_values) // 2]
                    if bandwidth_values
                    else 0,
                },
                "latency_stats": {
                    "best_ms": min(latency_values, default=0),
                    "average_ms": sum(latency_values) / len(latency_values)
                    if latency_values
                    else 0,
                    "max_ms": max(latency_values, default=0),
                    "median_ms": sorted(latency_values)[len(latency_values) // 2]
                    if latency_values
                    else 0,
                },
                "kernel_type_breakdown": {
                    "by_pipeline": pipeline_stats,
                    "by_scheduler": scheduler_stats,
                    "by_data_type": data_type_stats,
                },
                "total_problem_configurations": len(best_kernels)
                if best_kernels
                else 0,
            },
            "kernel_results": self.results,
            "best_kernels_by_problem": best_kernels or {},
        }

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"JSON results exported to {filename}")
        print(f"  - Total kernels: {len(self.results)}")
        print(f"  - Successful runs: {len(successful_results)}")
        print(f"  - Best TFLOPS: {max(tflops_values, default=0):.2f}")
        print(f"  - Best bandwidth: {max(bandwidth_values, default=0):.2f} GB/s")
        print(f"  - Best latency: {min(latency_values, default=0):.2f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Preshuffle Kernel Benchmarking Tool"
    )
    parser.add_argument(
        "build_dir", help="Build directory containing kernel executables"
    )
    parser.add_argument(
        "--problem-sizes",
        nargs="+",
        default=["1024,1024,1024", "2048,2048,2048", "4096,4096,4096"],
        help="Problem sizes as M,N,K tuples",
    )
    parser.add_argument(
        "--split-k", nargs="+", type=int, default=[1], help="Split-K values to test"
    )
    parser.add_argument("--verify", action="store_true", help="Enable verification")
    parser.add_argument(
        "--csv",
        default="gemm_preshuffle_benchmark_results.csv",
        help="CSV output filename",
    )
    parser.add_argument(
        "--best", default="best_kernels.txt", help="Best kernels output filename"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=True,
        help="Enable cache flushing (default: True)",
    )
    parser.add_argument(
        "--rotating-count",
        type=int,
        default=1000,
        help="Number of iterations to rotate cache (default: 1000)",
    )
    parser.add_argument("--json", help="JSON output filename (optional)")

    args = parser.parse_args()

    # Parse problem sizes
    problem_sizes = []
    for size_str in args.problem_sizes:
        try:
            m, n, k = map(int, size_str.split(","))
            problem_sizes.append((m, n, k))
        except ValueError:
            print(f"Invalid problem size: {size_str}")
            return 1

    # Create benchmark instance
    benchmark = GemmPreshuffleBenchmark(args.build_dir, verbose=args.verbose)

    # Run benchmark sweep
    print("Starting GEMM Preshuffle kernel benchmark sweep...")
    start_time = time.time()

    best_kernels = benchmark.benchmark_sweep(
        problem_sizes=problem_sizes,
        split_k_values=args.split_k,
        verify=args.verify,
        warmup=args.warmup,
        repeat=args.repeat,
        flush_cache=args.flush_cache,
        rotating_count=args.rotating_count,
    )

    elapsed_time = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed_time:.2f} seconds")

    # Export results
    benchmark.export_csv(args.csv)
    benchmark.export_best_kernels(best_kernels, args.best)

    # Export JSON if requested
    if args.json:
        benchmark.export_json(args.json, best_kernels)

    return 0


if __name__ == "__main__":
    sys.exit(main())
