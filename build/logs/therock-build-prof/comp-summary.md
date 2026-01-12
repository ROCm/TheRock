# TheRock Build Resource Observability Report

## Build Concurrency Summary

- build_span_min: 13.68
- avg_concurrency_build: 108.43
- peak_concurrency_build: 397
- histogram_bin_seconds: 60

### Concurrency Over Time (binned)

| t_offset_min | avg_concurrency |
| --- | --- |
| 0.00 | 117.58 |
| 1.00 | 155.18 |
| 2.00 | 206.29 |
| 3.00 | 237.91 |
| 4.00 | 143.35 |
| 5.00 | 178.00 |
| 6.00 | 246.67 |
| 7.00 | 23.05 |
| 8.00 | 2.13 |
| 9.00 | 73.87 |
| 10.00 | 64.66 |
| 11.00 | 30.39 |
| 12.00 | 3.26 |
| 13.00 | 1.00 |

## Per-Component Summary

| component | wall_time_sum (minutes) | wall_time_span (minutes) | wall_time_est_elapsed (minutes) | avg_concurrency | peak_concurrency | critical_path_est_min | cpu_sum (minutes) | user_sum (minutes) | sys_sum (minutes) | avg_threads | max_rss_mb | max_rss_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amd-llvm | 1210.35 | 12.49 | 11.16 | 96.94 | 258 | 12.49 | 1198.76 | 1071.97 | 126.79 | 0.99 | 6663.41 | 6.5072 |
| sysdeps | 117.74 | 1.93 | 1.09 | 60.85 | 258 | 1.93 | 112.24 | 89.09 | 23.15 | 0.95 | 660.39 | 0.6449 |
| host-blas | 111.47 | 1.43 | 1.03 | 78.04 | 235 | 1.43 | 109.26 | 59.13 | 50.13 | 0.98 | 182.00 | 0.1777 |
| unknown | 36.14 | 1.87 | 0.33 | 19.34 | 219 | 1.87 | 35.51 | 28.94 | 6.57 | 0.98 | 726.08 | 0.7091 |
| fftw3 | 3.19 | 0.17 | 0.03 | 18.66 | 170 | 0.17 | 3.06 | 1.96 | 1.10 | 0.96 | 70.28 | 0.0686 |
| support | 1.66 | 1.65 | 0.02 | 1.01 | 10 | 1.65 | 1.64 | 1.24 | 0.39 | 0.99 | 392.00 | 0.3828 |
| base | 1.11 | 1.70 | 0.01 | 0.65 | 11 | 1.70 | 1.09 | 0.86 | 0.23 | 0.99 | 386.00 | 0.3770 |
| prim | 0.83 | 0.18 | 0.01 | 4.70 | 6 | 0.18 | 0.82 | 0.64 | 0.18 | 0.99 | 372.00 | 0.3633 |
| rand | 0.43 | 1.63 | 0.00 | 0.26 | 13 | 1.63 | 0.42 | 0.27 | 0.15 | 0.98 | 186.00 | 0.1816 |
| blas | 0.08 | 0.04 | 0.00 | 1.78 | 4 | 0.04 | 0.05 | 0.05 | 0.01 | 0.69 | 42.00 | 0.0410 |
| composable-kernel | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| core-hip | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| core-hipinfo | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| core-ocl | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| core-runtime | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| fft | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| hipdnn | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| hipify | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| host-suite-sparse | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| miopen | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| miopen-plugin | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rccl | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rdc | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rocprofiler-compute | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rocprofiler-sdk | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rocprofiler-systems | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |
| rocwmma | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0000 |

---

## General FAQ

(See HTML report for formatted FAQ.)

Key terms:
- avg_concurrency: parallelism across processes (tool overlap)
- avg_threads: parallelism within a process (cpu_sum / wall_time_sum)
- critical_path_est_min: proxy via wall_time_span_min

## CI Regression Check (Concurrency)

- baseline: concurrency_baseline.json
- threshold_drop: 0.20
- avg_concurrency_build: current=108.43, baseline=108.43, drop=0.00%
- peak_concurrency_build: current=397, baseline=397, drop=0.00%

✅ No concurrency collapse detected.
