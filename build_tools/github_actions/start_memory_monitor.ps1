param(
  [string]$MonitorMemory = "false"
)

# Start memory monitoring if enabled
if ($MonitorMemory -eq "true") {
  New-Item -ItemType Directory -Force -Path "${env:BUILD_DIR}\logs" | Out-Null
  $monitorArgs = @(
    "build_tools\memory_monitor.py",
    "--phase", "Build Phase",
    "--log-file", "${env:BUILD_DIR}\logs\build_memory_log_${env:GITHUB_JOB}.jsonl",
    "--stop-signal-file", "${env:BUILD_DIR}\logs\stop_monitor_${env:GITHUB_JOB}.signal",
    "--background"
  )
  $p = Start-Process -FilePath "python" -ArgumentList $monitorArgs -PassThru `
    -RedirectStandardOutput "${env:BUILD_DIR}\logs\monitor_output_${env:GITHUB_JOB}.txt"
  "$($p.Id)" | Out-File -FilePath "${env:BUILD_DIR}\logs\memory_monitor.pid" -Encoding ASCII
  Write-Host "Memory monitoring started with PID: $($p.Id)"
}
