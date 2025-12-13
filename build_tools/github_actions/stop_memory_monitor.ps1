# Stop memory monitoring if enabled
if ("${{ inputs.monitor_memory }}" -eq "true") {
  $monitorPid = Get-Content "${env:BUILD_DIR}\logs\memory_monitor.pid" | Select-Object -First 1
  Write-Host "Gracefully stopping memory monitor (PID: $monitorPid)"
  python build_tools\graceful_shutdown.py $monitorPid --timeout 60 --stop-signal-file "${env:BUILD_DIR}\logs\stop_monitor_${env:GITHUB_JOB}.signal"
  if (Test-Path "${env:BUILD_DIR}\logs\monitor_output_${env:GITHUB_JOB}.txt") {
    Write-Host "=== Memory Monitor Output ==="
    Get-Content "${env:BUILD_DIR}\logs\monitor_output_${env:GITHUB_JOB}.txt"
  }
}
