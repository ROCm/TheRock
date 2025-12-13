param(
  [string]$MonitorMemory = "false"
)

# Stop memory monitoring if it was started
if ($MonitorMemory -eq "true") {
  $pidFile = "${env:BUILD_DIR}\logs\memory_monitor.pid"
  $outputFile = "${env:BUILD_DIR}\logs\monitor_output_${env:GITHUB_JOB}.txt"

  if (Test-Path $pidFile) {
    $pid = Get-Content $pidFile
    Write-Host "Stopping memory monitor (PID: $pid)"
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
  }

  # Display the monitor output file if it exists
  if (Test-Path $outputFile) {
    Write-Host ""
    Write-Host "=== Memory Monitor Output ===" -ForegroundColor Cyan
    Get-Content $outputFile
    Write-Host "=== End of Memory Monitor Output ===" -ForegroundColor Cyan
  } else {
    Write-Host "Warning: Monitor output file not found: $outputFile" -ForegroundColor Yellow
  }
}
