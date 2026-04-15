# Phase 6: Integrated Monitoring Setup
# This script opens 5 VSCode terminals and starts all port-forwards
# Run with: powershell -ExecutionPolicy Bypass -File start_phase6_monitoring.ps1

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  PHASE 6: INTEGRATED MONITORING SETUP" -ForegroundColor Cyan
Write-Host "  Opening 5 VSCode Terminals with Port-Forwards" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if VSCode is open
$vscodeRunning = Get-Process code -ErrorAction SilentlyContinue
if (-not $vscodeRunning) {
    Write-Host "Warning: VSCode is not running. Please start VSCode first." -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Opening integrated terminals..." -ForegroundColor Cyan
Write-Host ""

# Show instructions for manual setup
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "MANUAL SETUP - Copy and paste into separate terminal tabs/windows" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Terminal 1 - API Port-Forward (8000:8000)" -ForegroundColor Yellow
Write-Host "kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000" -ForegroundColor White
Write-Host ""

Write-Host "Terminal 2 - Prometheus Port-Forward (9090:9090)" -ForegroundColor Yellow
Write-Host "kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090" -ForegroundColor White
Write-Host ""

Write-Host "Terminal 3 - Grafana Port-Forward (3000:3000)" -ForegroundColor Yellow
Write-Host "kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000" -ForegroundColor White
Write-Host ""

Write-Host "Terminal 4 - Jaeger Port-Forward (16686:16686)" -ForegroundColor Yellow
Write-Host "kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686" -ForegroundColor White
Write-Host ""

Write-Host "Terminal 5 - AlertManager Port-Forward (9093:9093)" -ForegroundColor Yellow
Write-Host "kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093" -ForegroundColor White
Write-Host ""

# Try to open VSCode integrated terminal
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Option 1: VSCode Integrated Terminal" -ForegroundColor Green
Write-Host "  1. Press Ctrl+` to open integrated terminal in VSCode" -ForegroundColor White
Write-Host "  2. Create 5 tabs (Ctrl+Shift+`)" -ForegroundColor White
Write-Host "  3. Paste each kubectl command above into separate tabs" -ForegroundColor White
Write-Host "  4. Keep all 5 running during Phase 6" -ForegroundColor White
Write-Host ""

Write-Host "Option 2: Windows Terminal (Recommended)" -ForegroundColor Green
Write-Host "  1. Open Windows Terminal" -ForegroundColor White
Write-Host "  2. Create 5 panes (Alt+Shift+D for split)" -ForegroundColor White
Write-Host "  3. Paste each command into separate panes" -ForegroundColor White
Write-Host "  4. Keep all 5 running during Phase 6" -ForegroundColor White
Write-Host ""

Write-Host "Option 3: Regular PowerShell Windows" -ForegroundColor Green
Write-Host "  1. Open 5 PowerShell windows" -ForegroundColor White
Write-Host "  2. Paste one kubectl command in each" -ForegroundColor White
Write-Host "  3. Press Enter and keep all 5 running" -ForegroundColor White
Write-Host ""

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "VERIFICATION" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Once all 5 port-forwards are running (showing 'Forwarding...'), run:" -ForegroundColor Green
Write-Host "  s:\Ryot\verify_phase6_monitoring.bat" -ForegroundColor White
Write-Host ""

Write-Host "Then open these URLs in your browser:" -ForegroundColor Green
Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "  - Grafana:    http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  - Jaeger:     http://localhost:16686" -ForegroundColor White
Write-Host "  - AlertMgr:   http://localhost:9093" -ForegroundColor White
Write-Host ""

Write-Host "Finally, follow the Phase 6 checklist:" -ForegroundColor Green
Write-Host "  s:\Ryot\PHASE6_EXECUTION_CHECKLIST.md" -ForegroundColor White
Write-Host ""

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "Phase 6 Setup Ready!" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
