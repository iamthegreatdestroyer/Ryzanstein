@echo off
REM Phase 6: Automated Monitoring Verification
REM This script verifies all monitoring services are accessible and responding

setlocal enabledelayedexpansion

echo ============================================================
echo  Phase 6: Integration Testing - Monitoring Verification
echo  Ryzanstein LLM API
echo ============================================================
echo.

set /a passed=0
set /a failed=0

echo [CHECK] Verifying monitoring services connectivity...
echo.

REM Test API
echo Testing API Health...
curl -s -m 5 http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo   ❌ FAIL - API not responding
    set /a failed+=1
) else (
    echo   ✅ PASS - API health endpoint responding
    set /a passed+=1
)

REM Test Prometheus
echo Testing Prometheus...
curl -s -m 5 http://localhost:9090/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ❌ FAIL - Prometheus not responding
    set /a failed+=1
) else (
    echo   ✅ PASS - Prometheus health check passed
    set /a passed+=1
)

REM Test Grafana
echo Testing Grafana...
curl -s -m 5 http://localhost:3000/api/health > nul 2>&1
if errorlevel 1 (
    echo   ❌ FAIL - Grafana not responding
    set /a failed+=1
) else (
    echo   ✅ PASS - Grafana health check passed
    set /a passed+=1
)

REM Test Jaeger
echo Testing Jaeger...
curl -s -m 5 -I http://localhost:16686/ 2>&1 | find "200" > nul
if errorlevel 1 (
    echo   ❌ FAIL - Jaeger not responding
    set /a failed+=1
) else (
    echo   ✅ PASS - Jaeger is accessible
    set /a passed+=1
)

REM Test AlertManager
echo Testing AlertManager...
curl -s -m 5 http://localhost:9093/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ❌ FAIL - AlertManager not responding
    set /a failed+=1
) else (
    echo   ✅ PASS - AlertManager health check passed
    set /a passed+=1
)

echo.
echo ============================================================
echo  Results Summary
echo ============================================================
echo  Services Verified: !passed! / 5
echo  Services Failed:   !failed! / 5
echo.

if !failed! equ 0 (
    echo ✅ All monitoring services are operational!
    echo.
    echo Dashboard URLs:
    echo   - API:           http://localhost:8000/health
    echo   - Prometheus:    http://localhost:9090
    echo   - Grafana:       http://localhost:3000 (admin/admin)
    echo   - Jaeger:        http://localhost:16686
    echo   - AlertManager:  http://localhost:9093
    echo.
    echo Next Steps:
    echo   1. Open each dashboard URL in your browser
    echo   2. Follow PHASE6_EXECUTION_CHECKLIST.md
    echo   3. Verify metrics collection and visualization
    echo   4. Run test queries in Prometheus
    echo   5. Review Grafana dashboards
    echo   6. Inspect Jaeger traces
    echo   7. Check AlertManager alerts
    echo.
    pause
    exit /b 0
) else (
    echo ❌ Some services are not responding!
    echo.
    echo Make sure port-forwards are active:
    echo   kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
    echo   kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
    echo   kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
    echo   kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
    echo   kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093
    echo.
    pause
    exit /b 1
)
