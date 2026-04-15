@echo off
REM Check which Phase 6 ports are already listening

setlocal enabledelayedexpansion

echo ============================================================
echo  PHASE 6: CHECK ACTIVE PORTS
echo ============================================================
echo.

set /a active=0
set /a needed=0

echo Checking active port-forwards...
echo.

REM Test each port
echo Testing Port 8000 (API)...
curl -s -m 2 http://localhost:8000/health > nul 2>&1
if errorlevel 1 (
    echo   ❌ NOT listening
    set /a needed+=1
) else (
    echo   ✅ ACTIVE - API port-forward running
    set /a active+=1
)

echo Testing Port 9090 (Prometheus)...
curl -s -m 2 http://localhost:9090/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ❌ NOT listening
    set /a needed+=1
) else (
    echo   ✅ ACTIVE - Prometheus port-forward running
    set /a active+=1
)

echo Testing Port 3000 (Grafana)...
curl -s -m 2 http://localhost:3000/api/health > nul 2>&1
if errorlevel 1 (
    echo   ❌ NOT listening
    set /a needed+=1
) else (
    echo   ✅ ACTIVE - Grafana port-forward running
    set /a active+=1
)

echo Testing Port 16686 (Jaeger)...
curl -s -m 2 -I http://localhost:16686/ 2>&1 | find "200" > nul
if errorlevel 1 (
    echo   ❌ NOT listening
    set /a needed+=1
) else (
    echo   ✅ ACTIVE - Jaeger port-forward running
    set /a active+=1
)

echo Testing Port 9093 (AlertManager)...
curl -s -m 2 http://localhost:9093/-/healthy > nul 2>&1
if errorlevel 1 (
    echo   ❌ NOT listening
    set /a needed+=1
) else (
    echo   ✅ ACTIVE - AlertManager port-forward running
    set /a active+=1
)

echo.
echo ============================================================
echo  SUMMARY
echo ============================================================
echo  Active Ports:     !active! / 5
echo  Ports Needed:     !needed! / 5
echo.

if !needed! equ 0 (
    echo ✅ ALL 5 PORTS ARE ACTIVE!
    echo.
    echo You can proceed directly to Phase 6 verification:
    echo   s:\Ryot\verify_phase6_monitoring.bat
    echo.
    echo Or open dashboards:
    echo   - Prometheus: http://localhost:9090
    echo   - Grafana:    http://localhost:3000 (admin/admin)
    echo   - Jaeger:     http://localhost:16686
    echo   - AlertMgr:   http://localhost:9093
) else (
    echo ⚠️  !needed! port^(s^) still need to be forwarded
    echo.
    echo Run these commands in separate terminal windows:
    echo.
    if !active! lss 1 (
        echo   kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000
    )
    if !active! lss 2 (
        echo   kubectl port-forward -n ryzanstein-staging svc/prometheus 9090:9090
    )
    if !active! lss 3 (
        echo   kubectl port-forward -n ryzanstein-staging svc/grafana 3000:3000
    )
    if !active! lss 4 (
        echo   kubectl port-forward -n ryzanstein-staging svc/jaeger 16686:16686
    )
    if !active! lss 5 (
        echo   kubectl port-forward -n ryzanstein-staging svc/alertmanager 9093:9093
    )
)

echo.
pause
