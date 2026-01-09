# ============================================================================
# 🚀 RYZANSTEIN COMPLETE ECOSYSTEM - MASTER ORCHESTRATION SCRIPT
# ============================================================================
#
# Purpose: Fully automated setup of BOTH Desktop App AND VS Code Extension
# Author: ARCHITECT Mode
# Date: January 8, 2026
#
# This orchestration script handles:
# - Sequential setup of Desktop Application
# - Sequential setup of VS Code Extension
# - Coordination between both platforms
# - Final validation and testing
#
# Usage: .\SETUP_COMPLETE_ECOSYSTEM_MASTER.ps1
# ============================================================================

param(
    [ValidateSet("Full", "Desktop", "Extension", "Dev")]
    [string]$SetupType = "Full",
    [switch]$SkipDependencies = $false,
    [switch]$Verbose = $false
)

# ============================================================================
# CONFIGURATION
# ============================================================================

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptPath
$desktopPath = Join-Path $projectRoot "desktop"
$extensionPath = Join-Path $projectRoot "vscode-extension"

$colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
    Progress = "Magenta"
    Header = "DarkCyan"
}

$startTime = Get-Date

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Write-Banner {
    param([string]$message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor $colors.Header
    Write-Host "║ $($message.PadRight(78)) ║" -ForegroundColor $colors.Header
    Write-Host "╚════════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor $colors.Header
    Write-Host ""
}

function Write-Section {
    param([string]$message, [int]$number)
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $colors.Info
    Write-Host "PHASE $number: $message" -ForegroundColor $colors.Info
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $colors.Info
}

function Write-Success {
    param([string]$message)
    Write-Host "  ✓ $message" -ForegroundColor $colors.Success
}

function Write-Warning {
    param([string]$message)
    Write-Host "  ⚠ $message" -ForegroundColor $colors.Warning
}

function Write-Error {
    param([string]$message)
    Write-Host "  ✗ $message" -ForegroundColor $colors.Error
}

function Write-Progress {
    param([string]$message)
    Write-Host "  ▶ $message" -ForegroundColor $colors.Progress
}

function Test-CommandExists {
    param([string]$command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

function Perform-PreflightChecks {
    Write-Section "PRE-FLIGHT CHECKS" 0
    
    Write-Progress "Verifying system requirements..."
    
    $checks = @{
        "PowerShell 5.0+" = { $PSVersionTable.PSVersion.Major -ge 5 }
        "Administrator Rights" = { ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator") }
    }
    
    $allGood = $true
    foreach ($check in $checks.GetEnumerator()) {
        if (& $check.Value) {
            Write-Success $check.Name
        } else {
            Write-Error $check.Name
            $allGood = $false
        }
    }
    
    if (-not $allGood) {
        throw "System requirements check failed"
    }
    
    Write-Host ""
}

# ============================================================================
# DESKTOP APP SETUP
# ============================================================================

function Setup-DesktopApp {
    Write-Section "DESKTOP APPLICATION SETUP" 1
    
    if (-not (Test-Path $desktopPath)) {
        Write-Error "Desktop directory not found at $desktopPath"
        throw "Desktop path not found"
    }
    
    $setupScript = Join-Path $desktopPath "SETUP_DESKTOP_APP_MASTER.ps1"
    
    if (-not (Test-Path $setupScript)) {
        Write-Error "Setup script not found at $setupScript"
        throw "Setup script not found"
    }
    
    Write-Progress "Running desktop app setup..."
    Write-Host ""
    
    $params = @{
        SkipDependencies = $SkipDependencies
        DevelopmentOnly = $true
    }
    
    Push-Location $desktopPath
    try {
        & $setupScript @params
        Write-Success "Desktop app setup completed"
    } catch {
        Write-Error "Desktop app setup failed: $_"
        throw
    } finally {
        Pop-Location
    }
    
    Write-Host ""
}

# ============================================================================
# VS CODE EXTENSION SETUP
# ============================================================================

function Setup-VSCodeExtension {
    Write-Section "VS CODE EXTENSION SETUP" 2
    
    if (-not (Test-Path $extensionPath)) {
        Write-Error "Extension directory not found at $extensionPath"
        throw "Extension path not found"
    }
    
    $setupScript = Join-Path $extensionPath "SETUP_VSCODE_EXTENSION_MASTER.ps1"
    
    if (-not (Test-Path $setupScript)) {
        Write-Error "Setup script not found at $setupScript"
        throw "Setup script not found"
    }
    
    Write-Progress "Running VS Code extension setup..."
    Write-Host ""
    
    $params = @{
        SkipDependencies = $SkipDependencies
        PackageOnly = $false
    }
    
    Push-Location $extensionPath
    try {
        & $setupScript @params
        Write-Success "VS Code extension setup completed"
    } catch {
        Write-Error "VS Code extension setup failed: $_"
        throw
    } finally {
        Pop-Location
    }
    
    Write-Host ""
}

# ============================================================================
# INTEGRATION VERIFICATION
# ============================================================================

function Verify-Integration {
    Write-Section "INTEGRATION VERIFICATION" 3
    
    Write-Progress "Verifying Desktop App..."
    
    $desktopChecks = @(
        (Test-Path (Join-Path $desktopPath "packages\desktop\src\components\ChatPanel.tsx")),
        (Test-Path (Join-Path $desktopPath "packages\desktop\src\hooks\useChat.ts")),
        (Test-Path (Join-Path $desktopPath "packages\desktop\src\store\chatStore.ts")),
        (Test-Path (Join-Path $desktopPath "packages\desktop\src\services\api.ts")),
        (Test-Path (Join-Path $desktopPath "cmd\ryzanstein\main.go")),
        (Test-Path (Join-Path $desktopPath "wails.json"))
    )
    
    if ($desktopChecks -contains $false) {
        Write-Warning "Some desktop app files are missing"
    } else {
        Write-Success "Desktop app files verified"
    }
    
    Write-Progress "Verifying VS Code Extension..."
    
    $extensionChecks = @(
        (Test-Path (Join-Path $extensionPath "src\extension.ts")),
        (Test-Path (Join-Path $extensionPath "src\webview\chatPanel.ts")),
        (Test-Path (Join-Path $extensionPath "src\services\ryzansteinAPI.ts")),
        (Test-Path (Join-Path $extensionPath "tsconfig.json")),
        (Test-Path (Join-Path $extensionPath "package.json"))
    )
    
    if ($extensionChecks -contains $false) {
        Write-Warning "Some extension files are missing"
    } else {
        Write-Success "VS Code extension files verified"
    }
    
    Write-Host ""
}

# ============================================================================
# FINAL STATUS REPORT
# ============================================================================

function Show-FinalReport {
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Banner "✅ SETUP COMPLETED SUCCESSFULLY"
    
    Write-Host ""
    Write-Host "Setup Summary:" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "📱 Desktop Application" -ForegroundColor $colors.Progress
    Write-Host "   Location: $desktopPath" -ForegroundColor $colors.Info
    Write-Host "   Status: Ready for Development" -ForegroundColor $colors.Success
    Write-Host "   Start Dev: cd $desktopPath && wails dev" -ForegroundColor $colors.Info
    Write-Host "   Build Prod: cd $desktopPath && wails build -nsis" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "💻 VS Code Extension" -ForegroundColor $colors.Progress
    Write-Host "   Location: $extensionPath" -ForegroundColor $colors.Info
    Write-Host "   Status: Ready for Development" -ForegroundColor $colors.Success
    Write-Host "   Start Dev: cd $extensionPath && npm run watch" -ForegroundColor $colors.Info
    Write-Host "   Build: cd $extensionPath && npm run compile" -ForegroundColor $colors.Info
    Write-Host "   Package: cd $extensionPath && npm run package" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "⏱️  Setup Duration: $($duration.TotalSeconds -as [int]) seconds" -ForegroundColor $colors.Progress
    Write-Host ""
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "Next Steps:" -ForegroundColor $colors.Warning
    Write-Host ""
    Write-Host "  1. Desktop App Development:" -ForegroundColor $colors.Progress
    Write-Host "     cd $desktopPath" -ForegroundColor $colors.Info
    Write-Host "     wails dev" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "  2. VS Code Extension Development:" -ForegroundColor $colors.Progress
    Write-Host "     cd $extensionPath" -ForegroundColor $colors.Info
    Write-Host "     npm run watch" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "  3. Test Integration:" -ForegroundColor $colors.Progress
    Write-Host "     - Start Desktop App (step 1)" -ForegroundColor $colors.Info
    Write-Host "     - Launch VS Code" -ForegroundColor $colors.Info
    Write-Host "     - Press F5 to start extension development host" -ForegroundColor $colors.Info
    Write-Host "     - Test chat functionality" -ForegroundColor $colors.Info
    Write-Host ""
    
    Write-Host "Resources:" -ForegroundColor $colors.Warning
    Write-Host ""
    Write-Host "  📖 Documentation: See NEXT_STEPS_DETAILED_ACTION_PLAN.md" -ForegroundColor $colors.Info
    Write-Host "  🔗 API Server: http://localhost:8000" -ForegroundColor $colors.Info
    Write-Host "  📱 Desktop: Windows/macOS/Linux" -ForegroundColor $colors.Info
    Write-Host "  💻 Extension: VS Code 1.85.0+" -ForegroundColor $colors.Info
    Write-Host ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function Main {
    Clear-Host
    Write-Banner "🚀 RYZANSTEIN COMPLETE ECOSYSTEM SETUP"
    
    try {
        Perform-PreflightChecks
        
        switch ($SetupType) {
            "Full" {
                Setup-DesktopApp
                Setup-VSCodeExtension
                Verify-Integration
                Show-FinalReport
            }
            "Desktop" {
                Setup-DesktopApp
                Write-Success "Desktop app setup completed"
            }
            "Extension" {
                Setup-VSCodeExtension
                Write-Success "VS Code extension setup completed"
            }
            "Dev" {
                Write-Progress "Development mode - skipping setup"
                Show-FinalReport
            }
        }
        
    } catch {
        Write-Error "Setup failed: $_"
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor $colors.Warning
        Write-Host "  1. Ensure all dependencies are installed (Go, Node.js, npm)" -ForegroundColor $colors.Info
        Write-Host "  2. Check that directories exist at specified paths" -ForegroundColor $colors.Info
        Write-Host "  3. Run as Administrator" -ForegroundColor $colors.Info
        Write-Host ""
        exit 1
    }
}

# Run main
Main
