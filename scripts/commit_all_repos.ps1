#!/usr/bin/env pwsh
<#
.SYNOPSIS
Commits and pushes all changes across all 18 dependent repositories and main repo
.DESCRIPTION
Goes through each submodule, commits any changes, then commits main repo changes
.PARAMETER Message
Optional custom commit message
#>

param(
    [string]$Message = "docs: update Next Steps Master Action Plan for autonomous development"
)

$ErrorActionPreference = 'Continue'
$repos = @(
    'agentmem', 'ann-hybrid', 'archaeo', 'causedb', 'cpu-infer', 'flowstate',
    'intent-spec', 'mcp-mesh', 'neurectomy-shell', 'semlog', 'sigma-api',
    'sigma-compress', 'sigma-diff', 'sigma-index', 'sigma-telemetry',
    'vault-git', 'zkaudit', 'dep-bloom'
)

$mainRepo = 'S:\Ryot'
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

Write-Host "🚀 Starting commit and push across all repositories" -ForegroundColor Cyan
Write-Host "Timestamp: $timestamp" -ForegroundColor Gray
Write-Host ""

$commitCount = 0
$pushCount = 0
$failedRepos = @()

# Process each submodule
foreach ($repo in $repos) {
    $repoPath = Join-Path $mainRepo "dependencies" $repo
    
    if (-not (Test-Path $repoPath)) {
        Write-Host "⚠️  $($repo): Directory not found, skipping..." -ForegroundColor Yellow
        continue
    }
    
    Push-Location $repoPath
    
    # Check for changes
    $status = & git status --porcelain 2>&1
    
    if ($status) {
        Write-Host "📝 $($repo): Found changes, committing..." -ForegroundColor Yellow
        
        # Stage all changes
        & git add -A 2>&1 | Out-Null
        
        # Create a descriptive commit message
        $commitMsg = "chore: update scaffolding for Master Action Plan`n`n- Part of automated rollout for autonomous development framework`n- Updated dependencies aligned with ecosystem architecture`n- Commit timestamp: $timestamp"
        
        # Commit with message
        & git commit -m $commitMsg 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Committed" -ForegroundColor Green
            $commitCount++
            
            # Push changes
            & git push origin main 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ Pushed" -ForegroundColor Green
                $pushCount++
            }
            else {
                Write-Host "  ❌ Push failed" -ForegroundColor Red
                $failedRepos += $repo
            }
        }
        else {
            Write-Host "  ⚠️  No changes to commit (working tree clean)" -ForegroundColor Gray
        }
    }
    else {
        Write-Host "✅ $($repo): No changes" -ForegroundColor Green
    }
    
    Pop-Location
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Now handle main repository
Write-Host "📦 Processing main repository (Ryzanstein)..." -ForegroundColor Cyan

Push-Location $mainRepo

# Stage IMPLEMENTATION_QUICK_REFERENCE.md if not already staged
& git add IMPLEMENTATION_QUICK_REFERENCE.md 2>&1 | Out-Null

# Also add the submodule updates
& git add dependencies/ 2>&1 | Out-Null

$mainStatus = & git status --porcelain 2>&1

if ($mainStatus) {
    Write-Host "📝 Committing main repository changes..." -ForegroundColor Yellow
    
    $mainCommitMsg = "docs(architect): add Next Steps Master Action Plan for autonomous development`n`n" +
    "- Comprehensive 8-phase automation roadmap (18 days)`n" +
    "- Phase 1: Foundation automation (CI/CD, dependencies, quality gates)`n" +
    "- Phase 2: Development automation (auto-updates, integration tests, AI review)`n" +
    "- Phase 3: Monitoring & observability (health dashboard, regression detection)`n" +
    "- Phase 4: Security automation (scanning, secrets, compliance)`n" +
    "- Phase 5: Release automation (semantic versioning, coordinated releases)`n" +
    "- Phase 6: Analytics & insights (metrics, predictions, recommendations)`n" +
    "- Phase 7: Developer productivity (environment setup, AI assistants)`n" +
    "- Phase 8: Continuous improvement (retrospectives, self-healing)`n" +
    "`n" +
    "KPI targets:`n" +
    "- Time to first commit: 45 min → 5 min`n" +
    "- CI/CD pass rate: 75% → 95%+`n" +
    "- Deployment frequency: 1/week → 5+/week`n" +
    "- MTTR: 4 hours → <1 hour`n" +
    "- Expected ROI: 10x developer productivity within 30 days`n" +
    "`n" +
    "Includes implementation checklist, success metrics, and quick-start commands.`n" +
    "`n" +
    "Updated: $timestamp"
    
    & git commit -m $mainCommitMsg 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Committed" -ForegroundColor Green
        
        # Push changes
        & git push origin sprint6/api-integration 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Pushed to sprint6/api-integration" -ForegroundColor Green
        }
        else {
            Write-Host "  ❌ Push failed" -ForegroundColor Red
            $failedRepos += 'Ryzanstein (main)'
        }
    }
    else {
        Write-Host "  ❌ Commit failed" -ForegroundColor Red
        $failedRepos += 'Ryzanstein (main)'
    }
}
else {
    Write-Host "✅ No staged changes in main repository" -ForegroundColor Green
}

Pop-Location

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "📊 Summary:" -ForegroundColor Cyan
Write-Host "  Total commits: $commitCount" -ForegroundColor Green
Write-Host "  Total pushes: $pushCount" -ForegroundColor Green

if ($failedRepos.Count -gt 0) {
    Write-Host "  Failed repos: $($failedRepos -join ', ')" -ForegroundColor Red
}
else {
    Write-Host "  Failed repos: None" -ForegroundColor Green
}

Write-Host ""
Write-Host "✅ Commit and push operation completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📈 Next: Begin Phase 1 (Foundation Automation) implementation" -ForegroundColor Cyan
