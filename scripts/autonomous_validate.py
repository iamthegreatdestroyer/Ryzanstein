#!/usr/bin/env python3
"""
autonomous_validate.py — Full-stack validation runner for Ryzanstein LLM.

Runs all cross-language checks without human interaction. Designed to be
invoked by CI, an AI agent, or a developer with zero arguments.

Usage:
    python scripts/autonomous_validate.py
    python scripts/autonomous_validate.py --fast        # skip slow tests
    python scripts/autonomous_validate.py --json        # machine-readable output
    python scripts/autonomous_validate.py --min-pass 8  # require N passing checks
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    output: str
    duration_ms: float
    skipped: bool = False
    skip_reason: str = ""


# ─────────────────────────────────────────────────
# Individual check definitions
# ─────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.resolve()


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    timeout: int = 120,
    env: Optional[dict] = None,
) -> tuple[bool, str]:
    """Execute a command, return (success, combined_output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return False, f"Command not found: {e}"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────
# Check registry
# ─────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    cmd: list[str]
    cwd: Optional[Path] = None
    timeout: int = 120
    required: bool = True  # if False, failure is a warning only
    fast: bool = True       # if False, skipped in --fast mode
    skip_if_missing: Optional[str] = None  # skip if this binary not found


CHECKS: list[Check] = [
    # ── Rust workspace ──
    Check(
        name="Rust: cargo check (workspace)",
        cmd=["cargo", "check", "--workspace", "--all-targets"],
        timeout=180,
        skip_if_missing="cargo",
    ),
    Check(
        name="Rust: cargo test (workspace)",
        cmd=["cargo", "test", "--workspace", "--all-features", "--", "--test-threads=4"],
        timeout=300,
        fast=False,
        skip_if_missing="cargo",
    ),
    Check(
        name="Rust: clippy (deny warnings)",
        cmd=["cargo", "clippy", "--workspace", "--all-features", "--", "-D", "warnings"],
        timeout=180,
        skip_if_missing="cargo",
    ),
    Check(
        name="Rust: cargo fmt (check)",
        cmd=["cargo", "fmt", "--all", "--", "--check"],
        skip_if_missing="cargo",
    ),
    # ── Python: agentmem ──
    Check(
        name="Python: agentmem tests",
        cmd=["python", "-m", "pytest", "dependencies/agentmem/tests", "-q", "--tb=short"],
        timeout=60,
        skip_if_missing="pytest",
    ),
    # ── Python: archaeo ──
    Check(
        name="Python: archaeo tests",
        cmd=["python", "-m", "pytest", "dependencies/archaeo/tests", "-q", "--tb=short"],
        timeout=60,
        skip_if_missing="pytest",
    ),
    # ── Python: core API tests ──
    Check(
        name="Python: core tests",
        cmd=["python", "-m", "pytest", "tests/", "-q", "--tb=short",
             "--ignore=tests/performance", "-x"],
        timeout=120,
        fast=False,
    ),
    # ── Go: mcp-mesh ──
    Check(
        name="Go: mcp-mesh tests",
        cmd=["go", "test", "./...", "-v", "-count=1"],
        cwd=REPO_ROOT / "dependencies" / "mcp-mesh",
        timeout=60,
        skip_if_missing="go",
    ),
    # ── Go: vault-git ──
    Check(
        name="Go: vault-git tests",
        cmd=["go", "test", "./...", "-v", "-count=1"],
        cwd=REPO_ROOT / "dependencies" / "vault-git",
        timeout=60,
        skip_if_missing="go",
    ),
    # ── TypeScript: flowstate ──
    Check(
        name="TS: flowstate build + test",
        cmd=["npm", "test"],
        cwd=REPO_ROOT / "dependencies" / "flowstate",
        timeout=90,
        fast=False,
        skip_if_missing="npm",
    ),
    # ── TypeScript: intent-spec ──
    Check(
        name="TS: intent-spec build + test",
        cmd=["npm", "test"],
        cwd=REPO_ROOT / "dependencies" / "intent-spec",
        timeout=90,
        fast=False,
        skip_if_missing="npm",
    ),
    # ── Security audits ──
    Check(
        name="Security: cargo audit",
        cmd=["cargo", "audit"],
        required=False,
        fast=False,
        skip_if_missing="cargo",
    ),
    Check(
        name="Security: pip-audit",
        cmd=["pip-audit", "--desc"],
        required=False,
        fast=False,
        skip_if_missing="pip-audit",
    ),
]


# ─────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────

class AutonomousValidator:
    def __init__(self, fast: bool = False):
        self.fast = fast

    def _should_skip(self, check: Check) -> tuple[bool, str]:
        if self.fast and not check.fast:
            return True, "--fast mode"
        if check.skip_if_missing:
            from shutil import which
            if not which(check.skip_if_missing):
                return True, f"'{check.skip_if_missing}' not installed"
        cwd = check.cwd or REPO_ROOT
        if not cwd.exists():
            return True, f"Directory not found: {cwd}"
        return False, ""

    def run(self, checks: list[Check] = CHECKS) -> list[CheckResult]:
        results: list[CheckResult] = []

        for check in checks:
            skip, reason = self._should_skip(check)
            if skip:
                results.append(CheckResult(
                    name=check.name,
                    passed=True,  # skipped ≠ failed
                    output="",
                    duration_ms=0,
                    skipped=True,
                    skip_reason=reason,
                ))
                marker = "⏭" if reason == "--fast mode" else "⚠"
                print(f"  {marker}  {check.name}  [{reason}]")
                continue

            t0 = time.perf_counter()
            passed, output = run_cmd(check.cmd, check.cwd, check.timeout)
            ms = (time.perf_counter() - t0) * 1000

            # Required=False → always passes (warning only)
            effective_pass = passed or not check.required

            results.append(CheckResult(
                name=check.name,
                passed=effective_pass,
                output=output[-2000:],  # last 2k chars only
                duration_ms=ms,
            ))

            icon = "✅" if passed else ("⚠️ " if not check.required else "❌")
            suffix = "" if passed else (" (warning)" if not check.required else " FAILED")
            print(f"  {icon} {check.name}  ({ms:.0f}ms){suffix}")
            if not passed and output:
                # Show last 5 lines of output on failure
                last_lines = "\n".join(output.splitlines()[-5:])
                print(f"     {last_lines.replace(chr(10), chr(10) + '     ')}")

        return results

    @staticmethod
    def report(results: list[CheckResult]) -> dict:
        actual = [r for r in results if not r.skipped]
        skipped = [r for r in results if r.skipped]
        passed = [r for r in actual if r.passed]
        failed = [r for r in actual if not r.passed]
        total_ms = sum(r.duration_ms for r in results)

        return {
            "total": len(actual),
            "passed": len(passed),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": f"{100 * len(passed) / max(len(actual), 1):.1f}%",
            "total_duration_ms": round(total_ms),
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "duration_ms": round(r.duration_ms),
                    "output_tail": r.output[-500:] if not r.passed else "",
                }
                for r in results
            ],
        }


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Ryzanstein autonomous validator")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--min-pass", type=int, default=0,
                        help="Minimum passing checks required (0 = all must pass)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to this file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RYZANSTEIN — AUTONOMOUS VALIDATION SUITE")
    print(f"  Mode: {'FAST' if args.fast else 'FULL'}")
    print("=" * 60 + "\n")

    validator = AutonomousValidator(fast=args.fast)
    results = validator.run()
    report = validator.report(results)

    # Output
    print("\n" + "=" * 60)
    print(f"  RESULT: {report['passed']}/{report['total']} checks passed "
          f"({report['success_rate']})  "
          f"[skipped: {report['skipped']}]  "
          f"[{report['total_duration_ms']}ms total]")
    print("=" * 60 + "\n")

    if args.json:
        print(json.dumps(report, indent=2))

    # Persist report
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = args.output or reports_dir / "validation_latest.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {report_path}")

    # Determine exit code
    if args.min_pass > 0:
        return 0 if report["passed"] >= args.min_pass else 1
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
