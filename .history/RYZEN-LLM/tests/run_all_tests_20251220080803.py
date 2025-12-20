#!/usr/bin/env python3
"""
RYZEN-LLM Phase 2 Comprehensive Test Runner & Report Generator
[REF:PHASE2-TEST-RUNNER] - Unified test execution and validation

Orchestrates:
1. Unit test execution (C++ and Python)
2. Integration test execution
3. End-to-end workflow validation
4. Performance benchmarking
5. Test report generation
6. Test result analysis
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import platform

PROJECT_ROOT = Path(__file__).parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
RYZEN_LLM_DIR = PROJECT_ROOT / "RYZEN-LLM"
BUILD_DIR = RYZEN_LLM_DIR / "build"

# ===========================================================================
# Test Execution Infrastructure
# ===========================================================================

class TestRunner:
    """Unified test runner for all test suites"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "test_suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0
            }
        }
        self.execution_times = {}
    
    def run_python_tests(self, test_file: Path, suite_name: str) -> Dict:
        """Run Python test suite using pytest"""
        print(f"\n📝 Running {suite_name}...")
        print(f"   Test file: {test_file}")
        
        if not test_file.exists():
            print(f"   ❌ Test file not found: {test_file}")
            return {
                "status": "ERROR",
                "error": f"Test file not found: {test_file}",
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--color=yes",
            "-ra",
            f"--junit-xml={TESTS_DIR / f'results_{suite_name}.xml'}",
            f"--html={TESTS_DIR / f'report_{suite_name}.html'}",
            "--self-contained-html",
            "--disable-warnings"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            elapsed = time.time() - start_time
            self.execution_times[suite_name] = elapsed
            
            # Parse output for test counts
            output = result.stdout
            status = "PASSED" if result.returncode == 0 else "FAILED"
            
            # Extract test statistics
            import re
            
            # Look for pytest summary line
            summary_match = re.search(
                r'(\d+) passed|(\d+) failed|(\d+) error|(\d+) skipped',
                output
            )
            
            return {
                "status": status,
                "return_code": result.returncode,
                "elapsed_time_sec": round(elapsed, 2),
                "output": output[-2000:] if len(output) > 2000 else output,  # Last 2000 chars
                "passed": len(re.findall(r'PASSED', output)),
                "failed": len(re.findall(r'FAILED', output)),
                "skipped": len(re.findall(r'SKIPPED', output)),
                "errors": len(re.findall(r'ERROR', output))
            }
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "status": "TIMEOUT",
                "error": f"Test suite timed out after {elapsed}s",
                "elapsed_time_sec": round(elapsed, 2),
                "passed": 0,
                "failed": 1,
                "skipped": 0
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
    
    def run_cpp_tests(self) -> Dict:
        """Run C++ tests using CTest"""
        print(f"\n📝 Running C++ Unit Tests (CTest)...")
        
        if not BUILD_DIR.exists():
            print(f"   ⚠️  Build directory not found: {BUILD_DIR}")
            print(f"   Run: cmake --build {BUILD_DIR} first")
            return {
                "status": "SKIPPED",
                "error": "Build directory not found",
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        
        start_time = time.time()
        
        try:
            # Run CTest
            result = subprocess.run(
                ["ctest", "-V"],
                cwd=str(BUILD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            elapsed = time.time() - start_time
            self.execution_times["cpp_unit"] = elapsed
            
            output = result.stdout + result.stderr
            status = "PASSED" if result.returncode == 0 else "FAILED"
            
            import re
            passed = len(re.findall(r'100% tests passed', output))
            failed = result.returncode if result.returncode != 0 else 0
            
            return {
                "status": status,
                "return_code": result.returncode,
                "elapsed_time_sec": round(elapsed, 2),
                "output": output[-2000:] if len(output) > 2000 else output,
                "passed": passed,
                "failed": failed,
                "skipped": 0
            }
        
        except FileNotFoundError:
            print(f"   ⚠️  CTest not found. Install CMake/CTest or run: cmake --build {BUILD_DIR}")
            return {
                "status": "SKIPPED",
                "error": "CTest not found",
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return {
                "status": "TIMEOUT",
                "error": f"C++ tests timed out after {elapsed}s",
                "elapsed_time_sec": round(elapsed, 2),
                "passed": 0,
                "failed": 1,
                "skipped": 0
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
    
    def run_all_tests(self) -> bool:
        """Run all test suites"""
        print("="*80)
        print("RYZEN-LLM PHASE 2 - COMPREHENSIVE TEST EXECUTION")
        print("="*80)
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"Platform: {platform.system()}")
        print(f"Python: {sys.version}")
        print()
        
        # Test suites to run
        test_suites = [
            (TESTS_DIR / "e2e" / "test_end_to_end_pipeline.py", "e2e_pipeline"),
            (TESTS_DIR / "e2e" / "test_cpp_integration.py", "cpp_integration"),
            (TESTS_DIR / "integration" / "test_bitnet_generation.py", "integration_bitnet"),
        ]
        
        # Run C++ unit tests
        print("\n" + "="*80)
        print("PHASE 1: C++ UNIT TESTS")
        print("="*80)
        cpp_results = self.run_cpp_tests()
        self.results["test_suites"]["cpp_unit"] = cpp_results
        self.update_summary(cpp_results)
        
        # Run Python test suites
        print("\n" + "="*80)
        print("PHASE 2: PYTHON INTEGRATION TESTS")
        print("="*80)
        for test_file, suite_name in test_suites:
            result = self.run_python_tests(test_file, suite_name)
            self.results["test_suites"][suite_name] = result
            self.update_summary(result)
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        # Return overall success
        return self.results["summary"]["failed"] == 0 and self.results["summary"]["errors"] == 0
    
    def update_summary(self, result: Dict):
        """Update overall summary"""
        self.results["summary"]["total_tests"] += result.get("passed", 0) + result.get("failed", 0) + result.get("skipped", 0)
        self.results["summary"]["passed"] += result.get("passed", 0)
        self.results["summary"]["failed"] += result.get("failed", 0)
        self.results["summary"]["skipped"] += result.get("skipped", 0)
        self.results["summary"]["errors"] += result.get("errors", 0) if "error" in result else 0
    
    def print_summary(self):
        """Print test execution summary"""
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        
        summary = self.results["summary"]
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests:  {summary['total_tests']}")
        print(f"   ✅ Passed:    {summary['passed']}")
        print(f"   ❌ Failed:    {summary['failed']}")
        print(f"   ⊘  Skipped:   {summary['skipped']}")
        print(f"   💥 Errors:    {summary['errors']}")
        
        success_rate = (summary['passed'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        print(f"\n   Success Rate: {success_rate:.1f}%")
        
        print(f"\n⏱️  Execution Times:")
        for suite, elapsed in self.execution_times.items():
            print(f"   {suite}: {elapsed:.2f}s")
        
        total_time = sum(self.execution_times.values())
        print(f"   Total: {total_time:.2f}s")
        
        print("\n📋 Detailed Results:")
        for suite_name, result in self.results["test_suites"].items():
            status_icon = "✅" if result.get("status") == "PASSED" else "❌" if result.get("status") == "FAILED" else "⊘"
            print(f"   {status_icon} {suite_name}: {result.get('status')}")
            if result.get("error"):
                print(f"      Error: {result.get('error')}")
    
    def save_results(self):
        """Save test results to JSON"""
        results_file = TESTS_DIR / "test_results_summary.json"
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


# ===========================================================================
# Test Validation & Quality Gates
# ===========================================================================

class TestValidator:
    """Validate test results against quality gates"""
    
    @staticmethod
    def validate_quality_gates(results: Dict) -> Tuple[bool, List[str]]:
        """Check if results meet quality gates for release"""
        issues = []
        
        summary = results["summary"]
        
        # Gate 1: No critical failures
        if summary["failed"] > 0:
            issues.append(f"❌ {summary['failed']} tests failed (gate: 0 failures)")
        
        # Gate 2: No errors
        if summary["errors"] > 0:
            issues.append(f"❌ {summary['errors']} test errors (gate: 0 errors)")
        
        # Gate 3: Minimum success rate (95%)
        total = summary["total_tests"]
        if total > 0:
            success_rate = (summary["passed"] / total) * 100
            if success_rate < 95:
                issues.append(f"❌ Success rate {success_rate:.1f}% < 95% (gate: 95%)")
        
        # Gate 4: C++ tests must pass
        if "cpp_unit" in results["test_suites"]:
            cpp_result = results["test_suites"]["cpp_unit"]
            if cpp_result.get("status") == "FAILED":
                issues.append("❌ C++ unit tests failed (critical)")
        
        # Gate 5: E2E tests must pass
        if "e2e_pipeline" in results["test_suites"]:
            e2e_result = results["test_suites"]["e2e_pipeline"]
            if e2e_result.get("status") == "FAILED":
                issues.append("❌ E2E pipeline tests failed (critical)")
        
        passed = len(issues) == 0
        return passed, issues


# ===========================================================================
# Main Execution
# ===========================================================================

def main():
    """Main test execution entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Validate quality gates
    print("\n" + "="*80)
    print("QUALITY GATES VALIDATION")
    print("="*80)
    
    validator = TestValidator()
    gates_passed, issues = validator.validate_quality_gates(runner.results)
    
    if gates_passed:
        print("\n✅ All quality gates PASSED")
        print("\n🎉 Ready for performance benchmarking!")
        return 0
    else:
        print("\n❌ Quality gate violations:")
        for issue in issues:
            print(f"   {issue}")
        print("\n⚠️  Fix failing tests before proceeding to benchmarking")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
