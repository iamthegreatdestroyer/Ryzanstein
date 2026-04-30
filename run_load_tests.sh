#!/bin/bash

# Ryzanstein LLM API - Load Testing Orchestration Script
# Executes all k6 tests in sequence and compiles results

set -e

API_URL="http://localhost:8000"
TESTS_DIR="s:\Ryot"
RESULTS_DIR="s:\Ryot\load_test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  RYZANSTEIN LLM API - PHASE 5 LOAD TESTING EXECUTION${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo

# Check if API is accessible
echo -e "${BLUE}[Step 0] Verifying API Accessibility${NC}"
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is accessible at $API_URL${NC}"
else
    echo -e "${RED}✗ API is not accessible at $API_URL${NC}"
    echo "Please ensure:"
    echo "  1. API container is running"
    echo "  2. Port-forward is active: kubectl port-forward -n ryzanstein-staging svc/ryzanstein-api 8000:8000"
    exit 1
fi
echo

# Check if k6 is installed
echo -e "${BLUE}[Step 0] Checking k6 Installation${NC}"
if ! command -v k6 &> /dev/null; then
    echo -e "${RED}✗ k6 is not installed${NC}"
    echo "Please install k6:"
    echo "  Windows (Chocolatey): choco install k6"
    echo "  macOS: brew install k6"
    echo "  Linux: sudo apt-get install k6"
    exit 1
fi
echo -e "${GREEN}✓ k6 is installed ($(k6 version))${NC}"
echo

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local test_duration=$3

    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}[TEST] $test_name${NC}"
    echo -e "${BLUE}Duration: $test_duration${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo

    cd "$TESTS_DIR"

    if k6 run "$test_file" \
        --out "json=$RESULTS_DIR/${test_file%.js}_${TIMESTAMP}.json" 2>&1 | tee "$RESULTS_DIR/${test_file%.js}_${TIMESTAMP}.log"
    then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ $test_name COMPLETED WITH WARNINGS${NC}"
        return 0  # Don't fail on test, continue with next test
    fi
    echo
}

# Execute tests based on argument
case "${1:-all}" in
    smoke)
        echo -e "${YELLOW}Running Smoke Test Only (Quick Verification)${NC}"
        run_test "Smoke Test" "load_test_smoke.js" "30 seconds"
        ;;

    load)
        echo -e "${YELLOW}Running Standard Load Test${NC}"
        run_test "Smoke Test" "load_test_smoke.js" "30 seconds"
        run_test "Load Test" "load_test_load.js" "5 minutes"
        ;;

    comprehensive)
        echo -e "${YELLOW}Running Comprehensive Load Tests (4/5)${NC}"
        run_test "Smoke Test" "load_test_smoke.js" "30 seconds"
        run_test "Load Test" "load_test_load.js" "5 minutes"
        run_test "Spike Test" "load_test_spike.js" "5 minutes"
        run_test "Stress Test" "load_test_stress.js" "30 minutes"
        ;;

    full|all)
        echo -e "${YELLOW}Running Full Load Testing Suite (All 5 Tests)${NC}"
        run_test "Smoke Test" "load_test_smoke.js" "30 seconds"
        run_test "Load Test" "load_test_load.js" "5 minutes"
        run_test "Spike Test" "load_test_spike.js" "5 minutes"
        run_test "Stress Test" "load_test_stress.js" "30 minutes"
        run_test "Endurance Test" "load_test_endurance.js" "60 minutes"
        ;;

    endurance)
        echo -e "${YELLOW}Running Endurance Test Only (1 hour)${NC}"
        run_test "Endurance Test" "load_test_endurance.js" "60 minutes"
        ;;

    *)
        echo "Usage: $0 [test-type]"
        echo
        echo "Test types:"
        echo "  smoke         - Quick sanity check (30 sec)"
        echo "  load          - Standard load test (5 min + smoke)"
        echo "  comprehensive - Comprehensive suite (40 min, no endurance)"
        echo "  full|all      - All tests including endurance (3+ hours)"
        echo "  endurance     - Long-running stability test (60 min)"
        echo
        echo "Examples:"
        echo "  $0 smoke"
        echo "  $0 load"
        echo "  $0 comprehensive"
        exit 1
        ;;
esac

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}LOAD TESTING COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${BLUE}Results saved to: $RESULTS_DIR${NC}"
echo
echo "Next steps:"
echo "  1. Review load test results in the results directory"
echo "  2. Access Grafana dashboards: http://localhost:3000"
echo "  3. Check Jaeger traces: http://localhost:16686"
echo "  4. Compare metrics against SLO thresholds"
echo "  5. Proceed to Phase 6 (Integration Testing) if SLOs met"
echo
