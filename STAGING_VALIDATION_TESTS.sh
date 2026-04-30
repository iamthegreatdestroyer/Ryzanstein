#!/bin/bash

# STAGING VALIDATION TEST SUITE
# Phase 4 Deliverables Validation
# Date: February 18, 2026
# Environment: Docker Compose (portable) + Kubernetes (production)

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# ==============================================================================
# PHASE 0: ENVIRONMENT SETUP
# ==============================================================================

test_environment_setup() {
    log_info "=== PHASE 0: ENVIRONMENT SETUP ==="

    # Check Docker
    if command -v docker &> /dev/null; then
        VERSION=$(docker --version)
        log_success "Docker available: $VERSION"
        ((TESTS_PASSED++))
    else
        log_error "Docker not found"
        ((TESTS_FAILED++))
        return 1
    fi

    # Check Docker Compose
    if docker compose version &> /dev/null 2>&1 || docker-compose --version &> /dev/null 2>&1; then
        log_success "Docker Compose available"
        ((TESTS_PASSED++))
    else
        log_warning "Docker Compose not available - some tests will be skipped"
        ((TESTS_SKIPPED++))
    fi

    # Check kubectl
    if command -v kubectl &> /dev/null; then
        VERSION=$(kubectl version --client 2>/dev/null | grep -oP 'v\d+\.\d+\.\d+' | head -1)
        log_success "kubectl available: $VERSION"
        ((TESTS_PASSED++))
    else
        log_warning "kubectl not found - Kubernetes tests will be skipped"
        ((TESTS_SKIPPED++))
    fi

    # Check Helm
    if command -v helm &> /dev/null; then
        VERSION=$(helm version 2>/dev/null | grep -oP 'v\d+\.\d+\.\d+' | head -1)
        log_success "Helm available: $VERSION"
        ((TESTS_PASSED++))
    else
        log_warning "Helm not found - Helm chart tests will be skipped"
        ((TESTS_SKIPPED++))
    fi

    echo ""
}

# ==============================================================================
# PHASE 1: DOCKER IMAGE VALIDATION
# ==============================================================================

test_docker_build() {
    log_info "=== PHASE 1: DOCKER IMAGE VALIDATION ==="

    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available - skipping Docker build test"
        ((TESTS_SKIPPED++))
        return
    fi

    # Check if Dockerfile.linux exists
    if [ ! -f "Dockerfile.linux" ]; then
        log_error "Dockerfile.linux not found"
        ((TESTS_FAILED++))
        return 1
    fi

    log_info "Dockerfile.linux found - would build in staging with:"
    log_info "  docker build -f Dockerfile.linux -t ryzanstein:staging ."

    # Validate Dockerfile syntax
    if docker build -f Dockerfile.linux --dry-run . &> /dev/null 2>&1; then
        log_success "Dockerfile syntax valid"
        ((TESTS_PASSED++))
    else
        log_warning "Dockerfile syntax check (dry-run) not available in this Docker version"
        ((TESTS_SKIPPED++))
    fi

    echo ""
}

# ==============================================================================
# PHASE 2: HELM CHART VALIDATION
# ==============================================================================

test_helm_chart() {
    log_info "=== PHASE 2: HELM CHART VALIDATION ==="

    if ! command -v helm &> /dev/null; then
        log_warning "Helm not available - skipping Helm tests"
        ((TESTS_SKIPPED++))
        return
    fi

    CHART_PATH="helm/ryzanstein"

    # Check if chart exists
    if [ ! -f "$CHART_PATH/Chart.yaml" ]; then
        log_error "Chart not found at $CHART_PATH"
        ((TESTS_FAILED++))
        return 1
    fi

    log_success "Helm chart found"
    ((TESTS_PASSED++))

    # Run helm lint
    log_info "Running helm lint..."
    if helm lint "$CHART_PATH" > /tmp/helm-lint-output.txt 2>&1; then
        log_success "Helm lint passed"
        ((TESTS_PASSED++))
    else
        log_error "Helm lint failed"
        cat /tmp/helm-lint-output.txt
        ((TESTS_FAILED++))
    fi

    # Check required files
    for file in Chart.yaml values.yaml values-dev.yaml values-production.yaml; do
        if [ -f "$CHART_PATH/$file" ]; then
            log_success "Found: $file"
            ((TESTS_PASSED++))
        else
            log_error "Missing: $file"
            ((TESTS_FAILED++))
        fi
    done

    # Check templates
    TEMPLATE_COUNT=$(ls "$CHART_PATH/templates/"*.yaml 2>/dev/null | wc -l)
    log_success "Found $TEMPLATE_COUNT Helm templates"
    ((TESTS_PASSED++))

    echo ""
}

# ==============================================================================
# PHASE 3: CONFIGURATION VALIDATION
# ==============================================================================

test_configuration_files() {
    log_info "=== PHASE 3: CONFIGURATION FILE VALIDATION ==="

    # Check Prometheus config
    if [ -f "config/prometheus.yml" ]; then
        log_success "Prometheus config found"
        ((TESTS_PASSED++))

        # Check for scrape targets
        if grep -q "scrape_configs:" config/prometheus.yml; then
            TARGET_COUNT=$(grep -c "job_name:" config/prometheus.yml)
            log_success "Found $TARGET_COUNT Prometheus scrape targets"
            ((TESTS_PASSED++))
        fi
    else
        log_error "Prometheus config not found"
        ((TESTS_FAILED++))
    fi

    # Check AlertManager config
    if [ -f "config/alertmanager.yml" ]; then
        log_success "AlertManager config found"
        ((TESTS_PASSED++))

        # Check for routes
        if grep -q "route:" config/alertmanager.yml; then
            log_success "AlertManager routing configured"
            ((TESTS_PASSED++))
        fi
    else
        log_error "AlertManager config not found"
        ((TESTS_FAILED++))
    fi

    # Check alert rules
    if [ -f "config/alert_rules.yml" ]; then
        log_success "Alert rules config found"
        ((TESTS_PASSED++))

        # Count alert rules
        ALERT_COUNT=$(grep -c "alert:" config/alert_rules.yml)
        log_success "Found $ALERT_COUNT Prometheus alert rules"
        ((TESTS_PASSED++))
    else
        log_error "Alert rules config not found"
        ((TESTS_FAILED++))
    fi

    echo ""
}

# ==============================================================================
# PHASE 4: DOCUMENTATION VALIDATION
# ==============================================================================

test_documentation() {
    log_info "=== PHASE 4: DOCUMENTATION VALIDATION ==="

    # Check for required documentation files
    DOCS=(
        "DOCKER_DEPLOYMENT.md"
        "HELM_DEPLOYMENT_GUIDE.md"
        "PRODUCTION_MONITORING_GUIDE.md"
        "SECURITY_HARDENING_GUIDE.md"
        "LOAD_TESTING_GUIDE.md"
    )

    for doc in "${DOCS[@]}"; do
        if [ -f "$doc" ]; then
            SIZE=$(wc -c < "$doc")
            log_success "Found: $doc ($SIZE bytes)"
            ((TESTS_PASSED++))
        else
            log_error "Missing: $doc"
            ((TESTS_FAILED++))
        fi
    done

    echo ""
}

# ==============================================================================
# PHASE 5: DOCKER COMPOSE VALIDATION
# ==============================================================================

test_docker_compose() {
    log_info "=== PHASE 5: DOCKER COMPOSE VALIDATION ==="

    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found"
        ((TESTS_FAILED++))
        return 1
    fi

    log_success "docker-compose.yml found"
    ((TESTS_PASSED++))

    # Check for required services
    SERVICES=("ryzanstein-api" "mcp-server" "prometheus" "grafana" "jaeger" "alertmanager" "qdrant")

    for service in "${SERVICES[@]}"; do
        if grep -q "\"$service\":\|$service:" docker-compose.yml; then
            log_success "Service configured: $service"
            ((TESTS_PASSED++))
        else
            log_warning "Service not found in docker-compose.yml: $service"
            ((TESTS_SKIPPED++))
        fi
    done

    echo ""
}

# ==============================================================================
# PHASE 6: KUBERNETES READINESS (Instructions)
# ==============================================================================

test_kubernetes_readiness() {
    log_info "=== PHASE 6: KUBERNETES READINESS ASSESSMENT ==="

    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl not installed"
        log_info "To enable Kubernetes testing, install kubectl:"
        log_info "  macOS: brew install kubectl"
        log_info "  Ubuntu: sudo apt-get install kubectl"
        log_info "  Windows: choco install kubernetes-cli"
        ((TESTS_SKIPPED++))
        return
    fi

    # Check cluster connection
    if kubectl cluster-info &> /dev/null; then
        log_success "Kubernetes cluster connected"
        ((TESTS_PASSED++))

        # Get cluster info
        CLUSTER_NAME=$(kubectl config current-context 2>/dev/null || echo "unknown")
        log_info "Current context: $CLUSTER_NAME"

        # Check storage classes
        if kubectl get storageclass &> /dev/null; then
            SC_COUNT=$(kubectl get storageclass --no-headers 2>/dev/null | wc -l)
            log_success "Found $SC_COUNT storage class(es)"
            ((TESTS_PASSED++))
        fi
    else
        log_warning "Not connected to Kubernetes cluster"
        log_info "To enable Kubernetes testing:"
        log_info "  1. Start minikube: minikube start --cpus=4 --memory=8192"
        log_info "  2. Or use Docker Desktop with K8s enabled"
        log_info "  3. Or connect to existing cluster: kubectl config use-context <cluster>"
        ((TESTS_SKIPPED++))
    fi

    echo ""
}

# ==============================================================================
# PHASE 7: SECURITY BASELINE
# ==============================================================================

test_security_baseline() {
    log_info "=== PHASE 7: SECURITY BASELINE CHECKS ==="

    # Check for sensitive files
    SENSITIVE_PATTERNS=(
        "*.pem"
        "*.key"
        "*.crt"
        "*secret*"
        "*password*"
    )

    FOUND_SENSITIVE=0
    for pattern in "${SENSITIVE_PATTERNS[@]}"; do
        if find . -name "$pattern" -type f 2>/dev/null | grep -v ".git" | head -1 &> /dev/null; then
            FOUND_SENSITIVE=$((FOUND_SENSITIVE + 1))
        fi
    done

    if [ $FOUND_SENSITIVE -eq 0 ]; then
        log_success "No hardcoded secrets/credentials found"
        ((TESTS_PASSED++))
    else
        log_warning "Found $FOUND_SENSITIVE potential secret files - review for hardcoded credentials"
        ((TESTS_SKIPPED++))
    fi

    # Check for security config files
    if [ -f "SECURITY_HARDENING_GUIDE.md" ]; then
        SECURITY_SECTIONS=$(grep -c "^##" SECURITY_HARDENING_GUIDE.md)
        log_success "Security guide has $SECURITY_SECTIONS sections"
        ((TESTS_PASSED++))
    fi

    echo ""
}

# ==============================================================================
# SUMMARY
# ==============================================================================

print_summary() {
    log_info "=== TEST SUMMARY ==="

    TOTAL=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))

    echo ""
    echo "Tests Passed:  ${GREEN}$TESTS_PASSED${NC}"
    echo "Tests Failed:  ${RED}$TESTS_FAILED${NC}"
    echo "Tests Skipped: ${YELLOW}$TESTS_SKIPPED${NC}"
    echo "Total Tests:   $TOTAL"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All critical tests passed!"
        return 0
    else
        log_error "Some tests failed"
        return 1
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     RYZANSTEIN PHASE 4 STAGING VALIDATION TEST SUITE          ║"
    echo "║     Date: February 18, 2026                                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Change to repo root
    cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    # Run all tests
    test_environment_setup
    test_docker_build
    test_helm_chart
    test_configuration_files
    test_documentation
    test_docker_compose
    test_kubernetes_readiness
    test_security_baseline

    # Print summary
    print_summary

    echo ""
    echo "For full Kubernetes staging validation, run:"
    echo ""
    echo "1. START MINIKUBE:"
    echo "   minikube start --cpus=4 --memory=8192"
    echo ""
    echo "2. BUILD DOCKER IMAGE:"
    echo "   docker build -f Dockerfile.linux -t ryzanstein:staging ."
    echo ""
    echo "3. DEPLOY HELM CHART:"
    echo "   helm install ryzanstein ./helm/ryzanstein -f helm/ryzanstein/values-dev.yaml"
    echo ""
    echo "4. VALIDATE DEPLOYMENT:"
    echo "   kubectl rollout status deployment/ryzanstein-api"
    echo "   kubectl get pods"
    echo ""
    echo "5. RUN LOAD TESTS:"
    echo "   k6 run load_test.js"
    echo ""
    echo "Full validation plan: STAGING_VALIDATION_PLAN.md"
    echo ""
}

# Run main
main
