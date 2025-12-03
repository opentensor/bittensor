#!/bin/bash
# Comprehensive security test runner for Bittensor

set -e

echo "========================================="
echo "Bittensor Security Test Suite"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR"

echo "Test directory: $TEST_DIR"
echo ""

# Function to run tests with specific markers
run_test_suite() {
    local suite_name=$1
    local test_file=$2
    local marker=$3
    
    echo -e "${YELLOW}Running $suite_name...${NC}"
    
    if [ -n "$marker" ]; then
        pytest "$test_file" -v -m "$marker" --tb=short
    else
        pytest "$test_file" -v --tb=short
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $suite_name PASSED${NC}"
    else
        echo -e "${RED}✗ $suite_name FAILED${NC}"
        return 1
    fi
    echo ""
}

# Run all test suites
echo "========================================="
echo "1. Nonce Security Tests"
echo "========================================="
run_test_suite "Nonce Thread Safety" "test_nonce_security.py::TestNonceThreadSafety"
run_test_suite "Nonce Freshness Validation" "test_nonce_security.py::TestNonceFreshnessValidation"

echo "========================================="
echo "2. Race Condition Tests"
echo "========================================="
run_test_suite "Race Condition Prevention" "test_race_conditions.py::TestNonceRaceConditions"

echo "========================================="
echo "3. Transfer Validation Tests"
echo "========================================="
run_test_suite "Transfer Amount Validation" "test_transfer_validation.py::TestTransferAmountValidation"
run_test_suite "Transfer Edge Cases" "test_transfer_validation.py::TestTransferEdgeCases"

echo "========================================="
echo "4. Staking Validation Tests"
echo "========================================="
run_test_suite "Staking Amount Validation" "test_staking_validation.py::TestStakingAmountValidation"
run_test_suite "Staking Edge Cases" "test_staking_validation.py::TestStakingEdgeCases"

echo ""
echo "========================================="
echo "Running Stress Tests (may take longer)"
echo "========================================="
run_test_suite "Nonce Stress Tests" "test_nonce_security.py::TestNonceStressTests"
run_test_suite "Race Condition Stress Test" "test_race_conditions.py::TestNonceRaceConditions::test_stress_test_concurrent_load"

echo ""
echo "========================================="
echo "Test Coverage Report"
echo "========================================="
pytest --cov=bittensor.core.axon \
       --cov=bittensor.core.extrinsics.transfer \
       --cov=bittensor.core.extrinsics.staking \
       --cov-report=term-missing \
       --cov-report=html:coverage_html \
       test_*.py

echo ""
echo "========================================="
echo "Security Test Summary"
echo "========================================="
pytest test_*.py --tb=no --quiet --no-header -v | grep -E "(PASSED|FAILED|ERROR)" | sort | uniq -c

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}All Security Tests Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Coverage report generated in: coverage_html/index.html"
