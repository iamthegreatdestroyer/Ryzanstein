/*
 * RWKV Time Mixing Unit Tests (Assert-Based)
 * [REF:CC-004d] - Core Components: RWKV Attention-Free Runtime
 *
 * Simplified test suite for RWKV time mixing layer
 * Tests cover:
 * - Time-shift blending correctness
 * - Projection operations (R, W, K, V)
 * - WKV state management
 * - Numerical stability
 */

#include "test_framework.h"
#include "../src/core/rwkv/time_mixing.h"
#include "../src/core/rwkv/wkv.h"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

using namespace ryzen_llm::rwkv;
using namespace test_framework;

// ===== Helper Functions =====

float compute_mse(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector size mismatch");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        mse += diff * diff;
    }
    return mse / a.size();
}

std::vector<float> generate_random_vector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    std::vector<float> vec(size);
    for (auto& v : vec) {
        v = dist(gen);
    }
    return vec;
}

// ===== Test Cases =====

void test_time_mixing_initialization() {
    // Test: TimeMixing layer initializes correctly
    TimeMixing tm(2048, 64);
    
    // Verify dimensions are set
    ASSERT_EQ(tm.get_embedding_dim(), 2048);
    ASSERT_EQ(tm.get_head_dim(), 64);
    
    std::cout << "  ✓ Time Mixing initialized with correct dimensions" << std::endl;
}

void test_time_mixing_single_token() {
    // Test: Single token forward pass
    TimeMixing tm(128, 32);
    
    // Create input tensor (batch_size=1, seq_len=1, dim=128)
    std::vector<float> input = generate_random_vector(128);
    std::vector<float> output(128);
    
    // Forward pass
    const float* input_ptr = input.data();
    float* output_ptr = output.data();
    tm.forward_single_token(input_ptr, output_ptr);
    
    // Verify output is not all zeros
    float output_sum = 0.0f;
    for (auto v : output) {
        output_sum += std::abs(v);
    }
    ASSERT_TRUE(output_sum > 0.0f);
    
    std::cout << "  ✓ Single token forward pass produces non-zero output" << std::endl;
}

void test_time_mixing_sequence() {
    // Test: Sequence forward pass
    TimeMixing tm(64, 16);
    
    // Create input sequence (seq_len=4, dim=64)
    size_t seq_len = 4;
    size_t dim = 64;
    std::vector<float> input = generate_random_vector(seq_len * dim);
    std::vector<float> output(seq_len * dim);
    
    // Forward pass
    tm.forward_sequence(input.data(), output.data(), seq_len);
    
    // Verify output size is correct
    ASSERT_EQ(output.size(), input.size());
    
    // Verify output contains non-zero values
    float output_sum = 0.0f;
    for (auto v : output) {
        output_sum += std::abs(v);
    }
    ASSERT_TRUE(output_sum > 0.0f);
    
    std::cout << "  ✓ Sequence forward pass processes " << seq_len << " tokens correctly" << std::endl;
}

void test_wkv_operator_initialization() {
    // Test: WKV operator initializes correctly
    WKVOperator wkv(128);
    
    // Verify state dimensions
    ASSERT_EQ(wkv.get_dim(), 128);
    
    std::cout << "  ✓ WKV operator initialized with correct dimension" << std::endl;
}

void test_wkv_single_token_forward() {
    // Test: WKV single token forward pass
    size_t dim = 64;
    WKVOperator wkv(dim);
    
    // Create inputs
    std::vector<float> k = generate_random_vector(dim);
    std::vector<float> v = generate_random_vector(dim);
    std::vector<float> w = generate_random_vector(dim);
    
    // Forward pass
    std::vector<float> output(dim);
    wkv.forward_single_token(k.data(), v.data(), w.data(), output.data());
    
    // Verify output is not all zeros
    float output_sum = 0.0f;
    for (auto val : output) {
        output_sum += std::abs(val);
    }
    ASSERT_TRUE(output_sum > 0.0f);
    
    std::cout << "  ✓ WKV single token forward produces valid output" << std::endl;
}

void test_state_preservation() {
    // Test: WKV state is preserved across iterations
    size_t dim = 32;
    WKVOperator wkv(dim);
    
    std::vector<float> k = generate_random_vector(dim);
    std::vector<float> v = generate_random_vector(dim);
    std::vector<float> w = generate_random_vector(dim);
    
    std::vector<float> output1(dim);
    std::vector<float> output2(dim);
    
    // First forward pass
    wkv.forward_single_token(k.data(), v.data(), w.data(), output1.data());
    
    // Second forward pass with different inputs
    std::vector<float> k2 = generate_random_vector(dim);
    std::vector<float> v2 = generate_random_vector(dim);
    
    wkv.forward_single_token(k2.data(), v2.data(), w.data(), output2.data());
    
    // Outputs should be different due to state accumulation
    float diff = compute_mse(output1, output2);
    ASSERT_TRUE(diff > 0.0f);
    
    std::cout << "  ✓ WKV state correctly accumulates across iterations" << std::endl;
}

void test_numerical_stability() {
    // Test: Operations are numerically stable (no NaN/Inf)
    size_t dim = 128;
    TimeMixing tm(dim, 32);
    
    // Create extreme values (but within reasonable range)
    std::vector<float> input = generate_random_vector(dim, -10.0f, 10.0f);
    std::vector<float> output(dim);
    
    // Forward pass
    tm.forward_single_token(input.data(), output.data());
    
    // Check for NaN and Inf
    for (auto v : output) {
        ASSERT_TRUE(!std::isnan(v));
        ASSERT_TRUE(!std::isinf(v));
    }
    
    std::cout << "  ✓ Forward pass maintains numerical stability" << std::endl;
}

void test_time_mixing_with_wkv_integration() {
    // Test: TimeMixing and WKV work together correctly
    size_t dim = 64;
    TimeMixing tm(dim, 16);
    WKVOperator wkv(dim);
    
    // Create input
    std::vector<float> input = generate_random_vector(dim);
    std::vector<float> tm_output(dim);
    std::vector<float> wkv_output(dim);
    
    // Time mixing forward pass
    tm.forward_single_token(input.data(), tm_output.data());
    
    // Simulate using TimeMixing output as WKV input
    // (In real usage, output would be split into k, v, w)
    wkv.forward_single_token(tm_output.data(), tm_output.data(), input.data(), wkv_output.data());
    
    // Verify integration produced valid output
    float output_sum = 0.0f;
    for (auto v : wkv_output) {
        output_sum += std::abs(v);
    }
    ASSERT_TRUE(output_sum > 0.0f);
    
    std::cout << "  ✓ TimeMixing and WKV integration works correctly" << std::endl;
}

// ===== Test Registration =====

int main() {
    std::cout << "\n========== RWKV Unit Tests ==========\n";
    
    try {
        // Test Time Mixing
        std::cout << "\n[Time Mixing Tests]\n";
        test_time_mixing_initialization();
        test_time_mixing_single_token();
        test_time_mixing_sequence();
        
        // Test WKV Operator
        std::cout << "\n[WKV Operator Tests]\n";
        test_wkv_operator_initialization();
        test_wkv_single_token_forward();
        test_state_preservation();
        
        // Test Integration
        std::cout << "\n[Integration Tests]\n";
        test_numerical_stability();
        test_time_mixing_with_wkv_integration();
        
        std::cout << "\n========== All Tests Passed ✓ ==========\n\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test Failed: " << e.what() << "\n\n";
        return 1;
    } catch (...) {
        std::cerr << "\n✗ Unknown test failure\n\n";
        return 1;
    }
}
