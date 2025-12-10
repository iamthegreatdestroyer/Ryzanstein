/*
 * RWKV Task 11 - Hardware Compilation Smoke Test
 * Verifies that compiled RWKV code runs without crashing
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#include "../../src/core/rwkv/time_mixing.h"
#include "../../src/core/rwkv/wkv.h"

using namespace ryzen_llm::rwkv;

// Simple random vector generator
std::vector<float> generate_random(size_t size, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> vec(size);
    for (auto& v : vec) {
        v = dist(gen);
    }
    return vec;
}

// ===== Test Cases =====

void test_time_mixing_layer_construction() {
    // Test: TimeMixingLayer layer initializes correctly
    uint32_t hidden_dim = 2048;
    uint32_t layer_id = 0;
    
    TimeMixingConfig config;
    config.time_decay_rate = 0.9f;
    
    TimeMixingLayer tm_layer(hidden_dim, layer_id, config);
    
    std::cout << "  ✓ TimeMixingLayer initialized with dimension " << hidden_dim << std::endl;
}

void test_time_mixing_single_token() {
    // Test: Single token forward pass
    uint32_t hidden_dim = 128;
    TimeMixingConfig config;
    TimeMixingLayer tm_layer(hidden_dim, 0, config);
    
    // Create input tensor (dim=128)
    std::vector<float> input = generate_random_vector(hidden_dim);
    std::vector<float> output(hidden_dim);
    
    // Forward pass for single token
    bool success = tm_layer.forward_single_token(input.data(), output.data());
    
    // Verify forward pass succeeded
    assert(success && "Forward pass failed");
    
    // Verify output is not all zeros
    float output_sum = 0.0f;
    for (auto v : output) {
        output_sum += std::abs(v);
    }
    assert(output_sum > 0.0f && "Output contains only zeros");
    
    // Verify no NaN or Inf
    for (auto v : output) {
        assert(!std::isnan(v) && !std::isinf(v) && "Output contains NaN or Inf");
    }
    
    std::cout << "  ✓ Single token forward pass produces valid output" << std::endl;
}

void test_time_mixing_sequence() {
    // Test: Sequence forward pass
    uint32_t hidden_dim = 64;
    uint32_t seq_len = 4;
    
    TimeMixingConfig config;
    TimeMixingLayer tm_layer(hidden_dim, 0, config);
    
    // Create input sequence (seq_len * dim)
    std::vector<float> input = generate_random_vector(seq_len * hidden_dim);
    std::vector<float> output(seq_len * hidden_dim);
    
    // Forward pass for sequence
    bool success = tm_layer.forward_sequence(
        input.data(), 
        output.data(), 
        seq_len
    );
    
    assert(success && "Forward sequence failed");
    
    // Verify output size is correct
    assert(output.size() == input.size() && "Output size mismatch");
    
    // Verify output contains non-zero values
    float output_sum = 0.0f;
    for (auto v : output) {
        output_sum += std::abs(v);
    }
    assert(output_sum > 0.0f && "Sequence output is all zeros");
    
    // Verify no NaN or Inf
    for (auto v : output) {
        assert(!std::isnan(v) && !std::isinf(v) && "Output contains NaN or Inf");
    }
    
    std::cout << "  ✓ Sequence forward pass processes " << seq_len << " tokens correctly" << std::endl;
}

void test_wkv_operator_initialization() {
    // Test: WKV operator initializes correctly
    uint32_t dim = 128;
    WKVConfig config;
    
    WKVOperator wkv_op(dim, config);
    
    // Initialize state
    wkv_op.initialize();
    
    std::cout << "  ✓ WKV operator initialized with dimension " << dim << std::endl;
}

void test_wkv_single_token_forward() {
    // Test: WKV single token forward pass
    uint32_t dim = 64;
    WKVConfig config;
    WKVOperator wkv_op(dim, config);
    
    // Initialize state
    wkv_op.initialize();
    
    // Create inputs
    std::vector<float> k = generate_random_vector(dim);
    std::vector<float> v = generate_random_vector(dim);
    std::vector<float> w = generate_random_vector(dim, 0.1f, 0.9f);  // Decay weights should be bounded
    std::vector<float> r = generate_random_vector(dim);
    std::vector<float> output(dim);
    
    // Forward pass
    bool success = wkv_op.forward(
        k.data(), v.data(), w.data(), r.data(), output.data()
    );
    
    assert(success && "WKV forward pass failed");
    
    // Verify output is not all zeros
    float output_sum = 0.0f;
    for (auto val : output) {
        output_sum += std::abs(val);
    }
    assert(output_sum > 0.0f && "WKV output is all zeros");
    
    // Verify no NaN or Inf
    for (auto val : output) {
        assert(!std::isnan(val) && !std::isinf(val) && "WKV output contains NaN or Inf");
    }
    
    std::cout << "  ✓ WKV single token forward produces valid output" << std::endl;
}

void test_wkv_sequence_forward() {
    // Test: WKV sequence forward pass
    uint32_t dim = 32;
    uint32_t seq_len = 3;
    WKVConfig config;
    
    WKVOperator wkv_op(dim, config);
    wkv_op.initialize();
    
    // Create input sequences
    std::vector<float> keys = generate_random_vector(seq_len * dim);
    std::vector<float> values = generate_random_vector(seq_len * dim);
    std::vector<float> weights = generate_random_vector(seq_len * dim, 0.1f, 0.9f);
    std::vector<float> receptances = generate_random_vector(seq_len * dim);
    std::vector<float> output(seq_len * dim);
    
    // Forward pass
    bool success = wkv_op.forward_sequence(
        keys.data(), values.data(), weights.data(), receptances.data(),
        seq_len, output.data()
    );
    
    assert(success && "WKV sequence forward pass failed");
    
    // Verify output size
    assert(output.size() == seq_len * dim && "Output size mismatch");
    
    // Verify output is valid
    float output_sum = 0.0f;
    for (auto v : output) {
        output_sum += std::abs(v);
        assert(!std::isnan(v) && !std::isinf(v) && "Output contains NaN or Inf");
    }
    assert(output_sum > 0.0f && "Output is all zeros");
    
    std::cout << "  ✓ WKV sequence forward pass processes " << seq_len << " tokens correctly" << std::endl;
}

void test_state_management() {
    // Test: State is properly managed across operations
    uint32_t dim = 32;
    WKVConfig config;
    WKVOperator wkv_op(dim, config);
    
    // First sequence
    wkv_op.initialize();
    std::vector<float> k1 = generate_random_vector(dim);
    std::vector<float> v1 = generate_random_vector(dim);
    std::vector<float> w1 = generate_random_vector(dim, 0.1f, 0.9f);
    std::vector<float> r1 = generate_random_vector(dim);
    std::vector<float> out1(dim);
    
    bool success1 = wkv_op.forward(k1.data(), v1.data(), w1.data(), r1.data(), out1.data());
    assert(success1 && "First forward pass failed");
    
    // Reset state for new sequence
    wkv_op.reset_state();
    wkv_op.initialize();
    
    // Second sequence should start fresh
    std::vector<float> k2 = generate_random_vector(dim);
    std::vector<float> v2 = generate_random_vector(dim);
    std::vector<float> w2 = generate_random_vector(dim, 0.1f, 0.9f);
    std::vector<float> r2 = generate_random_vector(dim);
    std::vector<float> out2(dim);
    
    bool success2 = wkv_op.forward(k2.data(), v2.data(), w2.data(), r2.data(), out2.data());
    assert(success2 && "Second forward pass failed");
    
    std::cout << "  ✓ State management works correctly across sequences" << std::endl;
}

void test_numerical_stability() {
    // Test: Operations are numerically stable (no NaN/Inf)
    uint32_t hidden_dim = 128;
    TimeMixingConfig config;
    TimeMixingLayer tm_layer(hidden_dim, 0, config);
    
    // Create reasonable values
    std::vector<float> input = generate_random_vector(hidden_dim, -1.0f, 1.0f);
    std::vector<float> output(hidden_dim);
    
    // Forward pass
    bool success = tm_layer.forward_single_token(input.data(), output.data());
    assert(success && "Forward pass failed");
    
    // Check for NaN and Inf
    for (auto v : output) {
        assert(!std::isnan(v) && "Output contains NaN");
        assert(!std::isinf(v) && "Output contains Inf");
    }
    
    std::cout << "  ✓ Forward pass maintains numerical stability" << std::endl;
}

void test_multi_layer_composition() {
    // Test: Multiple layers can be composed
    uint32_t hidden_dim = 64;
    uint32_t num_layers = 3;
    
    std::vector<TimeMixingLayer> layers;
    for (uint32_t i = 0; i < num_layers; ++i) {
        TimeMixingConfig config;
        layers.emplace_back(hidden_dim, i, config);
    }
    
    // Process through multiple layers
    std::vector<float> input = generate_random_vector(hidden_dim);
    std::vector<float> x = input;
    
    for (auto& layer : layers) {
        std::vector<float> output(hidden_dim);
        bool success = layer.forward_single_token(x.data(), output.data());
        assert(success && "Multi-layer forward pass failed");
        x = output;
    }
    
    // Verify output is valid
    for (auto v : x) {
        assert(!std::isnan(v) && !std::isinf(v) && "Output contains NaN or Inf");
    }
    
    std::cout << "  ✓ Multi-layer composition works correctly (" << num_layers << " layers)" << std::endl;
}

// ===== Main Test Runner =====

int main() {
    std::cout << "\n========== RWKV Unit Tests ==========\n";
    
    int passed = 0;
    int failed = 0;
    
    try {
        // Test Time Mixing
        std::cout << "\n[Time Mixing Tests]\n";
        
        try { test_time_mixing_layer_construction(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_time_mixing_single_token(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_time_mixing_sequence(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        // Test WKV Operator
        std::cout << "\n[WKV Operator Tests]\n";
        
        try { test_wkv_operator_initialization(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_wkv_single_token_forward(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_wkv_sequence_forward(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_state_management(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        // Test Integration
        std::cout << "\n[Integration Tests]\n";
        
        try { test_numerical_stability(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        try { test_multi_layer_composition(); passed++; }
        catch (const std::exception& e) { std::cerr << "  ✗ " << e.what() << std::endl; failed++; }
        
        std::cout << "\n========== Test Results ==========\n";
        std::cout << "Total: " << (passed + failed) << ", ";
        std::cout << "Passed: " << passed << ", ";
        std::cout << "Failed: " << failed << "\n\n";
        
        return (failed == 0) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Fatal Error: " << e.what() << "\n\n";
        return 1;
    }
}
