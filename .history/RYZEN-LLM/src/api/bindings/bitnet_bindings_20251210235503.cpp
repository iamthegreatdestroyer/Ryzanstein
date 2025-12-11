#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "bitnet/quantize.h"

// C interface for ctypes
extern "C" {
    __declspec(dllexport) int test_function() {
        return 42;
    }

    // Test function that returns a constant
    __declspec(dllexport) int test_quantize_scalar() {
        return 12345;
    }

    // Test function that just creates a TernaryWeight object
    __declspec(dllexport) int test_simple_loop() {
        try {
            // Test fabs function
            float weights[16] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16};
            float sum = 0.0f;
            for (int i = 0; i < 16; ++i) {
                sum += fabs(weights[i]); // Use fabs instead of std::abs
            }
            return (int)sum; // Should return 136
        } catch (...) {
            return -1;
        }
    }

    // Test nested loops like in quantization
    __declspec(dllexport) int test_nested_loops() {
        try {
            // Test nested loops for element counting
            size_t rows = 4;
            size_t cols = 4;
            size_t total_size = 0;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    total_size++;
                }
            }
            return (int)total_size; // Should return 16
        } catch (...) {
            return -1;
        }
    }

    __declspec(dllexport) int test_weight_quantize_only() {
        try {
            
            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            
            ryzen_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;
            
            auto ternary_weight = ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);
            
            return 1;
        } catch (const std::exception& e) {
            return -1;
        } catch (...) {
            return -2;
        }
    }
    
    __declspec(dllexport) int test_activation_quantize_only() {
        try {
            
            // Create test data
            float activations[4] = {1, -2, 3, -4};
            
            ryzen_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;
            
            auto quantized_activation = ryzen_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);
            
            return 1;
        } catch (const std::exception& e) {
            return -1;
        } catch (...) {
            return -2;
        }
    }

    __declspec(dllexport) int test_scalar_quantize_direct() {
        try {
            
            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            float activations[4] = {1, -2, 3, -4};
            
            ryzen_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;
            
            auto ternary_weight = ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);
            
            auto quantized_activation = ryzen_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);
            
            return 1;
        } catch (const std::exception& e) {
            return -1;
        } catch (...) {
            return -2;
        }
    }

    // Test function that calls quantize_weights_ternary_scalar without creating objects
    __declspec(dllexport) int test_quantize_weights_only_scalar() {
        try {
            std::cout << "DEBUG: test_quantize_weights_only_scalar called" << std::endl;
            // Create dummy data
            float weights[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
            ryzen_llm::bitnet::QuantConfig config;
            
            // Call the function but don't use the result
            ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);
            
            std::cout << "DEBUG: quantize_weights_ternary_scalar completed successfully" << std::endl;
            return 1; // Success
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception in quantize_weights_ternary_scalar: " << e.what() << std::endl;
            return -1; // Failure
        } catch (...) {
            std::cout << "DEBUG: Unknown exception in quantize_weights_ternary_scalar" << std::endl;
            return -2; // Failure
        }
    }

    // Test function that replicates test_basic_quantize_ops but with larger data
    __declspec(dllexport) int test_basic_quantize_large() {
        try {
            // Same data as test_quantize_no_vector
            const int size = 16;
            float weights[size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                  9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
            float mean_abs = 0.0f;
            
            // Compute mean absolute value
            for (int i = 0; i < size; ++i) {
                mean_abs += std::abs(weights[i]);
            }
            mean_abs /= size;
            
            // Threshold
            const float threshold = 0.7f * mean_abs;
            
            // Quantize
            int8_t result[size];
            for (int i = 0; i < size; ++i) {
                const float w = weights[i];
                if (std::abs(w) > threshold) {
                    result[i] = (w > 0) ? 1 : -1;
                } else {
                    result[i] = 0;
                }
            }
            
            return 1; // Success
        } catch (const std::exception& e) {
            return -1; // Failure
        } catch (...) {
            return -2; // Failure
        }
    }

    // Quantization functions
    __declspec(dllexport) int test_quantize_weights_only(const float* weights, uint32_t rows, uint32_t cols) {
        std::cout << "DEBUG: test_quantize_weights_only called" << std::endl;
        // Just do some basic computation without creating objects
        float sum = 0.0f;
        for (uint32_t i = 0; i < rows * cols; ++i) {
            sum += weights[i];
        }
        return static_cast<int>(sum * 1000);  // Return scaled sum
    }
    __declspec(dllexport) void* quantize_weights_ternary_c(
        const float* weights, uint32_t rows, uint32_t cols) {
        ryzen_llm::bitnet::QuantConfig config;
        ryzen_llm::bitnet::TernaryWeight result =
            ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, rows, cols, config);
        ryzen_llm::bitnet::TernaryWeight* result_ptr =
            new ryzen_llm::bitnet::TernaryWeight(result);
        return result_ptr;
    }

    __declspec(dllexport) void* quantize_activations_int8_c(
        const float* activations, size_t size) {
        ryzen_llm::bitnet::QuantConfig config;
        ryzen_llm::bitnet::QuantizedActivation* result =
            new ryzen_llm::bitnet::QuantizedActivation(
                ryzen_llm::bitnet::quantize_activations_int8_scalar(activations, size, config));
        return result;
    }

    __declspec(dllexport) void dequantize_weights_c(
        void* ternary_weight_ptr, float* output) {
        auto* ternary_weight = static_cast<ryzen_llm::bitnet::TernaryWeight*>(ternary_weight_ptr);
        ryzen_llm::bitnet::dequantize_weights_scalar(*ternary_weight, output, ternary_weight->rows, ternary_weight->cols);
    }

    __declspec(dllexport) void dequantize_activations_c(
        void* quantized_ptr, float* output) {
        auto* quantized = static_cast<ryzen_llm::bitnet::QuantizedActivation*>(quantized_ptr);
        ryzen_llm::bitnet::dequantize_activations_scalar(*quantized, output, quantized->values.size());
    }

    __declspec(dllexport) float compute_quantization_error_c(
        const float* original, const float* quantized, size_t size) {
        return ryzen_llm::bitnet::compute_quantization_error_scalar(original, quantized, size);
    }

    // Memory management
    __declspec(dllexport) void free_ternary_weight(void* ptr) {
        delete static_cast<ryzen_llm::bitnet::TernaryWeight*>(ptr);
    }

    __declspec(dllexport) void free_quantized_activation(void* ptr) {
        delete static_cast<ryzen_llm::bitnet::QuantizedActivation*>(ptr);
    }

    // Test floating-point accumulation operations
    __declspec(dllexport) int test_floating_point_accumulation() {
        // Test basic floating-point operations
        float a = 1.0f;
        float b = 2.0f;
        float sum = a + b;
        
        // Test std::abs
        float val = -3.5f;
        float abs_val = std::abs(val);
        
        // Test accumulation loop (this is where it fails)
        float sum_abs = 0.0f;
        float weights[] = {1.0f, -2.0f, 3.0f, -4.0f};
        int num_weights = 4;
        
        for (int i = 0; i < num_weights; ++i) {
            float abs_weight = std::abs(weights[i]);
            sum_abs += abs_weight;  // This line likely triggers the error
        }
        
        return static_cast<int>(sum_abs * 1000);  // Return scaled integer
    }

    // Test division operations
    __declspec(dllexport) int test_division_operations() {
        // Test basic division
        float numerator = 10.0f;
        float denominator = 4.0f;
        float result = numerator / denominator;
        
        // Test division by zero protection
        float safe_div = (denominator > 0) ? (numerator / denominator) : 1.0f;
        
        // Test the exact division from quantization
        float sum_abs_original = 10.0f;
        float sum_abs_quantized = 5.0f;
        float scale = (sum_abs_quantized > 0) ? (sum_abs_original / sum_abs_quantized) : 1.0f;
        
        // Test mean calculation
        float total = 0.0f;
        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        int count = 4;
        for (int i = 0; i < count; ++i) {
            total += values[i];
        }
        float mean = total / count;
        
        return static_cast<int>(scale * 1000);
    }

    // Test std::vector operations
    __declspec(dllexport) int test_vector_operations() {
        // Test basic vector creation
        std::vector<int8_t> values;
        
        // Test resize operation (this might trigger AVX-512)
        size_t size = 16;
        values.resize(size);
        
        // Test filling vector
        for (size_t i = 0; i < size; ++i) {
            values[i] = static_cast<int8_t>(i % 3 - 1); // -1, 0, 1 pattern
        }
        
        // Test float vector
        std::vector<float> scales;
        scales.resize(1);
        scales[0] = 2.5f;
        
        // Test larger resize
        size_t large_size = 64;
        values.resize(large_size);
        
        // Test multiple groups resize (like TernaryWeight)
        uint32_t r = 4, c = 4, gs = 8;
        uint32_t total_size = r * c;
        uint32_t num_groups = (gs > 0) ? ((total_size + gs - 1) / gs) : 1;
        
        std::vector<int8_t> test_values;
        std::vector<float> test_scales;
        
        test_values.resize(total_size);
        test_scales.resize(num_groups);
        
        return static_cast<int>(values.size() + scales.size() + test_values.size() + test_scales.size());
    }

    // Test avoiding std::min_element and std::max_element
    __declspec(dllexport) int test_min_max_avoidance() {
        // Test data - same as used in quantize_activations_int8_scalar
        float activations[4] = {1.0f, -2.0f, 3.0f, -4.0f};
        size_t size = 4;
        
        // Manual min/max calculation instead of std::min_element/max_element
        float min_val = activations[0];
        float max_val = activations[0];
        
        for (size_t i = 1; i < size; ++i) {
            if (activations[i] < min_val) min_val = activations[i];
            if (activations[i] > max_val) max_val = activations[i];
        }
        
        // Continue with quantization logic (simplified)
        float abs_max = (fabs(min_val) > fabs(max_val)) ? fabs(min_val) : fabs(max_val);
        float scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
        
        // Test quantization
        int8_t quantized[4];
        for (size_t i = 0; i < size; ++i) {
            float clamped = (activations[i] < -abs_max) ? -abs_max : 
                           ((activations[i] > abs_max) ? abs_max : activations[i]);
            quantized[i] = static_cast<int8_t>(round(clamped * scale));
        }
        
        return 42; // Success
    }

    __declspec(dllexport) int test_int8_vector_operations() {
        try {
            // Test std::vector<int8_t> operations
            std::vector<int8_t> values;
            
            values.resize(16);
            
            // Fill with test data
            for (size_t i = 0; i < 16; ++i) {
                values[i] = static_cast<int8_t>(i - 8); // -8 to 7
            }
            
            // Test reading back
            int sum = 0;
            for (size_t i = 0; i < 16; ++i) {
                sum += values[i];
            }
            
            return sum;
        } catch (const std::exception& e) {
            return -1;
        } catch (...) {
            return -2;
        }
    }
}

PYBIND11_MODULE(test_module, m)
{
    try {
        m.def("test_function", []() { return 42; });
    } catch (const std::exception& e) {
    } catch (...) {
    }
}