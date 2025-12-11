#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <memory>
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
    __declspec(dllexport) int test_create_ternary_weight() {
        try {
            // Create a simple TernaryWeight with empty data
            ryzen_llm::bitnet::TernaryWeight* weight = new ryzen_llm::bitnet::TernaryWeight();
            delete weight;
            return 1; // Success
        } catch (const std::exception& e) {
            return -1; // Failure
        } catch (...) {
            return -2; // Failure
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
            
            std::cout << "DEBUG: Calling quantize_weights_ternary_scalar" << std::endl;
            auto ternary_weight = ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);
            std::cout << "DEBUG: Weight quantization completed successfully" << std::endl;
            
            return 1;
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception in test_weight_quantize_only: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cout << "DEBUG: Unknown exception in test_weight_quantize_only" << std::endl;
            return -2;
        }
    }
    
    __declspec(dllexport) int test_activation_quantize_only() {
        try {
            std::cout << "DEBUG: test_activation_quantize_only called" << std::endl;
            
            // Create test data
            float activations[4] = {1, -2, 3, -4};
            
            ryzen_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;
            
            std::cout << "DEBUG: Calling quantize_activations_int8_scalar" << std::endl;
            auto quantized_activation = ryzen_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);
            std::cout << "DEBUG: Activation quantization completed successfully" << std::endl;
            
            return 1;
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception in test_activation_quantize_only: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cout << "DEBUG: Unknown exception in test_activation_quantize_only" << std::endl;
            return -2;
        }
    }

    __declspec(dllexport) int test_scalar_quantize_direct() {
        try {
            std::cout << "DEBUG: test_scalar_quantize_direct called" << std::endl;
            
            // Create test data
            float weights[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            float activations[4] = {1, -2, 3, -4};
            
            ryzen_llm::bitnet::QuantConfig config;
            config.per_group_scaling = false;
            config.weight_group_size = 0;
            config.activation_clip_value = 6.0f;
            config.symmetric_activations = true;
            
            std::cout << "DEBUG: Config created, calling weight quantization" << std::endl;
            auto ternary_weight = ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, 4, 4, config);
            std::cout << "DEBUG: Weight quantization completed successfully" << std::endl;
            
            std::cout << "DEBUG: Calling activation quantization" << std::endl;
            auto quantized_activation = ryzen_llm::bitnet::quantize_activations_int8_scalar(activations, 4, config);
            std::cout << "DEBUG: Activation quantization completed successfully" << std::endl;
            
            return 1;
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception in test_scalar_quantize_direct: " << e.what() << std::endl;
            return -1;
        } catch (...) {
            std::cout << "DEBUG: Unknown exception in test_scalar_quantize_direct" << std::endl;
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
            std::cout << "DEBUG: test_basic_quantize_large called" << std::endl;
            
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
            
            std::cout << "DEBUG: mean_abs computed: " << mean_abs << std::endl;
            
            // Threshold
            const float threshold = 0.7f * mean_abs;
            std::cout << "DEBUG: threshold computed: " << threshold << std::endl;
            
            // Quantize
            int8_t result[size];
            for (int i = 0; i < size; ++i) {
                const float w = weights[i];
                if (std::abs(w) > threshold) {
                    result[i] = (w > 0) ? 1 : -1;
                } else {
                    result[i] = 0;
                }
                std::cout << "DEBUG: weight " << i << ": " << w << " -> " << (int)result[i] << std::endl;
            }
            
            std::cout << "DEBUG: basic quantization large completed successfully" << std::endl;
            return 1; // Success
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception in basic quantize large: " << e.what() << std::endl;
            return -1; // Failure
        } catch (...) {
            std::cout << "DEBUG: Unknown exception in basic quantize large" << std::endl;
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
        printf("DEBUG: C interface called with rows=%u, cols=%u\n", rows, cols);
        ryzen_llm::bitnet::QuantConfig config;
        printf("DEBUG: About to call scalar function\n");
        ryzen_llm::bitnet::TernaryWeight result =
            ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, rows, cols, config);
        printf("DEBUG: Scalar function returned, about to allocate\n");
        ryzen_llm::bitnet::TernaryWeight* result_ptr =
            new ryzen_llm::bitnet::TernaryWeight(result);
        printf("DEBUG: Allocation complete, about to return\n");
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
        std::cout << "DEBUG: test_floating_point_accumulation called" << std::endl;
        
        // Test basic floating-point operations
        float a = 1.0f;
        float b = 2.0f;
        float sum = a + b;
        std::cout << "DEBUG: basic addition: " << sum << std::endl;
        
        // Test std::abs
        float val = -3.5f;
        float abs_val = std::abs(val);
        std::cout << "DEBUG: std::abs: " << abs_val << std::endl;
        
        // Test accumulation loop (this is where it fails)
        float sum_abs = 0.0f;
        float weights[] = {1.0f, -2.0f, 3.0f, -4.0f};
        int num_weights = 4;
        
        std::cout << "DEBUG: starting accumulation loop" << std::endl;
        for (int i = 0; i < num_weights; ++i) {
            std::cout << "DEBUG: weight " << i << ": " << weights[i] << std::endl;
            float abs_weight = std::abs(weights[i]);
            std::cout << "DEBUG: abs_weight: " << abs_weight << std::endl;
            sum_abs += abs_weight;  // This line likely triggers the error
            std::cout << "DEBUG: sum_abs after addition: " << sum_abs << std::endl;
        }
        
        std::cout << "DEBUG: accumulation completed: " << sum_abs << std::endl;
        return static_cast<int>(sum_abs * 1000);  // Return scaled integer
    }

    // Test division operations
    __declspec(dllexport) int test_division_operations() {
        std::cout << "DEBUG: test_division_operations called" << std::endl;
        
        // Test basic division
        float numerator = 10.0f;
        float denominator = 4.0f;
        float result = numerator / denominator;
        std::cout << "DEBUG: basic division: " << result << std::endl;
        
        // Test division by zero protection
        float safe_div = (denominator > 0) ? (numerator / denominator) : 1.0f;
        std::cout << "DEBUG: safe division: " << safe_div << std::endl;
        
        // Test the exact division from quantization
        float sum_abs_original = 10.0f;
        float sum_abs_quantized = 5.0f;
        float scale = (sum_abs_quantized > 0) ? (sum_abs_original / sum_abs_quantized) : 1.0f;
        std::cout << "DEBUG: quantization-style division: " << scale << std::endl;
        
        // Test mean calculation
        float total = 0.0f;
        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        int count = 4;
        for (int i = 0; i < count; ++i) {
            total += values[i];
        }
        float mean = total / count;
        std::cout << "DEBUG: mean calculation: " << mean << std::endl;
        
        return static_cast<int>(scale * 1000);
    }

    // Test std::vector operations
    __declspec(dllexport) int test_vector_operations() {
        std::cout << "DEBUG: test_vector_operations called" << std::endl;
        
        // Test basic vector creation
        std::vector<int8_t> values;
        std::cout << "DEBUG: vector created" << std::endl;
        
        // Test resize operation (this might trigger AVX-512)
        size_t size = 16;
        values.resize(size);
        std::cout << "DEBUG: vector resized to " << size << std::endl;
        
        // Test filling vector
        for (size_t i = 0; i < size; ++i) {
            values[i] = static_cast<int8_t>(i % 3 - 1); // -1, 0, 1 pattern
        }
        std::cout << "DEBUG: vector filled" << std::endl;
        
        // Test float vector
        std::vector<float> scales;
        scales.resize(1);
        scales[0] = 2.5f;
        std::cout << "DEBUG: float vector created and set: " << scales[0] << std::endl;
        
        // Test larger resize
        size_t large_size = 64;
        values.resize(large_size);
        std::cout << "DEBUG: vector resized to " << large_size << std::endl;
        
        // Test multiple groups resize (like TernaryWeight)
        uint32_t r = 4, c = 4, gs = 8;
        uint32_t total_size = r * c;
        uint32_t num_groups = (gs > 0) ? ((total_size + gs - 1) / gs) : 1;
        
        std::vector<int8_t> test_values;
        std::vector<float> test_scales;
        
        test_values.resize(total_size);
        test_scales.resize(num_groups);
        
        std::cout << "DEBUG: TernaryWeight-style vectors created: values=" 
                  << total_size << ", scales=" << num_groups << std::endl;
        
        return static_cast<int>(values.size() + scales.size() + test_values.size() + test_scales.size());
    }

    // Test avoiding std::min_element and std::max_element
    __declspec(dllexport) int test_min_max_avoidance() {
        std::cout << "DEBUG: test_min_max_avoidance called" << std::endl;
        
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
        
        std::cout << "DEBUG: min_val=" << min_val << ", max_val=" << max_val << std::endl;
        
        // Continue with quantization logic (simplified)
        float abs_max = (fabs(min_val) > fabs(max_val)) ? fabs(min_val) : fabs(max_val);
        float scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
        
        std::cout << "DEBUG: abs_max=" << abs_max << ", scale=" << scale << std::endl;
        
        // Test quantization
        int8_t quantized[4];
        for (size_t i = 0; i < size; ++i) {
            float clamped = (activations[i] < -abs_max) ? -abs_max : 
                           ((activations[i] > abs_max) ? abs_max : activations[i]);
            quantized[i] = static_cast<int8_t>(round(clamped * scale));
            std::cout << "DEBUG: activation " << i << ": " << activations[i] 
                      << " -> " << (int)quantized[i] << std::endl;
        }
        
        return 42; // Success
    }

    __declspec(dllexport) int test_int8_vector_operations() {
        try {
            printf("DEBUG: test_int8_vector_operations called\n");
            
            // Test std::vector<int8_t> operations
            std::vector<int8_t> values;
            printf("DEBUG: int8_t vector created\n");
            
            values.resize(16);
            printf("DEBUG: int8_t vector resized to 16\n");
            
            // Fill with test data
            for (size_t i = 0; i < 16; ++i) {
                values[i] = static_cast<int8_t>(i - 8); // -8 to 7
                printf("DEBUG: set values[%zu] = %d\n", i, (int)values[i]);
            }
            printf("DEBUG: int8_t vector filled\n");
            
            // Test reading back
            int sum = 0;
            for (size_t i = 0; i < 16; ++i) {
                sum += values[i];
                printf("DEBUG: read values[%zu] = %d, sum=%d\n", i, (int)values[i], sum);
            }
            
            printf("DEBUG: int8_t vector operations completed successfully\n");
            return sum;
        } catch (const std::exception& e) {
            printf("DEBUG: Exception in test_int8_vector_operations: %s\n", e.what());
            return -1;
        } catch (...) {
            printf("DEBUG: Unknown exception in test_int8_vector_operations\n");
            return -2;
        }
    }
}

PYBIND11_MODULE(test_module, m)
{
    try {
        std::cout << "DEBUG: PYBIND11_MODULE test_module is executing!" << std::endl;
        m.def("test_function", []() { return 42; });
        std::cout << "DEBUG: test_function bound successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "DEBUG: Exception during binding: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "DEBUG: Unknown exception during binding!" << std::endl;
    }
}