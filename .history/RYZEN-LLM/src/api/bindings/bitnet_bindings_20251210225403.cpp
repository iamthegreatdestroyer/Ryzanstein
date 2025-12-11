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
            std::cout << "DEBUG: test_create_ternary_weight called" << std::endl;
            // Create a simple TernaryWeight with empty data
            ryzen_llm::bitnet::TernaryWeight* weight = new ryzen_llm::bitnet::TernaryWeight();
            std::cout << "DEBUG: TernaryWeight created successfully" << std::endl;
            delete weight;
            return 1; // Success
        } catch (const std::exception& e) {
            std::cout << "DEBUG: Exception creating TernaryWeight: " << e.what() << std::endl;
            return -1; // Failure
        } catch (...) {
            std::cout << "DEBUG: Unknown exception creating TernaryWeight" << std::endl;
            return -2; // Failure
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
        ryzen_llm::bitnet::QuantConfig config;
        ryzen_llm::bitnet::TernaryWeight* result =
            new ryzen_llm::bitnet::TernaryWeight(
                ryzen_llm::bitnet::quantize_weights_ternary_scalar(weights, rows, cols, config));
        return result;
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