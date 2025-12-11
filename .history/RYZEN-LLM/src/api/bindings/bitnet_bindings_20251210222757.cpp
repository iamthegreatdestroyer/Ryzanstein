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

    // Quantization functions
    __declspec(dllexport) void* quantize_weights_ternary_c(
        const float* weights, uint32_t rows, uint32_t cols) {
        ryzen_llm::bitnet::QuantConfig config;
        ryzen_llm::bitnet::TernaryWeight* result =
            new ryzen_llm::bitnet::TernaryWeight(
                ryzen_llm::bitnet::quantize_weights_ternary(weights, rows, cols, config));
        return result;
    }

    __declspec(dllexport) void* quantize_activations_int8_c(
        const float* activations, size_t size) {
        ryzen_llm::bitnet::QuantConfig config;
        ryzen_llm::bitnet::QuantizedActivation* result =
            new ryzen_llm::bitnet::QuantizedActivation(
                ryzen_llm::bitnet::quantize_activations_int8(activations, size, config));
        return result;
    }

    __declspec(dllexport) void dequantize_weights_c(
        void* ternary_weight_ptr, float* output) {
        auto* ternary_weight = static_cast<ryzen_llm::bitnet::TernaryWeight*>(ternary_weight_ptr);
        ryzen_llm::bitnet::dequantize_weights(*ternary_weight, output);
    }

    __declspec(dllexport) void dequantize_activations_c(
        void* quantized_ptr, float* output) {
        auto* quantized = static_cast<ryzen_llm::bitnet::QuantizedActivation*>(quantized_ptr);
        ryzen_llm::bitnet::dequantize_activations(*quantized, output);
    }

    __declspec(dllexport) float compute_quantization_error_c(
        const float* original, const float* quantized, size_t size) {
        return ryzen_llm::bitnet::compute_quantization_error(original, quantized, size);
    }

    // Memory management
    __declspec(dllexport) void free_ternary_weight(void* ptr) {
        delete static_cast<ryzen_llm::bitnet::TernaryWeight*>(ptr);
    }

    __declspec(dllexport) void free_quantized_activation(void* ptr) {
        delete static_cast<ryzen_llm::bitnet::QuantizedActivation*>(ptr);
    }

    // Accessor functions
    __declspec(dllexport) uint32_t get_ternary_weight_rows(void* ptr) {
        return static_cast<ryzen_llm::bitnet::TernaryWeight*>(ptr)->rows;
    }

    __declspec(dllexport) uint32_t get_ternary_weight_cols(void* ptr) {
        return static_cast<ryzen_llm::bitnet::TernaryWeight*>(ptr)->cols;
    }

    __declspec(dllexport) size_t get_quantized_activation_size(void* ptr) {
        return static_cast<ryzen_llm::bitnet::QuantizedActivation*>(ptr)->values.size();
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