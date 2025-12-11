#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bitnet/quantize.h"
#include "bitnet/engine.h"

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    m.doc() = "Ryzen-LLM BitNet bindings for high-performance inference";

    // Quantization functions
    m.def("quantize_activations_int8", [](const std::vector<float>& activations, size_t size) {
        bitnet::QuantConfig config; // Use default config
        return bitnet::quantize_activations_int8(activations.data(), size, config);
    }, pybind11::arg("activations"), pybind11::arg("size"),
       "Quantize float activations to INT8 with ternary weights");

    m.def("quantize_weights_ternary", [](const std::vector<float>& weights, size_t size) {
        return bitnet::quantize_weights_ternary(weights.data(), size);
    }, pybind11::arg("weights"), pybind11::arg("size"),
       "Quantize float weights to ternary (-1, 0, 1) representation");

    m.def("dequantize_ternary", [](const std::vector<int8_t>& ternary_weights, size_t size) {
        return bitnet::dequantize_ternary(ternary_weights.data(), size);
    }, pybind11::arg("ternary_weights"), pybind11::arg("size"),
       "Dequantize ternary weights back to float representation");

    // BitNet Engine class
    pybind11::class_<bitnet::BitNetEngine>(m, "BitNetEngine")
        .def(pybind11::init<>())
        .def("load_weights", &bitnet::BitNetEngine::load_weights,
             "Load model weights from file")
        .def("forward", &bitnet::BitNetEngine::forward,
             "Perform forward pass with quantized operations")
        .def("quantize_layer", &bitnet::BitNetEngine::quantize_layer,
             "Quantize a specific layer's weights and activations");

    // QuantConfig structure
    pybind11::class_<bitnet::QuantConfig>(m, "QuantConfig")
        .def(pybind11::init<>())
        .def_readwrite("activation_bits", &bitnet::QuantConfig::activation_bits)
        .def_readwrite("weight_bits", &bitnet::QuantConfig::weight_bits)
        .def_readwrite("group_size", &bitnet::QuantConfig::group_size);
}