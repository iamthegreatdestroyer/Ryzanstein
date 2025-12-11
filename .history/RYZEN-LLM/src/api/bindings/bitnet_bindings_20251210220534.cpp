#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bitnet/quantize.h"
#include "bitnet/engine.h"

PYBIND11_MODULE(ryzen_llm_bindings, m)
{
    printf("DEBUG: PYBIND11_MODULE ryzen_llm_bindings is executing!\n");
    m.doc() = "Ryzen-LLM BitNet bindings for high-performance inference";

    // Test function for validation
    m.def("test_function", []()
          { return std::string("Ryzen-LLM BitNet extension loaded successfully!"); }, "Test function to verify extension loading");

    // Quantization functions
    m.def("quantize_activations_int8", [](const std::vector<float> &activations, size_t size)
          {
        ryzen_llm::bitnet::QuantConfig config; // Use default config
        return ryzen_llm::bitnet::quantize_activations_int8(activations.data(), size, config); }, pybind11::arg("activations"), pybind11::arg("size"), "Quantize float activations to INT8 with ternary weights");

    m.def("quantize_weights_ternary", [](const std::vector<float> &weights, uint32_t rows, uint32_t cols)
          {
        ryzen_llm::bitnet::QuantConfig config; // Use default config
        return ryzen_llm::bitnet::quantize_weights_ternary(weights.data(), rows, cols, config); }, pybind11::arg("weights"), pybind11::arg("rows"), pybind11::arg("cols"), "Quantize float weights to ternary (-1, 0, 1) representation");

    m.def("dequantize_weights", [](const ryzen_llm::bitnet::TernaryWeight &ternary_weight)
          {
        std::vector<float> output(ternary_weight.rows * ternary_weight.cols);
        ryzen_llm::bitnet::dequantize_weights(ternary_weight, output.data());
        return output; }, pybind11::arg("ternary_weight"), "Dequantize ternary weights back to float representation");

    // BitNet Engine class
    pybind11::class_<ryzen_llm::bitnet::BitNetEngine>(m, "BitNetEngine")
        .def(pybind11::init<ryzen_llm::bitnet::ModelConfig>())
        .def("load_weights", &ryzen_llm::bitnet::BitNetEngine::load_weights,
             "Load model weights from file")
        .def("forward", &ryzen_llm::bitnet::BitNetEngine::forward,
             "Forward pass for single token")
        .def("reset_cache", &ryzen_llm::bitnet::BitNetEngine::reset_cache,
             "Reset KV cache for new sequence")
        .def("get_config", &ryzen_llm::bitnet::BitNetEngine::get_config,
             "Get model configuration");

    // ModelConfig structure
    pybind11::class_<ryzen_llm::bitnet::ModelConfig>(m, "ModelConfig")
        .def(pybind11::init<>())
        .def_readwrite("vocab_size", &ryzen_llm::bitnet::ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ryzen_llm::bitnet::ModelConfig::hidden_size)
        .def_readwrite("intermediate_size", &ryzen_llm::bitnet::ModelConfig::intermediate_size)
        .def_readwrite("num_layers", &ryzen_llm::bitnet::ModelConfig::num_layers)
        .def_readwrite("num_heads", &ryzen_llm::bitnet::ModelConfig::num_heads)
        .def_readwrite("head_dim", &ryzen_llm::bitnet::ModelConfig::head_dim)
        .def_readwrite("max_seq_length", &ryzen_llm::bitnet::ModelConfig::max_seq_length)
        .def_readwrite("rms_norm_eps", &ryzen_llm::bitnet::ModelConfig::rms_norm_eps)
        .def_readwrite("quant_config", &ryzen_llm::bitnet::ModelConfig::quant_config);

    // QuantConfig structure
    pybind11::class_<ryzen_llm::bitnet::QuantConfig>(m, "QuantConfig")
        .def(pybind11::init<>())
        .def_readwrite("per_group_scaling", &ryzen_llm::bitnet::QuantConfig::per_group_scaling)
        .def_readwrite("weight_group_size", &ryzen_llm::bitnet::QuantConfig::weight_group_size)
        .def_readwrite("activation_clip_value", &ryzen_llm::bitnet::QuantConfig::activation_clip_value)
        .def_readwrite("symmetric_activations", &ryzen_llm::bitnet::QuantConfig::symmetric_activations);

    // TernaryWeight structure
    pybind11::class_<ryzen_llm::bitnet::TernaryWeight>(m, "TernaryWeight")
        .def(pybind11::init<>())
        .def_readonly("values", &ryzen_llm::bitnet::TernaryWeight::values)
        .def_readonly("scales", &ryzen_llm::bitnet::TernaryWeight::scales)
        .def_readonly("rows", &ryzen_llm::bitnet::TernaryWeight::rows)
        .def_readonly("cols", &ryzen_llm::bitnet::TernaryWeight::cols)
        .def_readonly("group_size", &ryzen_llm::bitnet::TernaryWeight::group_size);

    // QuantizedActivation structure
    pybind11::class_<ryzen_llm::bitnet::QuantizedActivation>(m, "QuantizedActivation")
        .def(pybind11::init<>())
        .def_readonly("values", &ryzen_llm::bitnet::QuantizedActivation::values)
        .def_readonly("scale", &ryzen_llm::bitnet::QuantizedActivation::scale)
        .def_readonly("zero_point", &ryzen_llm::bitnet::QuantizedActivation::zero_point);
}