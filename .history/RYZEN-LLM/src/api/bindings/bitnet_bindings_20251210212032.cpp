/*
 * RYZEN-LLM Python Bindings
 * [REF:API-008c] - API Layer: Python-C++ Bindings
 *
 * This module provides Python bindings for the C++ inference engines
 * using pybind11, enabling seamless integration with the FastAPI server.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../core/bitnet/engine.h"
#include "../../core/bitnet/quantize.h"

namespace py = pybind11;
using namespace ryzen_llm::bitnet;

// Python bindings for BitNet engine
PYBIND11_MODULE(ryzen_llm_bindings, m)
{
    m.doc() = "RYZEN-LLM Python bindings for high-performance inference";

    // ModelConfig class
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        .def_readwrite("num_layers", &ModelConfig::num_layers)
        .def_readwrite("num_heads", &ModelConfig::num_heads)
        .def_readwrite("head_dim", &ModelConfig::head_dim)
        .def_readwrite("intermediate_size", &ModelConfig::intermediate_size)
        .def_readwrite("max_seq_length", &ModelConfig::max_seq_length)
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("use_tmac", &ModelConfig::use_tmac)
        .def_readwrite("tmac_precompute_on_load", &ModelConfig::tmac_precompute_on_load)
        .def("__repr__", [](const ModelConfig &config)
             { return py::str("ModelConfig(vocab_size={}, hidden_size={}, num_layers={}, num_heads={})")
                   .format(config.vocab_size, config.hidden_size, config.num_layers, config.num_heads); });

    // GenerationConfig class
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_tokens", &GenerationConfig::max_tokens)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("seed", &GenerationConfig::seed)
        .def("__repr__", [](const GenerationConfig &config)
             { return py::str("GenerationConfig(max_tokens={}, temperature={}, top_k={}, top_p={})")
                   .format(config.max_tokens, config.temperature, config.top_k, config.top_p); });

    // BitNetEngine class
    py::class_<BitNetEngine>(m, "BitNetEngine")
        .def(py::init<const ModelConfig &>(),
             py::arg("config"),
             "Initialize BitNet engine with model configuration")

        .def("load_weights", &BitNetEngine::load_weights,
             py::arg("weights_path"),
             "Load model weights from file")

        .def("forward", &BitNetEngine::forward,
             py::arg("token_id"),
             py::arg("position"),
             "Run forward pass for single token at given position")

        .def("generate", &BitNetEngine::generate,
             py::arg("input_tokens"),
             py::arg("gen_config"),
             "Generate text from input tokens with generation config")

        .def("reset_cache", &BitNetEngine::reset_cache,
             "Reset KV cache for new sequence")

        .def("get_config", &BitNetEngine::get_config,
             "Get current model configuration")

        .def("__repr__", [](const BitNetEngine &engine)
             {
            const auto& config = engine.get_config();
            return py::str("BitNetEngine(vocab_size={}, hidden_size={}, num_layers={})")
                .format(config.vocab_size, config.hidden_size, config.num_layers); });

    // QuantConfig class
    py::class_<QuantConfig>(m, "QuantConfig")
        .def(py::init<>())
        .def_readwrite("activation_clip_value", &QuantConfig::activation_clip_value)
        .def_readwrite("symmetric_activations", &QuantConfig::symmetric_activations);

    // QuantizedActivation class
    py::class_<QuantizedActivation>(m, "QuantizedActivation")
        .def(py::init<>())
        .def_readwrite("values", &QuantizedActivation::values)
        .def_readwrite("scale", &QuantizedActivation::scale)
        .def_readwrite("zero_point", &QuantizedActivation::zero_point);

    // TernaryWeight class
    py::class_<TernaryWeight>(m, "TernaryWeight")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("rows"), py::arg("cols"), py::arg("group_size") = 0)
        .def_readwrite("values", &TernaryWeight::values)
        .def_readwrite("scales", &TernaryWeight::scales)
        .def_readwrite("rows", &TernaryWeight::rows)
        .def_readwrite("cols", &TernaryWeight::cols)
        .def_readwrite("group_size", &TernaryWeight::group_size);

    // Test function to verify bindings are working
    m.def("test_quantization_available", []() {
        return true;
    }, "Test function to verify quantization bindings are loaded");

    // Quantization functions
    m.def("quantize_activations_int8", [](const py::array_t<float> &activations, const QuantConfig &config) {
        py::buffer_info buf = activations.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("activations must be 1-dimensional");
        }
        const float *data = static_cast<const float *>(buf.ptr);
        size_t size = buf.shape[0];
        return ryzen_llm::bitnet::quantize_activations_int8(data, size, config);
    }, py::arg("activations"), py::arg("config") = QuantConfig(),
    "Quantize activations to INT8 format");

    m.def("naive_ternary_matmul", [](const TernaryWeight &weights, const QuantizedActivation &activations, 
                                     py::array_t<float> &output, uint32_t M, uint32_t N, uint32_t K) {
        py::buffer_info buf = output.request();
        if (buf.ndim != 2 || buf.shape[0] != M || buf.shape[1] != N) {
            throw std::runtime_error("output must be M x N array");
        }
        float *output_data = static_cast<float *>(buf.ptr);
        ryzen_llm::bitnet::naive_ternary_matmul(weights, activations, output_data, M, N, K);
    }, py::arg("weights"), py::arg("activations"), py::arg("output"), py::arg("M"), py::arg("N"), py::arg("K"),
    "Perform naive ternary matrix multiplication");

    // Utility functions
    m.def("create_default_config", []()
          {
        ModelConfig config;
        // Set default BitNet b1.58 configuration
        config.vocab_size = 32000;
        config.hidden_size = 2048;
        config.num_layers = 26;
        config.num_heads = 32;
        config.head_dim = 64;
        config.intermediate_size = 5632;
        config.max_seq_length = 4096;
        config.rms_norm_eps = 1e-6f;
        config.use_tmac = true;
        config.tmac_precompute_on_load = true;
        return config; }, "Create default BitNet b1.58 model configuration");

    m.def("create_bitnet_1_58b_config", []()
          {
        ModelConfig config;
        config.vocab_size = 32000;
        config.hidden_size = 2048;
        config.num_layers = 26;
        config.num_heads = 32;
        config.head_dim = 64;
        config.intermediate_size = 5632;
        config.max_seq_length = 4096;
        config.rms_norm_eps = 1e-6f;
        config.use_tmac = true;
        config.tmac_precompute_on_load = true;
        return config; }, "Create BitNet 1.58B model configuration");

    m.def("create_bitnet_3b_config", []()
          {
        ModelConfig config;
        config.vocab_size = 32000;
        config.hidden_size = 3072;
        config.num_layers = 30;
        config.num_heads = 32;
        config.head_dim = 96;
        config.intermediate_size = 8192;
        config.max_seq_length = 4096;
        config.rms_norm_eps = 1e-6f;
        config.use_tmac = true;
        config.tmac_precompute_on_load = true;
        return config; }, "Create BitNet 3B model configuration");
}