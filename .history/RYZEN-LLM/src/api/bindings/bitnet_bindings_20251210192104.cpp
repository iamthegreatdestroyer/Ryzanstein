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
#include "../../core/bitnet/engine.h"  // For ModelConfig

namespace py = pybind11;
using namespace ryzen_llm::bitnet;

// Python bindings for BitNet engine
PYBIND11_MODULE(ryzen_llm_bindings, m) {
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
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        .def("__repr__", [](const ModelConfig& config) {
            return py::str("ModelConfig(vocab_size={}, hidden_size={}, num_layers={}, num_heads={})")
                .format(config.vocab_size, config.hidden_size, config.num_layers, config.num_heads);
        });

    // BitNetEngine class
    py::class_<BitNetEngine>(m, "BitNetEngine")
        .def(py::init<const ModelConfig&>(),
             py::arg("config"),
             "Initialize BitNet engine with model configuration")

        .def("load_weights", &BitNetEngine::load_weights,
             py::arg("weights_path"),
             "Load model weights from file")

        .def("forward", [](BitNetEngine& engine, const std::vector<int>& tokens) {
            // Convert tokens to numpy array for processing
            py::array_t<int> token_array(tokens.size());
            auto token_buf = token_array.request();
            int* token_ptr = static_cast<int*>(token_buf.ptr);
            std::copy(tokens.begin(), tokens.end(), token_ptr);

            // Call forward pass
            auto result = engine.forward(tokens);

            // Convert result to Python list
            return py::cast(result);
        },
        py::arg("tokens"),
        "Run forward pass on input tokens")

        .def("generate", [](BitNetEngine& engine,
                           const std::vector<int>& prompt_tokens,
                           int max_new_tokens,
                           float temperature,
                           int top_k,
                           float top_p) {
            // Call generation method
            auto result = engine.generate(prompt_tokens, max_new_tokens,
                                        temperature, top_k, top_p);

            // Convert result to Python list
            return py::cast(result);
        },
        py::arg("prompt_tokens"),
        py::arg("max_new_tokens") = 100,
        py::arg("temperature") = 0.7f,
        py::arg("top_k") = 50,
        py::arg("top_p") = 1.0f,
        "Generate text from prompt tokens")

        .def("reset_cache", &BitNetEngine::reset_cache,
             "Reset KV cache for new sequence")

        .def("get_config", &BitNetEngine::get_config,
             "Get current model configuration")

        .def("__repr__", [](const BitNetEngine& engine) {
            const auto& config = engine.get_config();
            return py::str("BitNetEngine(vocab_size={}, hidden_size={}, num_layers={})")
                .format(config.vocab_size, config.hidden_size, config.num_layers);
        });

    // Utility functions
    m.def("create_default_config", []() {
        ModelConfig config;
        // Set default BitNet b1.58 configuration
        config.vocab_size = 32000;
        config.hidden_size = 2048;
        config.num_layers = 26;
        config.num_heads = 32;
        config.head_dim = 64;
        config.intermediate_size = 5632;
        config.max_seq_length = 4096;
        config.rope_theta = 10000.0f;
        return config;
    }, "Create default BitNet b1.58 model configuration");

    m.def("create_bitnet_1_58b_config", []() {
        ModelConfig config;
        config.vocab_size = 32000;
        config.hidden_size = 2048;
        config.num_layers = 26;
        config.num_heads = 32;
        config.head_dim = 64;
        config.intermediate_size = 5632;
        config.max_seq_length = 4096;
        config.rope_theta = 10000.0f;
        return config;
    }, "Create BitNet 1.58B model configuration");

    m.def("create_bitnet_3b_config", []() {
        ModelConfig config;
        config.vocab_size = 32000;
        config.hidden_size = 3072;
        config.num_layers = 30;
        config.num_heads = 32;
        config.head_dim = 96;
        config.intermediate_size = 8192;
        config.max_seq_length = 4096;
        config.rope_theta = 10000.0f;
        return config;
    }, "Create BitNet 3B model configuration");
}