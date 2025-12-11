#include <pybind11/pybind11.h>

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    m.doc() = "Test module";
    m.def("test", []() { return 42; });
}