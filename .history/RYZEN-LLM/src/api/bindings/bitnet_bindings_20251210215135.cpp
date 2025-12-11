#include <pybind11/pybind11.h>

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    m.def("test_function", []() { return 42; });
}