#include <pybind11/pybind11.h>

PYBIND11_MODULE(test_module, m) {
    m.doc() = "Test module";
    m.def("test_function", []() { return 42; });
}