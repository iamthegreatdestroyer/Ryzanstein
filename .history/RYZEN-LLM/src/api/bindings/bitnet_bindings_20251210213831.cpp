#include <pybind11/pybind11.h>
#include <iostream>

PYBIND11_MODULE(test_module, m) {
    std::cout << "PYBIND11_MODULE test_module executing!" << std::endl;
    m.doc() = "Test module";
    m.def("test_function", []() { return 42; });
    throw std::runtime_error("Test exception from test_module");
}