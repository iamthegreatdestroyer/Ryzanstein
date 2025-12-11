#include <pybind11/pybind11.h>
#include <iostream>

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    std::cout << "PYBIND11_MODULE executing!" << std::endl;
    m.doc() = "Test module";
    m.def("test", []() { return 42; });
    throw std::runtime_error("Test exception from PYBIND11_MODULE");
}