#include <pybind11/pybind11.h>
#include <iostream>

PyObject* PyInit_ryzen_llm_bindings() {
    std::cout << "PyInit_ryzen_llm_bindings called!" << std::endl;
    throw std::runtime_error("Test exception from PyInit function");
    return nullptr;
}