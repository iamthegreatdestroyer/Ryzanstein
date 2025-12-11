#include <pybind11/pybind11.h>
#include <iostream>

PYBIND11_MODULE(test_module, m)
{
    std::cout << "DEBUG: PYBIND11_MODULE test_module is executing!" << std::endl;
    m.def("test_function", []() { return 42; });
}