#include <pybind11/pybind11.h>
#include <iostream>

PYBIND11_MODULE(test_module, m)
{
    try {
        std::cout << "DEBUG: PYBIND11_MODULE test_module is executing!" << std::endl;
        m.def("test_function", []() { return 42; });
        std::cout << "DEBUG: test_function bound successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "DEBUG: Exception during binding: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "DEBUG: Unknown exception during binding!" << std::endl;
    }
}