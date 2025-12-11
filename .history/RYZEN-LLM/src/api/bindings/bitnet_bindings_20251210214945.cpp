#include <pybind11/pybind11.h>
#include <iostream>

PYBIND11_MODULE(ryzen_llm_bindings, m) {
    std::cout << "DEBUG: PYBIND11_MODULE is executing!" << std::endl;
    m.doc() = "Ryzen LLM C++ bindings";
    m.def("test_function", []() { 
        std::cout << "DEBUG: test_function called!" << std::endl;
        return 42; 
    });
}