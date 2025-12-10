# RYZEN-LLM CMake Build Configuration
# [REF:AP-009] - Appendix: Technical Stack
# Requires CMake 3.20+ for C++17 support

cmake_minimum_required(VERSION 3.20)
project(RYZEN-LLM VERSION 0.1.0 LANGUAGES CXX)

# C++17 Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# AVX-512 Detection and Flags
include(CheckCXXCompilerFlag)

# Check for AVX-512 support
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)
check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512VNNI)

if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512DQ AND COMPILER_SUPPORTS_AVX512VL)
    message(STATUS "AVX-512 support detected")
    set(AVX512_FLAGS "-mavx512f -mavx512dq -mavx512vl")
    
    if(COMPILER_SUPPORTS_AVX512VNNI)
        message(STATUS "AVX-512 VNNI support detected")
        set(AVX512_FLAGS "${AVX512_FLAGS} -mavx512vnni")
    endif()
    
    add_compile_options(${AVX512_FLAGS})
    add_definitions(-DHAVE_AVX512)
else()
    message(WARNING "AVX-512 not supported by compiler, some optimizations will be disabled")
endif()

# FMA3 support
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_FMA)
    message(STATUS "FMA3 support detected")
    add_compile_options(-mfma)
    add_definitions(-DHAVE_FMA)
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/optimization
)

# Core inference engines
add_subdirectory(src/core)

# Optimization layer
add_subdirectory(src/optimization)

# Libraries
set(CORE_LIBRARIES
    ryzen_llm_core
    ryzen_llm_optimization
)

# Main executable (if needed)
# add_executable(ryzen-llm-server src/main.cpp)
# target_link_libraries(ryzen-llm-server ${CORE_LIBRARIES})

# Testing
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS ${CORE_LIBRARIES}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Print configuration summary
message(STATUS "")
message(STATUS "========== RYZEN-LLM Configuration ==========")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "AVX-512: ${COMPILER_SUPPORTS_AVX512F}")
message(STATUS "AVX-512 VNNI: ${COMPILER_SUPPORTS_AVX512VNNI}")
message(STATUS "FMA3: ${COMPILER_SUPPORTS_FMA}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "=============================================")
message(STATUS "")
