# T-MAC Lookup Table GEMM
# [REF:PHASE1-007] - Table-Based Matrix Multiplication

# Source files
set(TMAC_SOURCES
    lut_gemm.cpp
)

# Create T-MAC library
add_library(ryzen_llm_tmac STATIC ${TMAC_SOURCES})

# Include directories
target_include_directories(ryzen_llm_tmac PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link dependencies
target_link_libraries(ryzen_llm_tmac PUBLIC
    ryzen_llm_bitnet
    ryzen_llm_optimization
)

# Enable AVX-512 for T-MAC (vectorized table lookups)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ryzen_llm_tmac PRIVATE
        -mavx512f
        -mavx512bw
        -mavx512_vnni
        -march=native
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ryzen_llm_tmac PRIVATE
        /arch:AVX512
    )
endif()

# C++20 features required
target_compile_features(ryzen_llm_tmac PUBLIC cxx_std_20)
