# Core inference engines

set(CORE_SOURCES
    bitnet/engine.cpp
    bitnet/quantize.cpp
    tmac/lut_gemm.cpp
    mamba/ssm.cpp
    mamba/scan.cpp
    rwkv/wkv.cpp
    rwkv/time_mixing.cpp
)

add_library(ryzen_llm_core STATIC ${CORE_SOURCES})

target_include_directories(ryzen_llm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
