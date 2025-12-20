# Optimization layer

set(OPTIMIZATION_SOURCES
    cache_manager.cpp
    avx512/matmul.cpp
    avx512/activation.cpp
    avx512/vnni.cpp
    speculative/draft_model.cpp
    speculative/verifier.cpp
    memory/kv_cache.cpp
    memory/kv_cache_optimized.cpp
    memory/pool.cpp
    ../recycler/basic_recycler.cpp
)

add_library(ryzen_llm_optimization STATIC ${OPTIMIZATION_SOURCES})

target_include_directories(ryzen_llm_optimization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../recycler
)

target_link_libraries(ryzen_llm_optimization
    ryzen_llm_core
)

# Enable AVX-512 compilation flags for supported files
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        avx512/matmul.cpp
        avx512/activation.cpp
        avx512/vnni.cpp
        PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vnni -march=native"
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set_source_files_properties(
        avx512/matmul.cpp
        avx512/activation.cpp
        avx512/vnni.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX512"
    )
endif()
