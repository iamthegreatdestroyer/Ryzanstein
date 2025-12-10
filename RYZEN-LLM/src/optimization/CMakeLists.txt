# Optimization layer

set(OPTIMIZATION_SOURCES
    cache_manager.cpp
    avx512/matmul.cpp
    avx512/activation.cpp
    avx512/vnni.cpp
    speculative/draft_model.cpp
    speculative/verifier.cpp
    memory/kv_cache.cpp
    memory/pool.cpp
)

add_library(ryzen_llm_optimization STATIC ${OPTIMIZATION_SOURCES})

target_include_directories(ryzen_llm_optimization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ryzen_llm_optimization
    ryzen_llm_core
)
