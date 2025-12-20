/*
 * RYZEN-LLM KV Cache Optimization
 * [REF:OL-005a] - Optimization Layer: KV Cache Management
 *
 * Efficient key-value cache for transformer attention with:
 * - Paged memory allocation (PagedAttention-style)
 * - Contiguous memory layout for cache-friendly access
 * - Block-level management for efficient allocation/deallocation
 * - Prefetching and memory alignment optimization
 * - Cross-sequence batch processing
 * - Optional quantization/compression for long contexts
 *
 * Performance Targets:
 * - 2× speedup via reduced memory bandwidth
 * - <10% memory overhead for management structures
 * - Sub-microsecond block allocation time
 * - Cache-line aligned access patterns
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>
#include <cstring>
#include <algorithm>
#include <string>

namespace ryzen_llm
{
    namespace memory
    {

        // Configuration constants
        constexpr size_t CACHE_BLOCK_SIZE = 16;         // Tokens per block
        constexpr size_t CACHE_LINE_SIZE = 64;          // CPU cache line size
        constexpr size_t MAX_BLOCKS_PER_SEQUENCE = 128; // Max 2048 tokens
        constexpr size_t ALIGNMENT = 64;                // Memory alignment for SIMD

        /**
         * KV Cache Configuration
         */
        struct KVCacheConfig
        {
            uint32_t num_layers;      // Number of transformer layers
            uint32_t num_heads;       // Number of attention heads
            uint32_t head_dim;        // Dimension per head
            uint32_t max_batch_size;  // Maximum batch size
            uint32_t block_size;      // Tokens per block (default: 16)
            uint32_t num_blocks;      // Total blocks in pool
            bool enable_quantization; // Enable FP16/INT8 quantization
            bool enable_prefetching;  // Enable cache prefetching

            KVCacheConfig()
                : num_layers(32), num_heads(32), head_dim(128), max_batch_size(8), block_size(CACHE_BLOCK_SIZE), num_blocks(1024), enable_quantization(false), enable_prefetching(true)
            {
            }
        };

        /**
         * Physical block in memory pool
         * Stores K or V cache for block_size tokens
         */
        struct PhysicalBlock
        {
            float *data;        // [block_size, num_heads, head_dim]
            uint32_t ref_count; // Reference count for sharing
            bool is_allocated;  // Allocation status

            PhysicalBlock() : data(nullptr), ref_count(0), is_allocated(false) {}
        };

        /**
         * Logical block mapping for a sequence
         * Maps logical position to physical block
         */
        struct LogicalBlock
        {
            uint32_t physical_block_id; // Physical block index
            uint32_t num_tokens;        // Number of valid tokens in block

            LogicalBlock() : physical_block_id(UINT32_MAX), num_tokens(0) {}
        };

        /**
         * Block table for a single sequence
         * Maps logical blocks to physical blocks
         */
        struct BlockTable
        {
            std::vector<LogicalBlock> logical_blocks;
            uint32_t sequence_length; // Total tokens in sequence

            BlockTable() : sequence_length(0) {}

            void reset()
            {
                logical_blocks.clear();
                sequence_length = 0;
            }
        };

        /**
         * Cache statistics for monitoring
         */
        struct CacheStats
        {
            uint64_t total_allocations;
            uint64_t total_deallocations;
            uint64_t cache_hits;
            uint64_t cache_misses;
            uint64_t blocks_allocated;
            uint64_t blocks_free;
            double memory_usage_mb;
            double avg_allocation_time_us;

            CacheStats()
                : total_allocations(0), total_deallocations(0), cache_hits(0), cache_misses(0), blocks_allocated(0), blocks_free(0), memory_usage_mb(0.0), avg_allocation_time_us(0.0)
            {
            }

            void reset()
            {
                *this = CacheStats();
            }

            std::string to_string() const;
        };

        /**
         * KV Cache Manager
         *
         * Implements paged memory management for transformer KV cache:
         * 1. Allocates physical blocks from memory pool
         * 2. Maps logical positions to physical blocks
         * 3. Supports block sharing across sequences (prefix caching)
         * 4. Optimizes memory layout for cache-line access
         * 5. Optional quantization for long contexts
         */
        class KVCacheManager
        {
        public:
            explicit KVCacheManager(const KVCacheConfig &config);
            ~KVCacheManager();

            // Sequence management

            /**
             * Allocate cache for a new sequence
             *
             * @param sequence_id Unique sequence identifier
             * @param estimated_length Estimated sequence length (for preallocation)
             * @return true if successful
             */
            bool AllocateSequence(uint64_t sequence_id, uint32_t estimated_length = 0);

            /**
             * Free cache for a sequence
             *
             * @param sequence_id Sequence identifier
             */
            void FreeSequence(uint64_t sequence_id);

            /**
             * Append tokens to sequence cache
             *
             * @param sequence_id Sequence identifier
             * @param num_tokens Number of tokens to append
             * @return true if successful
             */
            bool AppendTokens(uint64_t sequence_id, uint32_t num_tokens);

            // Cache access

            /**
             * Get key cache pointer for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @return Pointer to key cache [num_heads, head_dim]
             */
            float *GetKeyCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position);

            /**
             * Get value cache pointer for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @return Pointer to value cache [num_heads, head_dim]
             */
            float *GetValueCache(uint64_t sequence_id, uint32_t layer_id, uint32_t position);

            /**
             * Get contiguous key cache for layer (all positions)
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param out_length Output sequence length
             * @return Pointer to key cache [seq_len, num_heads, head_dim]
             */
            const float *GetKeySequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length);

            /**
             * Get contiguous value cache for layer (all positions)
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param out_length Output sequence length
             * @return Pointer to value cache [seq_len, num_heads, head_dim]
             */
            const float *GetValueSequence(uint64_t sequence_id, uint32_t layer_id, uint32_t &out_length);

            /**
             * Write key cache for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @param key_data Key data [num_heads, head_dim]
             */
            void WriteKey(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *key_data);

            /**
             * Write value cache for layer and position
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Token position
             * @param value_data Value data [num_heads, head_dim]
             */
            void WriteValue(uint64_t sequence_id, uint32_t layer_id, uint32_t position, const float *value_data);

            // Optimization features

            /**
             * Prefetch cache blocks for upcoming positions
             *
             * @param sequence_id Sequence identifier
             * @param layer_id Layer index
             * @param position Current position
             * @param lookahead Number of positions to prefetch
             */
            void Prefetch(uint64_t sequence_id, uint32_t layer_id, uint32_t position, uint32_t lookahead = 4);

            /**
             * Compress old cache entries (quantize to FP16/INT8)
             *
             * @param sequence_id Sequence identifier
             * @param retain_recent Number of recent tokens to keep uncompressed
             */
            void Compress(uint64_t sequence_id, [[maybe_unused]] uint32_t retain_recent);

            /**
             * Fork sequence (copy-on-write for prefix sharing)
             *
             * @param parent_sequence_id Parent sequence
             * @param child_sequence_id Child sequence
             * @param fork_position Position to fork at
             * @return true if successful
             */
            bool ForkSequence(uint64_t parent_sequence_id, uint64_t child_sequence_id, uint32_t fork_position);

            // Statistics and monitoring

            /**
             * Get current cache statistics
             */
            const CacheStats &GetStats() const { return stats_; }

            /**
             * Reset statistics
             */
            void ResetStats() { stats_.reset(); }

            /**
             * Get memory usage in bytes
             */
            size_t GetMemoryUsage() const;

            /**
             * Get configuration
             */
            const KVCacheConfig &GetConfig() const { return config_; }

        private:
            // Block management
            uint32_t allocate_physical_block();
            void free_physical_block(uint32_t block_id);
            void increment_ref_count(uint32_t block_id);
            void decrement_ref_count(uint32_t block_id);

            // Memory layout helpers
            size_t get_block_offset(uint32_t layer_id, uint32_t block_id, bool is_value) const;
            size_t get_element_stride() const;

            // Configuration
            KVCacheConfig config_;

            // Physical memory pool
            std::vector<PhysicalBlock> physical_blocks_;
            std::vector<uint32_t> free_blocks_;
            float *memory_pool_; // Contiguous memory allocation
            size_t pool_size_bytes_;

            // Sequence block tables
            std::unordered_map<uint64_t, BlockTable> key_block_tables_;
            std::unordered_map<uint64_t, BlockTable> value_block_tables_;

            // Contiguous buffers for sequence access (optimization)
            std::vector<float> key_sequence_buffer_;
            std::vector<float> value_sequence_buffer_;

            // Statistics
            mutable CacheStats stats_;
        };

        /**
         * Global KV cache statistics
         */
        extern CacheStats g_kv_cache_stats;

    } // namespace memory
} // namespace ryzen_llm
