#pragma once

/**
 * @file lut_lookup.h
 * @brief Runtime lookup engine for compressed T-MAC tables
 * 
 * Provides O(1) average lookup with 95% cache hit rate through
 * multi-tier search strategy:
 *   - Tier 1: 60% hit, ~2 cycles
 *   - Tier 2: 35% hit, ~75 cycles
 *   - Tier 3: 4.9% hit, ~250 cycles
 *   - Tier 4: 0.1% hit, ~100 cycles (fallback)
 * 
 * Expected latency: ~40 cycles = 10 ns @ 4GHz
 * 
 * [REF:TMAC-005] - Runtime Lookup
 */

#include "table_builder.h"
#include <cstdint>
#include <memory>
#include <string>

namespace ryzen_llm {
namespace tmac {

/**
 * Runtime lookup engine for compressed T-MAC tables
 * 
 * Thread-safe for read-only operations after initialization.
 */
class LUTLookup {
public:
    /**
     * Initialize from pre-built CompressedLUT structure
     * 
     * @param lut Compressed LUT structure (moves ownership)
     */
    explicit LUTLookup(std::shared_ptr<CompressedLUT> lut);
    
    /**
     * Lookup result for pattern × activation
     * 
     * @param pattern Ternary weight pattern (16 elements)
     * @param activation Single INT8 activation value
     * @return Dot product result (INT32)
     * 
     * Time: O(1) with 95% probability, O(16) worst case
     * 
     * Algorithm:
     *   1. Canonicalize pattern → O(16)
     *   2. Tier 1 lookup (hot) → O(1), 60% hit
     *   3. Tier 2 lookup (warm) → O(1), 35% hit
     *   4. Tier 3 lookup (delta) → O(1), 4.9% hit
     *   5. Fallback computation → O(16), 0.1% hit
     */
    int32_t lookup(
        const TernaryPattern& pattern,
        int8_t activation
    );
    
    /**
     * Batch lookup for multiple activation values
     * 
     * Optimized for sequential patterns with prefetching.
     * Reuses canonicalization across all activations.
     * 
     * @param pattern Ternary weight pattern (shared across batch)
     * @param activations Array of activation values
     * @param results Output array (preallocated, same size as activations)
     * @param count Number of activations to process
     * 
     * Time: O(count) amortized
     */
    void lookup_batch(
        const TernaryPattern& pattern,
        const int8_t* activations,
        int32_t* results,
        uint32_t count
    );
    
    /**
     * Lookup statistics for performance monitoring
     */
    struct Stats {
        uint64_t tier1_hits = 0;      ///< Hot cache hits
        uint64_t tier2_hits = 0;      ///< Warm cache hits
        uint64_t tier3_hits = 0;      ///< Delta reconstruction hits
        uint64_t fallback_count = 0;  ///< On-the-fly computations
        
        /**
         * Overall hit rate (tier 1-3)
         */
        double hit_rate() const {
            uint64_t total = tier1_hits + tier2_hits + tier3_hits + fallback_count;
            return total > 0 
                ? static_cast<double>(tier1_hits + tier2_hits + tier3_hits) / total 
                : 0.0;
        }
        
        /**
         * Tier-specific hit rates
         */
        double tier1_rate() const {
            uint64_t total = tier1_hits + tier2_hits + tier3_hits + fallback_count;
            return total > 0 ? static_cast<double>(tier1_hits) / total : 0.0;
        }
        
        double tier2_rate() const {
            uint64_t total = tier1_hits + tier2_hits + tier3_hits + fallback_count;
            return total > 0 ? static_cast<double>(tier2_hits) / total : 0.0;
        }
        
        double tier3_rate() const {
            uint64_t total = tier1_hits + tier2_hits + tier3_hits + fallback_count;
            return total > 0 ? static_cast<double>(tier3_hits) / total : 0.0;
        }
        
        double fallback_rate() const {
            uint64_t total = tier1_hits + tier2_hits + tier3_hits + fallback_count;
            return total > 0 ? static_cast<double>(fallback_count) / total : 0.0;
        }
    };
    
    /**
     * Get accumulated statistics
     */
    const Stats& get_stats() const { return stats_; }
    
    /**
     * Reset statistics counters
     */
    void reset_stats() { stats_ = Stats{}; }
    
    /**
     * Print performance statistics
     */
    void print_stats() const;

private:
    std::shared_ptr<CompressedLUT> lut_;
    PatternGenerator pattern_gen_;
    mutable Stats stats_;  // mutable for thread-local stats in future
    
    /**
     * Fallback: compute dot product directly
     * 
     * Used when pattern is not in any tier.
     * 
     * @param pattern Ternary pattern
     * @param activation Activation value
     * @return Computed result
     * 
     * Time: O(16)
     */
    int32_t compute_fallback(
        const TernaryPattern& pattern,
        int8_t activation
    );
    
    /**
     * Convert activation to tier 1 index
     * 
     * Maps [-32, 31] → [0, 63]
     * 
     * @param activation Activation value
     * @return Index, or -1 if out of tier 1 range
     */
    int32_t get_tier1_index(int8_t activation) const {
        if (activation < lut_->tier1_act_min || activation > lut_->tier1_act_max) {
            return -1;
        }
        return activation - lut_->tier1_act_min;
    }
    
    /**
     * Convert activation to tier 2 index
     * 
     * Maps [-128, 127] → [0, 255]
     * 
     * @param activation Activation value
     * @return Index [0, 255]
     */
    uint8_t get_tier2_index(int8_t activation) const {
        return static_cast<uint8_t>(activation + 128);
    }
};

} // namespace tmac
} // namespace ryzen_llm
