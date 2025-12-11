#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>

namespace ryzen_llm {
namespace speculative {

/**
 * @brief Configuration for draft model inference
 * 
 * Controls draft model behavior including vocabulary size, context window,
 * quantization settings, and dynamic adjustment parameters.
 */
struct DraftModelConfig {
    // Model dimensions
    uint32_t vocab_size;        // Number of tokens (typically 32k-256k)
    uint32_t hidden_dim;        // Hidden state dimension (e.g., 1024)
    uint32_t num_layers;        // Number of transformer layers
    uint32_t max_seq_len;       // Maximum sequence length for context
    
    // Inference mode (affects computational cost)
    bool use_quantization;      // Use INT8 quantization for faster inference
    
    // Sampling strategy
    float temperature;          // Temperature for top-k sampling (typical: 0.7-1.0)
    uint32_t top_k;             // Top-k sampling (0 = disabled, typical: 50)
    float top_p;                // Nucleus sampling threshold (typical: 0.9)
    
    // Dynamic K tuning
    float acceptance_rate_target;  // Target acceptance rate (typical: 0.7-0.8)
    uint32_t min_K;                // Minimum draft tokens (typical: 1)
    uint32_t max_K;                // Maximum draft tokens (typical: 4-8)
    uint32_t K_adjust_frequency;   // Adjust K every N forward passes (typical: 100)
    
    // Performance tracking
    bool enable_statistics;     // Track draft model performance metrics
};

/**
 * @brief Statistics tracking for draft model performance
 */
struct DraftModelStats {
    uint64_t num_inferences;       // Total forward passes
    uint64_t total_draft_tokens;   // Sum of K across all inferences
    uint64_t num_accepted;         // Total tokens accepted by verifier
    
    // Constructor to zero-initialize
    DraftModelStats()
        : num_inferences(0), total_draft_tokens(0), num_accepted(0) {}
    
    // Calculate average acceptance rate
    float get_acceptance_rate() const {
        if (total_draft_tokens == 0) return 0.0f;
        return static_cast<float>(num_accepted) / static_cast<float>(total_draft_tokens);
    }
    
    // Calculate average K (draft tokens per call)
    float get_avg_k() const {
        if (num_inferences == 0) return 0.0f;
        return static_cast<float>(total_draft_tokens) / static_cast<float>(num_inferences);
    }
    
    // Reset statistics
    void reset() {
        num_inferences = 0;
        total_draft_tokens = 0;
        num_accepted = 0;
    }
};

/**
 * @brief Draft model for speculative decoding
 * 
 * Fast, lightweight model that proposes K candidate tokens for the target model
 * to verify in parallel. Typically a smaller version (350M) vs target (7B+).
 * 
 * Supports:
 * - Greedy sampling (highest probability)
 * - Top-k sampling (sample from top K tokens)
 * - Nucleus (top-p) sampling (cumulative probability threshold)
 * - Dynamic draft length tuning based on acceptance rates
 * 
 * Example usage:
 * @code
 *   DraftModelConfig config = {...};
 *   DraftModel draft(config);
 *   
 *   std::vector<int> prefix = {101, 2054, 2003, ...};
 *   uint32_t K = 4;
 *   std::vector<int> candidates = draft.generate_candidates(prefix, K);
 *   // candidates has K tokens
 * @endcode
 */
class DraftModel {
public:
    /**
     * @brief Initialize draft model with configuration
     * @param config Configuration for draft model behavior
     */
    explicit DraftModel(const DraftModelConfig& config);
    
    /**
     * @brief Generate K candidate tokens given a prefix
     * 
     * Performs forward pass on draft model to get logits, applies sampling
     * strategy to generate K candidate tokens for the target model to verify.
     * 
     * @param prefix Token IDs representing context (e.g., [101, 2054, 2003])
     * @param K Number of tokens to generate (typically 1-8)
     * @return Vector of K candidate token IDs
     * 
     * @note Prefix should not exceed max_seq_len from config
     * @note Returns empty vector on error (invalid prefix, etc.)
     */
    std::vector<int> generate_candidates(
        const std::vector<int>& prefix,
        uint32_t K
    );
    
    /**
     * @brief Record acceptance feedback for a draft token
     * 
     * Updates internal statistics and can trigger dynamic K adjustment
     * if acceptance_rate deviates from target.
     * 
     * @param token_id The draft token that was proposed
     * @param was_accepted Whether verifier accepted this token
     * 
     * @note Called by Verifier after each token is processed
     */
    void record_acceptance(int token_id, bool was_accepted);
    
    /**
     * @brief Get current draft length (K)
     * @return Current K value (number of tokens to generate)
     */
    uint32_t get_current_K() const { return current_K_; }
    
    /**
     * @brief Manually set draft length
     * @param K New draft length (clamped to [min_K, max_K])
     */
    void set_K(uint32_t K) {
        current_K_ = std::clamp(K, config_.min_K, config_.max_K);
    }
    
    /**
     * @brief Get current statistics
     * @return Copy of current performance statistics
     */
    DraftModelStats get_stats() const { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void reset_stats() { stats_.reset(); }
    
    /**
     * @brief Get configuration
     * @return Reference to configuration
     */
    const DraftModelConfig& get_config() const { return config_; }

private:
    DraftModelConfig config_;
    uint32_t current_K_;           // Current draft length (adaptive)
    DraftModelStats stats_;        // Performance tracking
    
    /**
     * @brief Internal: Get logits from model forward pass
     * 
     * Computes probability distribution over vocabulary for the next token
     * given a prefix. This would call the actual draft model inference.
     * 
     * @param prefix Token IDs representing context
     * @return Logits vector of size vocab_size
     */
    std::vector<float> forward(const std::vector<int>& prefix);
    
    /**
     * @brief Internal: Apply sampling strategy to logits
     * 
     * Converts logits to probabilities and applies temperature, top-k, 
     * and top-p filtering to create a valid sampling distribution.
     * 
     * @param logits Raw model outputs (vocab_size elements)
     * @return Probability distribution (sums to 1.0, size vocab_size)
     */
    std::vector<float> sample_distribution(const std::vector<float>& logits);
    
    /**
     * @brief Internal: Sample token from probability distribution
     * 
     * Draws a single token ID from the probability distribution using
     * cumulative probability (inverse transform sampling).
     * 
     * @param probs Probability distribution (sums to 1.0)
     * @return Sampled token ID (0 to vocab_size-1)
     */
    int sample_token(const std::vector<float>& probs);
    
    /**
     * @brief Internal: Adjust K based on acceptance rate
     * 
     * If current acceptance rate deviates from target, increase or decrease K.
     * Called periodically (every K_adjust_frequency inferences).
     */
    void adjust_K_adaptive();
    
    /**
     * @brief Internal: Apply softmax to logits
     * 
     * Converts logits to probabilities using softmax transformation.
     * Includes numerical stability tricks to avoid overflow.
     * 
     * @param logits Raw model outputs
     * @return Probability distribution (sums to 1.0)
     */
    static std::vector<float> softmax(const std::vector<float>& logits);
    
    /**
     * @brief Internal: Apply temperature scaling
     * 
     * Divides logits by temperature before softmax to control randomness:
     * - T → 0: Greedy sampling (argmax)
     * - T = 1: Standard softmax
     * - T → ∞: Uniform distribution
     * 
     * @param logits Raw model outputs
     * @param temperature Temperature value (typical: 0.7-1.0)
     * @return Scaled logits
     */
    static std::vector<float> apply_temperature(
        const std::vector<float>& logits,
        float temperature
    );
    
    /**
     * @brief Internal: Apply top-k filtering
     * 
     * Keeps only the K highest probability tokens, sets others to 0.
     * When K=0, disables top-k filtering.
     * 
     * @param probs Probability distribution
     * @param k Number of top tokens to keep
     * @return Filtered distribution (renormalized)
     */
    static std::vector<float> apply_top_k(
        const std::vector<float>& probs,
        uint32_t k
    );
    
    /**
     * @brief Internal: Apply nucleus (top-p) filtering
     * 
     * Keeps tokens until cumulative probability exceeds p.
     * When p=1.0, disables nucleus filtering.
     * 
     * @param probs Probability distribution
     * @param p Cumulative probability threshold (0.0-1.0)
     * @return Filtered distribution (renormalized)
     */
    static std::vector<float> apply_top_p(
        const std::vector<float>& probs,
        float p
    );
};

}  // namespace speculative
}  // namespace ryzen_llm
