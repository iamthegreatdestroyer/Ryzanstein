#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <cmath>
#include <algorithm>

namespace ryzen_llm
{
    namespace speculative
    {

        /**
         * @brief Configuration for verifier (target model batch verification)
         *
         * Controls how the verifier validates draft tokens against the target model's
         * output distribution. Includes rejection sampling parameters and statistical
         * tracking.
         */
        struct VerifierConfig
        {
            // Model dimensions
            uint32_t vocab_size; // Number of tokens (same as target model)
            uint32_t hidden_dim; // Hidden state dimension

            // Verification strategy
            float temperature;           // Temperature for resampling on rejection (typical: 1.0)
            bool use_rejection_sampling; // True: rejection sample on mismatch (correct distribution)
                                         // False: greedy accept longest match (faster but biased)

            // Statistics
            bool enable_statistics; // Track verification metrics
        };

        /**
         * @brief Statistics tracking for verifier performance
         */
        struct VerifierStats
        {
            uint64_t num_verifications;    // Total verification batches
            uint64_t total_draft_tokens;   // Total draft tokens presented
            uint64_t total_accepted;       // Total tokens accepted (correct on first try)
            uint64_t total_rejected;       // Total tokens rejected (needed resampling)
            uint64_t total_prefix_matches; // Total tokens in matching prefix

            // Constructor
            VerifierStats()
                : num_verifications(0), total_draft_tokens(0), total_accepted(0),
                  total_rejected(0), total_prefix_matches(0) {}

            // Calculate acceptance rate
            float get_acceptance_rate() const
            {
                if (total_draft_tokens == 0)
                    return 0.0f;
                return static_cast<float>(total_accepted) / static_cast<float>(total_draft_tokens);
            }

            // Calculate rejection rate
            float get_rejection_rate() const
            {
                return 1.0f - get_acceptance_rate();
            }

            // Average tokens per batch
            float get_avg_tokens_per_batch() const
            {
                if (num_verifications == 0)
                    return 0.0f;
                return static_cast<float>(total_draft_tokens) / static_cast<float>(num_verifications);
            }

            // Speedup estimate (assumes batch verification takes 1x cost)
            float get_estimated_speedup() const
            {
                if (total_draft_tokens == 0)
                    return 1.0f;
                // Speedup ≈ (K tokens verified) / (1 base inference + cost of rejections)
                // Simplified: speedup ≈ accepted / (1 + rejection overhead)
                float avg_k = get_avg_tokens_per_batch();
                float rejection_overhead = get_rejection_rate() * 0.5f; // 50% cost per rejection
                return avg_k / (1.0f + rejection_overhead);
            }

            // Reset statistics
            void reset()
            {
                num_verifications = 0;
                total_draft_tokens = 0;
                total_accepted = 0;
                total_rejected = 0;
                total_prefix_matches = 0;
            }
        };

        /**
         * @brief Result of verifying a batch of draft tokens
         */
        struct VerificationResult
        {
            std::vector<int> final_tokens; // Final accepted token sequence
            uint32_t num_accepted;         // Number of draft tokens accepted
            uint32_t longest_prefix;       // Length of matching prefix from draft
            bool had_rejections;           // True if any token was rejected/resampled

            // Constructor
            VerificationResult()
                : num_accepted(0), longest_prefix(0), had_rejections(false) {}
        };

        /**
         * @brief Verifier for speculative decoding
         *
         * Target model verifier that validates draft tokens against the target model's
         * output distribution. Performs batch inference on prefix + candidates, compares
         * probabilities, and accepts/rejects tokens accordingly.
         *
         * Algorithm (simplified):
         * 1. Run target model on [prefix + draft_token_1]
         * 2. If P_target[draft_token_1] ≥ P_draft[draft_token_1] * threshold:
         *    → Accept draft_token_1, continue to step 3
         * 3. Else: Reject, resample from target distribution, stop
         * 4. Return sequence of accepted tokens + resampled token
         *
         * This ensures output distribution matches target model exactly.
         *
         * Example usage:
         * @code
         *   VerifierConfig config = {...};
         *   Verifier verifier(config);
         *
         *   std::vector<int> prefix = {101, 2054, 2003};
         *   std::vector<int> draft_candidates = {2006, 1045, 2003, 2006};  // K=4
         *   std::vector<float> draft_probs = {...};  // From draft model
         *
         *   VerificationResult result = verifier.verify(
         *       prefix, draft_candidates, draft_probs
         *   );
         *
         *   // result.final_tokens might be [2006, 1045, 2003, 999]
         *   // (first 3 accepted from draft, 4th resampled)
         * @endcode
         */
        class Verifier
        {
        public:
            /**
             * @brief Initialize verifier with configuration
             * @param config Configuration for verification behavior
             */
            explicit Verifier(const VerifierConfig &config);

            /**
             * @brief Verify batch of draft tokens against target model
             *
             * Performs batch forward pass on target model for [prefix + each candidate],
             * compares probabilities, and determines which draft tokens to accept.
             * For rejected tokens, resamples from target distribution (optional).
             *
             * Algorithm:
             * - For i in [0, K):
             *   - Run target model on prefix + draft[0..i]
             *   - Compare target_prob[draft[i]] with draft_prob[draft[i]]
             *   - If acceptable: accept draft[i], continue
             *   - Else: reject, resample if enabled, stop
             *
             * @param prefix Current token context
             * @param draft_candidates K candidate tokens from draft model
             * @param draft_probs Probabilities from draft model (vocab_size elements)
             * @return VerificationResult with final tokens and statistics
             *
             * @note prefix and draft_candidates should not be empty
             * @note draft_probs must have exactly vocab_size elements
             */
            VerificationResult verify(
                const std::vector<int> &prefix,
                const std::vector<int> &draft_candidates,
                const std::vector<float> &draft_probs);

            /**
             * @brief Get current statistics
             * @return Copy of current verification statistics
             */
            VerifierStats get_stats() const { return stats_; }

            /**
             * @brief Reset statistics
             */
            void reset_stats() { stats_.reset(); }

            /**
             * @brief Get configuration
             * @return Reference to configuration
             */
            const VerifierConfig &get_config() const { return config_; }

        private:
            VerifierConfig config_;
            VerifierStats stats_;

            /**
             * @brief Internal: Get target model logits for a sequence
             *
             * Runs target model forward pass on the given sequence and returns
             * logits for the next token position.
             *
             * @param sequence Token IDs
             * @return Logits vector (size vocab_size)
             */
            std::vector<float> forward_target(const std::vector<int> &sequence);

            /**
             * @brief Internal: Check if draft token matches target distribution
             *
             * Compares P_target[token] with P_draft[token] * acceptance_threshold.
             * If target probability is high enough, accept the draft token.
             *
             * @param target_logits Logits from target model
             * @param token_id The draft token to check
             * @param draft_prob Probability from draft model (P_draft[token])
             * @return true if token should be accepted, false if should be rejected
             *
             * @note This implements the "acceptance test" from speculative decoding paper
             */
            bool should_accept(
                const std::vector<float> &target_logits,
                int token_id,
                float draft_prob);

            /**
             * @brief Internal: Resample token from target distribution
             *
             * When draft token is rejected, resample next token from target model's
             * distribution rather than using draft's choice. This maintains correct
             * output distribution.
             *
             * Implements: P_corrected(x) = max(0, P_target(x) - P_draft(x))
             * after renormalization.
             *
             * @param target_logits Logits from target model
             * @param draft_probs Probabilities from draft model (for correction term)
             * @return Resampled token ID
             */
            int resample_from_target(
                const std::vector<float> &target_logits,
                const std::vector<float> &draft_probs);

            /**
             * @brief Internal: Apply softmax to logits
             *
             * Converts logits to probabilities with numerical stability.
             *
             * @param logits Raw model outputs
             * @return Probability distribution (sums to 1.0)
             */
            static std::vector<float> softmax(const std::vector<float> &logits);

            /**
             * @brief Internal: Compute acceptance threshold from probabilities
             *
             * Used in should_accept() to determine if draft token matches target.
             * Higher threshold = stricter matching = fewer acceptances but faster.
             *
             * @param draft_prob Probability from draft model
             * @return Threshold for target probability
             *
             * @note Default implementation: threshold = draft_prob
             */
            float compute_acceptance_threshold(float draft_prob) const;
        };

    } // namespace speculative
} // namespace ryzen_llm
