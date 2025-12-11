#include "speculative_decoder.h"

namespace ryzen_llm
{
    namespace speculative
    {
        SpeculativeStats::SpeculativeStats() : total_tokens_generated(0), total_forward_passes(0)
        {
        }

        void SpeculativeStats::reset()
        {
            draft_stats.reset();
            verifier_stats.reset();
            total_tokens_generated = 0;
            total_forward_passes = 0;
        }

        SpeculativeDecoder::SpeculativeDecoder(const SpeculativeConfig &config)
            : config_(config),
              draft_model_(config.draft_config),
              verifier_(config.verifier_config),
              stats_()
        {
            // Validate configuration
            if (!config.enable_speculative_decoding && config.draft_config.K > 1)
            {
                // Warning: speculative decoding disabled but K > 1
            }
        }

        SpeculativeDecoder::~SpeculativeDecoder() = default;

        const SpeculativeConfig &SpeculativeDecoder::get_config() const
        {
            return config_;
        }

        DraftModel &SpeculativeDecoder::get_draft_model()
        {
            return draft_model_;
        }

        const DraftModel &SpeculativeDecoder::get_draft_model() const
        {
            return draft_model_;
        }

        Verifier &SpeculativeDecoder::get_verifier()
        {
            return verifier_;
        }

        const Verifier &SpeculativeDecoder::get_verifier() const
        {
            return verifier_;
        }

        const SpeculativeStats &SpeculativeDecoder::get_stats() const
        {
            return stats_;
        }

        void SpeculativeDecoder::reset_stats()
        {
            stats_.reset();
        }

        std::vector<uint32_t> SpeculativeDecoder::generate_candidates(
            const std::vector<uint32_t> &prefix,
            const std::vector<float> &context_vector)
        {
            // Delegate to draft model
            return draft_model_.generate_candidates(prefix, context_vector);
        }

        VerificationResult SpeculativeDecoder::verify_tokens(
            const std::vector<uint32_t> &draft_tokens,
            const std::vector<std::vector<float>> &target_logits)
        {
            // Delegate to verifier
            return verifier_.verify(draft_tokens, target_logits);
        }

    } // namespace speculative
} // namespace ryzen_llm
