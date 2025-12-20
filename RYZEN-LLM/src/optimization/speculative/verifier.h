#pragma once

#include <vector>
#include <cstdint>

namespace ryzen_llm {
namespace speculative {

struct VerifierConfig {
    uint32_t vocab_size = 0;
    float temperature = 1.0f;
    float rejection_threshold = 0.5f;
};

struct VerifierResult {
    std::vector<int> accepted_tokens;
    uint32_t num_accepted = 0;
    float acceptance_rate = 0.0f;
};

class Verifier {
public:
    explicit Verifier(const VerifierConfig &config);

    VerifierResult verify(
        const std::vector<int> &prefix,
        const std::vector<int> &draft_tokens,
        const std::vector<std::vector<float>> &target_logits);

private:
    int sample_token(const std::vector<float> &target_logits);

    bool check_acceptance_criteria(int draft_token, const std::vector<float> &target_probs);

    int rejection_sample(const std::vector<float> &target_probs, int rejected_token);

    std::vector<float> softmax(const std::vector<float> &logits);

    std::vector<float> apply_temperature(const std::vector<float> &logits, float temperature);

    VerifierConfig config_;
    uint64_t num_verifications_;
    uint64_t num_rejections_;
};

} // namespace speculative
} // namespace ryzen_llm
