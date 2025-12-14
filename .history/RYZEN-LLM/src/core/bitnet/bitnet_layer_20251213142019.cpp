/**
 * @file bitnet_layer.cpp
 * @brief Implementation of BitNet transformer layer
 * 
 * [REF:BITNET-001] - Forward Pass Implementation
 */

#include "bitnet_layer.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace ryzen_llm {
namespace bitnet {

// ============================================================================
// CONSTRUCTOR
// ============================================================================

BitNetLayer::BitNetLayer(
    const BitNetLayerParams& params,
    std::shared_ptr<tmac::TMACGemmOptimized> gemm_engine
)
    : params_(params)
    , gemm_engine_(std::move(gemm_engine))
{
    if (!gemm_engine_) {
        throw std::invalid_argument("GEMM engine cannot be null");
    }
    
    std::cout << "Initialized BitNet layer\n";
    std::cout << "  Hidden dim: " << params_.attn.hidden_dim << "\n";
    std::cout << "  Num heads: " << params_.attn.num_heads << "\n";
    std::cout << "  FFN dim: " << params_.ffn.ffn_dim << "\n";
}

size_t BitNetLayer::get_workspace_size(uint32_t batch_size, uint32_t seq_len) const {
    size_t hidden_dim = params_.attn.hidden_dim;
    
    // Intermediate activations needed:
    // - Attention: Q, K, V, scores, attention_output
    // - FFN: up_proj, down_proj
    
    size_t attn_qkv_size = 3 * batch_size * seq_len * hidden_dim * sizeof(float);
    size_t attn_scores_size = batch_size * params_.attn.num_heads * seq_len * seq_len * sizeof(float);
    size_t ffn_size = batch_size * seq_len * params_.ffn.ffn_dim * sizeof(float);
    
    return attn_qkv_size + attn_scores_size + ffn_size;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

void BitNetLayer::forward(
    const float* input,
    float* output,
    uint32_t batch_size,
    uint32_t seq_len,
    void* cache
) {
    size_t total_elements = batch_size * seq_len * params_.attn.hidden_dim;
    
    // Allocate workspace if needed
    size_t required_workspace = get_workspace_size(batch_size, seq_len);
    if (workspace_.size() < required_workspace / sizeof(float)) {
        workspace_.resize(required_workspace / sizeof(float));
    }
    
    // Use workspace for intermediate results
    float* attn_input = workspace_.data();
    float* attn_output = attn_input + total_elements;
    
    // Step 1: Pre-attention layer norm
    layer_norm(input, attn_input, params_.ln1, total_elements, params_.attn.hidden_dim);
    
    // Step 2: Multi-head self-attention
    multi_head_attention(attn_input, attn_output, batch_size, seq_len);
    
    // Step 3: Residual connection
    for (size_t i = 0; i < total_elements; ++i) {
        attn_output[i] += input[i];
    }
    
    // Step 4: Pre-FFN layer norm
    float* ffn_input = attn_input;  // Reuse buffer
    layer_norm(attn_output, ffn_input, params_.ln2, total_elements, params_.attn.hidden_dim);
    
    // Step 5: Feed-forward network
    float* ffn_output = output;
    feed_forward(ffn_input, ffn_output, batch_size, seq_len);
    
    // Step 6: Final residual connection
    for (size_t i = 0; i < total_elements; ++i) {
        output[i] += attn_output[i];
    }
}

// ============================================================================
// LAYER NORMALIZATION
// ============================================================================

void BitNetLayer::layer_norm(
    const float* input,
    float* output,
    const LayerNormParams& ln_params,
    uint32_t size,
    uint32_t hidden_dim
) {
    uint32_t num_vectors = size / hidden_dim;
    
    for (uint32_t i = 0; i < num_vectors; ++i) {
        const float* in_vec = input + i * hidden_dim;
        float* out_vec = output + i * hidden_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (uint32_t j = 0; j < hidden_dim; ++j) {
            mean += in_vec[j];
        }
        mean /= hidden_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (uint32_t j = 0; j < hidden_dim; ++j) {
            float diff = in_vec[j] - mean;
            variance += diff * diff;
        }
        variance /= hidden_dim;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(variance + ln_params.eps);
        for (uint32_t j = 0; j < hidden_dim; ++j) {
            float normalized = (in_vec[j] - mean) * inv_std;
            out_vec[j] = ln_params.gamma[j] * normalized + ln_params.beta[j];
        }
    }
}

// ============================================================================
// MULTI-HEAD ATTENTION
// ============================================================================

void BitNetLayer::multi_head_attention(
    const float* input,
    float* output,
    uint32_t batch_size,
    uint32_t seq_len
) {
    uint32_t hidden_dim = params_.attn.hidden_dim;
    uint32_t num_heads = params_.attn.num_heads;
    uint32_t head_dim = params_.attn.head_dim;
    
    // Allocate buffers for Q, K, V projections
    size_t qkv_size = batch_size * seq_len * hidden_dim;
    std::vector<int8_t> input_int8(qkv_size);
    std::vector<int32_t> qkv_int32(3 * qkv_size);
    std::vector<float> Q(qkv_size), K(qkv_size), V(qkv_size);
    
    // Quantize input to INT8
    float input_scale = quantize_to_int8(input, input_int8.data(), qkv_size);
    
    // Compute Q = input × W_q (using T-MAC)
    gemm_engine_->gemm(
        params_.attn.W_q.data(),
        input_int8.data(),
        qkv_int32.data(),
        hidden_dim, hidden_dim, batch_size * seq_len
    );
    dequantize_from_int32(qkv_int32.data(), Q.data(), qkv_size, input_scale);
    
    // Compute K = input × W_k
    gemm_engine_->gemm(
        params_.attn.W_k.data(),
        input_int8.data(),
        qkv_int32.data(),
        hidden_dim, hidden_dim, batch_size * seq_len
    );
    dequantize_from_int32(qkv_int32.data(), K.data(), qkv_size, input_scale);
    
    // Compute V = input × W_v
    gemm_engine_->gemm(
        params_.attn.W_v.data(),
        input_int8.data(),
        qkv_int32.data(),
        hidden_dim, hidden_dim, batch_size * seq_len
    );
    dequantize_from_int32(qkv_int32.data(), V.data(), qkv_size, input_scale);
    
    // Reshape to [batch, num_heads, seq_len, head_dim]
    // For simplicity, we'll process each head sequentially
    std::vector<float> attention_output(qkv_size, 0.0f);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            // Extract Q, K, V for this head
            size_t head_offset = h * head_dim;
            
            // Compute attention scores: scores = Q × K^T / sqrt(head_dim)
            std::vector<float> scores(seq_len * seq_len);
            
            for (uint32_t i = 0; i < seq_len; ++i) {
                for (uint32_t j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        size_t q_idx = b * seq_len * hidden_dim + i * hidden_dim + head_offset + d;
                        size_t k_idx = b * seq_len * hidden_dim + j * hidden_dim + head_offset + d;
                        score += Q[q_idx] * K[k_idx];
                    }
                    
                    scores[i * seq_len + j] = score * params_.attn.scale_factor;
                }
            }
            
            // Apply softmax
            softmax(scores.data(), seq_len, seq_len);
            
            // Compute attention output: output = scores × V
            for (uint32_t i = 0; i < seq_len; ++i) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    
                    for (uint32_t j = 0; j < seq_len; ++j) {
                        size_t v_idx = b * seq_len * hidden_dim + j * hidden_dim + head_offset + d;
                        sum += scores[i * seq_len + j] * V[v_idx];
                    }
                    
                    size_t out_idx = b * seq_len * hidden_dim + i * hidden_dim + head_offset + d;
                    attention_output[out_idx] = sum;
                }
            }
        }
    }
    
    // Output projection: output = attention_output × W_o
    std::vector<int8_t> attn_out_int8(qkv_size);
    float attn_scale = quantize_to_int8(attention_output.data(), attn_out_int8.data(), qkv_size);
    
    gemm_engine_->gemm(
        params_.attn.W_o.data(),
        attn_out_int8.data(),
        qkv_int32.data(),
        hidden_dim, hidden_dim, batch_size * seq_len
    );
    
    dequantize_from_int32(qkv_int32.data(), output, qkv_size, attn_scale);
}

// ============================================================================
// FEED-FORWARD NETWORK
// ============================================================================

void BitNetLayer::feed_forward(
    const float* input,
    float* output,
    uint32_t batch_size,
    uint32_t seq_len
) {
    uint32_t hidden_dim = params_.ffn.hidden_dim;
    uint32_t ffn_dim = params_.ffn.ffn_dim;
    size_t input_size = batch_size * seq_len * hidden_dim;
    
    // Up projection: hidden → ffn_dim
    std::vector<int8_t> input_int8(input_size);
    float input_scale = quantize_to_int8(input, input_int8.data(), input_size);
    
    size_t ffn_size = batch_size * seq_len * ffn_dim;
    std::vector<int32_t> up_proj_int32(ffn_size);
    std::vector<float> up_proj(ffn_size);
    
    gemm_engine_->gemm(
        params_.ffn.W_up.data(),
        input_int8.data(),
        up_proj_int32.data(),
        ffn_dim, hidden_dim, batch_size * seq_len
    );
    
    dequantize_from_int32(up_proj_int32.data(), up_proj.data(), ffn_size, input_scale);
    
    // Apply GELU activation
    gelu_activation(up_proj.data(), ffn_size);
    
    // Down projection: ffn_dim → hidden
    std::vector<int8_t> ffn_int8(ffn_size);
    float ffn_scale = quantize_to_int8(up_proj.data(), ffn_int8.data(), ffn_size);
    
    std::vector<int32_t> output_int32(input_size);
    
    gemm_engine_->gemm(
        params_.ffn.W_down.data(),
        ffn_int8.data(),
        output_int32.data(),
        hidden_dim, ffn_dim, batch_size * seq_len
    );
    
    dequantize_from_int32(output_int32.data(), output, input_size, ffn_scale);
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

void BitNetLayer::gelu_activation(float* x, size_t size) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
    constexpr float coeff = 0.044715f;
    
    for (size_t i = 0; i < size; ++i) {
        float x_val = x[i];
        float x_cubed = x_val * x_val * x_val;
        float tanh_arg = sqrt_2_over_pi * (x_val + coeff * x_cubed);
        x[i] = 0.5f * x_val * (1.0f + std::tanh(tanh_arg));
    }
}

void BitNetLayer::softmax(float* x, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        
        // Find max for numerical stability
        float max_val = row[0];
        for (uint32_t c = 1; c < cols; ++c) {
            max_val = std::max(max_val, row[c]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - max_val);
            sum += row[c];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (uint32_t c = 0; c < cols; ++c) {
            row[c] *= inv_sum;
        }
    }
}

// ============================================================================
// QUANTIZATION UTILITIES
// ============================================================================

float BitNetLayer::quantize_to_int8(
    const float* input,
    int8_t* output,
    size_t size
) {
    // Find max absolute value
    float max_abs = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::abs(input[i]));
    }
    
    // Compute scale
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;  // Avoid division by zero
    
    float inv_scale = 1.0f / scale;
    
    // Quantize
    for (size_t i = 0; i < size; ++i) {
        float scaled = input[i] * inv_scale;
        output[i] = static_cast<int8_t>(std::round(std::clamp(scaled, -127.0f, 127.0f)));
    }
    
    return scale;
}

void BitNetLayer::dequantize_from_int32(
    const int32_t* input,
    float* output,
    size_t size,
    float scale
) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

} // namespace bitnet
} // namespace ryzen_llm
