/*
 * RYZEN-LLM BitNet Quantization Implementation
 * [REF:PHASE1-001] - BitNet b1.58 Ternary Quantization
 */

#include "quantize.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace ryzen_llm
{
    namespace bitnet
    {

        TernaryWeight quantize_weights_ternary(
            const float *weights,
            uint32_t rows,
            uint32_t cols,
            const QuantConfig &config)
        {
            const size_t total_size = static_cast<size_t>(rows) * cols;

            TernaryWeight result(rows, cols, config.per_group_scaling ? config.weight_group_size : 0);

            if (config.per_group_scaling)
            {
                // Per-group quantization
                const uint32_t num_groups = (total_size + config.weight_group_size - 1) / config.weight_group_size;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    const size_t group_start = g * config.weight_group_size;
                    const size_t group_end = std::min(group_start + config.weight_group_size, total_size);
                    const size_t group_len = group_end - group_start;

                    // Compute mean absolute value for this group
                    float mean_abs = 0.0f;
                    for (size_t i = group_start; i < group_end; ++i)
                    {
                        mean_abs += std::abs(weights[i]);
                    }
                    mean_abs /= group_len;

                    // Threshold for ternary quantization (typical: 0.5 to 1.0 * mean_abs)
                    const float threshold = 0.7f * mean_abs;

                    // Quantize weights in this group
                    float sum_abs_original = 0.0f;
                    float sum_abs_quantized = 0.0f;

                    for (size_t i = group_start; i < group_end; ++i)
                    {
                        const float w = weights[i];
                        sum_abs_original += std::abs(w);

                        if (std::abs(w) > threshold)
                        {
                            result.values[i] = (w > 0) ? 1 : -1;
                            sum_abs_quantized += 1.0f;
                        }
                        else
                        {
                            result.values[i] = 0;
                        }
                    }

                    // Compute scale factor for this group
                    // Scale to preserve magnitude: s = mean(|W|) / mean(|W_q|)
                    result.scales[g] = (sum_abs_quantized > 0)
                                           ? (sum_abs_original / sum_abs_quantized)
                                           : 1.0f;
                }
            }
            else
            {
                // Per-layer quantization
                // Compute mean absolute value across entire weight matrix
                float mean_abs = 0.0f;
                for (size_t i = 0; i < total_size; ++i)
                {
                    mean_abs += std::abs(weights[i]);
                }
                mean_abs /= total_size;

                const float threshold = 0.7f * mean_abs;

                // Quantize all weights
                float sum_abs_original = 0.0f;
                float sum_abs_quantized = 0.0f;

                for (size_t i = 0; i < total_size; ++i)
                {
                    const float w = weights[i];
                    sum_abs_original += std::abs(w);

                    if (std::abs(w) > threshold)
                    {
                        result.values[i] = (w > 0) ? 1 : -1;
                        sum_abs_quantized += 1.0f;
                    }
                    else
                    {
                        result.values[i] = 0;
                    }
                }

                // Compute single scale factor
                result.scales[0] = (sum_abs_quantized > 0)
                                       ? (sum_abs_original / sum_abs_quantized)
                                       : 1.0f;
            }

            return result;
        }

        QuantizedActivation quantize_activations_int8(
            const float *activations,
            size_t size,
            const QuantConfig &config)
        {
            QuantizedActivation result(size);

            if (config.symmetric_activations)
            {
                // Symmetric quantization: [-clip, clip] -> [-127, 127]
                const float clip_value = config.activation_clip_value;
                result.scale = clip_value / 127.0f;
                result.zero_point = 0;

                for (size_t i = 0; i < size; ++i)
                {
                    // Clip and quantize
                    float clipped = std::max(-clip_value, std::min(clip_value, activations[i]));
                    int32_t quantized = static_cast<int32_t>(std::round(clipped / result.scale));

                    // Clamp to INT8 range
                    result.values[i] = static_cast<int8_t>(
                        std::max(-127, std::min(127, quantized)));
                }
            }
            else
            {
                // Asymmetric quantization: [min, max] -> [-128, 127]
                float min_val = activations[0];
                float max_val = activations[0];

                for (size_t i = 1; i < size; ++i)
                {
                    min_val = std::min(min_val, activations[i]);
                    max_val = std::max(max_val, activations[i]);
                }

                // Compute scale and zero point
                const float range = max_val - min_val;
                result.scale = range / 255.0f; // Map to 256 levels
                result.zero_point = static_cast<int8_t>(
                    std::round(-128.0f - min_val / result.scale));

                for (size_t i = 0; i < size; ++i)
                {
                    int32_t quantized = static_cast<int32_t>(
                        std::round(activations[i] / result.scale) + result.zero_point);

                    result.values[i] = static_cast<int8_t>(
                        std::max(-128, std::min(127, quantized)));
                }
            }

            return result;
        }

        void dequantize_weights(
            const TernaryWeight &ternary_weight,
            float *output)
        {
            const size_t total_size = static_cast<size_t>(ternary_weight.rows) * ternary_weight.cols;

            for (size_t i = 0; i < total_size; ++i)
            {
                const float scale = ternary_weight.get_scale(i);
                output[i] = static_cast<float>(ternary_weight.values[i]) * scale;
            }
        }

        void dequantize_activations(
            const QuantizedActivation &quantized,
            float *output)
        {
            const size_t size = quantized.values.size();

            for (size_t i = 0; i < size; ++i)
            {
                output[i] = (static_cast<float>(quantized.values[i]) - quantized.zero_point) * quantized.scale;
            }
        }

        float compute_quantization_error(
            const float *original,
            const float *quantized,
            size_t size)
        {
            double sum_squared_error = 0.0;

            for (size_t i = 0; i < size; ++i)
            {
                const double diff = original[i] - quantized[i];
                sum_squared_error += diff * diff;
            }

            return static_cast<float>(sum_squared_error / size);
        }

        std::vector<uint8_t> pack_ternary_weights(const TernaryWeight &ternary_weight)
        {
            const size_t total_size = static_cast<size_t>(ternary_weight.rows) * ternary_weight.cols;
            const size_t packed_size = (total_size + 3) / 4; // 4 ternary values per byte

            std::vector<uint8_t> packed(packed_size, 0);

            for (size_t i = 0; i < total_size; ++i)
            {
                const size_t byte_idx = i / 4;
                const size_t bit_offset = (i % 4) * 2;

                // Map: -1 -> 00, 0 -> 01, +1 -> 10
                uint8_t code;
                switch (ternary_weight.values[i])
                {
                case -1:
                    code = 0b00;
                    break;
                case 0:
                    code = 0b01;
                    break;
                case 1:
                    code = 0b10;
                    break;
                default:
                    code = 0b01;
                    break; // Invalid, treat as 0
                }

                packed[byte_idx] |= (code << bit_offset);
            }

            return packed;
        }

        TernaryWeight unpack_ternary_weights(
            const std::vector<uint8_t> &packed,
            uint32_t rows,
            uint32_t cols)
        {
            TernaryWeight result(rows, cols);
            const size_t total_size = static_cast<size_t>(rows) * cols;

            for (size_t i = 0; i < total_size; ++i)
            {
                const size_t byte_idx = i / 4;
                const size_t bit_offset = (i % 4) * 2;

                const uint8_t code = (packed[byte_idx] >> bit_offset) & 0b11;

                // Map: 00 -> -1, 01 -> 0, 10 -> +1
                switch (code)
                {
                case 0b00:
                    result.values[i] = -1;
                    break;
                case 0b01:
                    result.values[i] = 0;
                    break;
                case 0b10:
                    result.values[i] = 1;
                    break;
                default:
                    result.values[i] = 0;
                    break;
                }
            }

            return result;
        }

        void naive_ternary_matmul(
            const TernaryWeight &weights,
            const QuantizedActivation &activations,
            float *output,
            uint32_t M,
            uint32_t N,
            uint32_t K)
        {
            // Zero output matrix
            std::memset(output, 0, M * N * sizeof(float));

            const float activation_scale = activations.scale;
            const int8_t activation_zero_point = activations.zero_point;

            // Naive triple-loop matrix multiplication
            for (uint32_t m = 0; m < M; ++m)
            {
                for (uint32_t n = 0; n < N; ++n)
                {
                    float sum = 0.0f;

                    for (uint32_t k = 0; k < K; ++k)
                    {
                        // Get quantized activation
                        const int8_t quantized_x = activations.values[m * K + k];
                        const float dequantized_x =
                            (static_cast<float>(quantized_x) - activation_zero_point) * activation_scale;

                        // Get ternary weight and scale
                        const int8_t ternary_w = weights.values[k * N + n];
                        const float weight_scale = weights.get_scale(k * N + n);
                        const float scaled_weight = static_cast<float>(ternary_w) * weight_scale;

                        // Accumulate
                        sum += dequantized_x * scaled_weight;
                    }

                    output[m * N + n] = sum;
                }
            }
        }

        // Scalar-only versions for CPU compatibility (no SIMD instructions)

        TernaryWeight quantize_weights_ternary_scalar(
            const float *weights,
            uint32_t rows,
            uint32_t cols,
            const QuantConfig &config)
        {
            std::cout << "DEBUG: quantize_weights_ternary_scalar called with rows=" << rows << ", cols=" << cols << std::endl;
            const size_t total_size = static_cast<size_t>(rows) * cols;

            TernaryWeight result(rows, cols, config.per_group_scaling ? config.weight_group_size : 0);

            if (config.per_group_scaling)
            {
                // Per-group quantization
                const uint32_t num_groups = (total_size + config.weight_group_size - 1) / config.weight_group_size;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    const size_t group_start = g * config.weight_group_size;
                    const size_t group_end = std::min(group_start + config.weight_group_size, total_size);
                    const size_t group_len = group_end - group_start;

                    // Compute mean absolute value for this group
                    float mean_abs = 0.0f;
                    for (size_t i = group_start; i < group_end; ++i)
                    {
                        mean_abs += std::abs(weights[i]);
                    }
                    mean_abs /= group_len;

                    // Threshold for ternary quantization (typical: 0.5 to 1.0 * mean_abs)
                    const float threshold = 0.7f * mean_abs;

                    // Quantize weights in this group
                    float sum_abs_original = 0.0f;
                    float sum_abs_quantized = 0.0f;

                    for (size_t i = group_start; i < group_end; ++i)
                    {
                        const float w = weights[i];
                        sum_abs_original += std::abs(w);

                        if (std::abs(w) > threshold)
                        {
                            result.values[i] = (w > 0) ? 1 : -1;
                            sum_abs_quantized += 1.0f;
                        }
                        else
                        {
                            result.values[i] = 0;
                        }
                    }

                    // Compute scale factor for this group
                    result.scales[g] = (sum_abs_quantized > 0)
                                           ? (sum_abs_original / sum_abs_quantized)
                                           : 1.0f;
                }
            }
            else
            {
                // Per-layer quantization
                // Compute mean absolute value across entire weight matrix
                float mean_abs = 0.0f;
                for (size_t i = 0; i < total_size; ++i)
                {
                    mean_abs += std::abs(weights[i]);
                }
                mean_abs /= total_size;

                const float threshold = 0.7f * mean_abs;

                // Quantize all weights
                float sum_abs_original = 0.0f;
                float sum_abs_quantized = 0.0f;

                for (size_t i = 0; i < total_size; ++i)
                {
                    const float w = weights[i];
                    sum_abs_original += std::abs(w);

                    if (std::abs(w) > threshold)
                    {
                        result.values[i] = (w > 0) ? 1 : -1;
                        sum_abs_quantized += 1.0f;
                    }
                    else
                    {
                        result.values[i] = 0;
                    }
                }

                // Compute single scale factor for the layer
                result.scales[0] = (sum_abs_quantized > 0)
                                       ? (sum_abs_original / sum_abs_quantized)
                                       : 1.0f;
            }

            return result;
        }

        QuantizedActivation quantize_activations_int8_scalar(
            const float *activations,
            size_t size,
            const QuantConfig &config)
        {
            std::cout << "DEBUG: quantize_activations_int8_scalar called with size=" << size << std::endl;
            QuantizedActivation result(size);

            // Find min/max for symmetric quantization
            float min_val = *std::min_element(activations, activations + size);
            float max_val = *std::max_element(activations, activations + size);

            // Symmetric quantization range
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));

            // Apply clipping if specified
            if (config.activation_clip_value > 0.0f)
            {
                abs_max = std::min(abs_max, config.activation_clip_value);
            }

            // Quantization parameters
            const float scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
            const int8_t zero_point = 0; // Symmetric quantization

            result.scale = scale;
            result.zero_point = zero_point;

            // Quantize activations
            for (size_t i = 0; i < size; ++i)
            {
                float clamped = std::max(-abs_max, std::min(abs_max, activations[i]));
                float quantized = clamped * scale;
                result.values[i] = static_cast<int8_t>(std::round(quantized));
            }

            return result;
        }

        void dequantize_weights_scalar(
            const TernaryWeight &weights,
            float *output,
            uint32_t rows,
            uint32_t cols)
        {
            std::cout << "DEBUG: dequantize_weights_scalar called with rows=" << rows << ", cols=" << cols << std::endl;
            const size_t total_size = static_cast<size_t>(rows) * cols;

            for (size_t i = 0; i < total_size; ++i)
            {
                const int8_t ternary_val = weights.values[i];
                const float scale = weights.get_scale(i);
                output[i] = static_cast<float>(ternary_val) * scale;
            }
        }

        void dequantize_activations_scalar(
            const QuantizedActivation &activations,
            float *output,
            size_t size)
        {
            for (size_t i = 0; i < size; ++i)
            {
                const int8_t quantized_val = activations.values[i];
                const float scale = activations.scale;
                const int8_t zero_point = activations.zero_point;
                output[i] = (static_cast<float>(quantized_val) - zero_point) / scale;
            }
        }

        float compute_quantization_error_scalar(
            const float *original,
            const float *dequantized,
            size_t size)
        {
            float mse = 0.0f;
            for (size_t i = 0; i < size; ++i)
            {
                float diff = original[i] - dequantized[i];
                mse += diff * diff;
            }
            return mse / size;
        }

    } // namespace bitnet
} // namespace ryzen_llm
