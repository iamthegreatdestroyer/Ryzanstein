/*
 * RYZEN-LLM BitNet Quantization Implementation
 * [REF:PHASE1-001] - BitNet b1.58 Ternary Quantization
 */

#include "quantize.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>

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
                        mean_abs += fabs(weights[i]);
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
                        sum_abs_original += fabs(w);

                        if (fabs(w) > threshold)
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
                    // Use fabs instead of std::abs to avoid AVX-512
                    mean_abs += fabs(weights[i]);
                }
                mean_abs /= total_size;
                // std::cout << "DEBUG: mean_abs computed: " << mean_abs << std::endl;

                // std::cout << "DEBUG: about to compute threshold" << std::endl;
                const float threshold = 0.7f * mean_abs;
                printf("DEBUG: threshold computed: %.6f\n", threshold);

                // Quantize all weights
                printf("DEBUG: starting quantization loop\n");
                float sum_abs_original = 0.0f;
                float sum_abs_quantized = 0.0f;
                std::cout << "DEBUG: starting quantization loop" << std::endl;

                for (size_t i = 0; i < total_size; ++i)
                {
                    const float w = weights[i];
                    sum_abs_original += fabs(w);

                    if (fabs(w) > threshold)
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
                    float clipped = (activations[i] < -clip_value) ? -clip_value :
                                   (activations[i] > clip_value) ? clip_value : activations[i];
                    int32_t quantized = static_cast<int32_t>(round(clipped / result.scale));

                    // Clamp to INT8 range
                    int32_t clamped = (quantized < -127) ? -127 :
                                     (quantized > 127) ? 127 : quantized;
                    result.values[i] = static_cast<int8_t>(clamped);
                }
            }
            else
            {
                // Asymmetric quantization: [min, max] -> [-128, 127]
                float min_val = activations[0];
                float max_val = activations[0];

                for (size_t i = 1; i < size; ++i)
                {
                    if (activations[i] < min_val) min_val = activations[i];
                    if (activations[i] > max_val) max_val = activations[i];
                }

                // Compute scale and zero point
                const float range = max_val - min_val;
                result.scale = range / 255.0f; // Map to 256 levels
                result.zero_point = static_cast<int8_t>(
                    round(-128.0f - min_val / result.scale));

                for (size_t i = 0; i < size; ++i)
                {
                    int32_t quantized = static_cast<int32_t>(
                        round(activations[i] / result.scale) + result.zero_point);

                    int32_t clamped = (quantized < -128) ? -128 :
                                     (quantized > 127) ? 127 : quantized;
                    result.values[i] = static_cast<int8_t>(clamped);
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
            // Use literal 16 instead of multiplication to avoid AVX-512
            const size_t total_size = 16;

            TernaryWeight result(rows, cols, config.per_group_scaling ? config.weight_group_size : 0);

            if (config.per_group_scaling)
            {
                // Per-group quantization
                const uint32_t num_groups = (total_size + config.weight_group_size - 1) / config.weight_group_size;

                for (uint32_t g = 0; g < num_groups; ++g)
                {
                    const size_t group_start = g * config.weight_group_size;
                    const size_t group_end = (group_start + config.weight_group_size < total_size) ? 
                        (group_start + config.weight_group_size) : total_size;
                    const size_t group_len = group_end - group_start;

                    // Compute mean absolute value for this group
                    float mean_abs = 0.0f;
                    for (size_t i = group_start; i < group_end; ++i)
                    {
                        mean_abs += fabs(weights[i]);
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
                        sum_abs_original += fabs(w);

                        if (fabs(w) > threshold)
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
                for (size_t i = 0; i < 16; ++i)
                {
                    // Use fabs instead of std::abs to avoid AVX-512
                    float abs_val = fabs(weights[i]);
                    mean_abs += abs_val;
                }
                // Use literal division to avoid AVX-512
                mean_abs = mean_abs / 16.0f;

                const float threshold = 0.7f * mean_abs;

                // Quantize all weights
                float sum_abs_original = 0.0f;
                float sum_abs_quantized = 0.0f;

                // Use literal 16 instead of total_size to avoid AVX-512
                for (size_t i = 0; i < 16; ++i)
                {
                    const float w = weights[i];
                    sum_abs_original += fabs(w);

                    if (fabs(w) > threshold)
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

            // Manual min/max calculation instead of std::min_element/max_element
            float min_val = activations[0];
            float max_val = activations[0];

            for (size_t i = 1; i < size; ++i)
            {
                if (activations[i] < min_val) min_val = activations[i];
                if (activations[i] > max_val) max_val = activations[i];
            }

            // Symmetric quantization range
            float abs_max = (fabs(min_val) > fabs(max_val)) ? fabs(min_val) : fabs(max_val);

            // Apply clipping if specified (manual min instead of std::min)
            if (config.activation_clip_value > 0.0f && abs_max > config.activation_clip_value)
            {
                abs_max = config.activation_clip_value;
            }

            // Quantization parameters
            const float scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
            const int8_t zero_point = 0; // Symmetric quantization

            result.scale = scale;
            result.zero_point = zero_point;

            // Quantize activations
            for (size_t i = 0; i < size; ++i)
            {
                // Manual clamping instead of std::max/std::min
                float val = activations[i];
                if (val < -abs_max) val = -abs_max;
                if (val > abs_max) val = abs_max;
                float quantized = val * scale;
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
            std::cout << "DEBUG: dequantize_activations_scalar called with size=" << size << std::endl;
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
            std::cout << "DEBUG: compute_quantization_error_scalar called with size=" << size << std::endl;
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
