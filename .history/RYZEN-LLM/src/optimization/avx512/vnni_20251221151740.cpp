// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// VNNI INT8 Operations
// [REF:OL-005b] - Optimization Layer: AVX-512 SIMD Primitives
//
// This file implements INT8 quantized operations using AVX-512 VNNI
// (Vector Neural Network Instructions) for accelerated inference.
//
// Key Features:
// - INT8 dot products with VNNI
// - Accumulation into INT32
// - Dequantization to FP32
// - Per-tensor and per-channel quantization

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace ryzen_llm
{
    namespace avx512
    {

        class VNNI
        {
        public:
            VNNI() = default;
            ~VNNI() = default;

            /**
             * AVX-512 VNNI INT8 Dot Product
             * Computes dot product of two INT8 vectors using VNNI instructions
             * Result accumulates into INT32 for precision
             */
            void DotProductINT8(const int8_t *a, const int8_t *b, int32_t *result, size_t size)
            {
                __m512i sum = _mm512_setzero_si512();

                for (size_t i = 0; i < size; i += 64)
                { // Process 64 elements at a time
                    // Load 64 INT8 values (512 bits)
                    __m512i va = _mm512_loadu_si512((__m512i *)&a[i]);
                    __m512i vb = _mm512_loadu_si512((__m512i *)&b[i]);

                    // VNNI dot product: va * vb -> accumulate into INT32
                    __m512i prod = _mm512_dpbusds_epi32(sum, va, vb);
                    sum = _mm512_add_epi32(sum, prod);
                }

                // Horizontal sum of the 16 INT32 values in sum
                *result = _mm512_reduce_add_epi32(sum);
            }

            /**
             * Quantize FP32 to INT8 with per-tensor scaling
             */
            void QuantizeINT8(const float *input, int8_t *output, size_t size,
                              float scale, int8_t zero_point)
            {
                __m512 scale_vec = _mm512_set1_ps(scale);
                __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_point));

                for (size_t i = 0; i < size; i += 16)
                {
                    // Load 16 floats
                    __m512 x = _mm512_loadu_ps(&input[i]);

                    // Quantize: clamp(round(x / scale) + zero_point, -128, 127)
                    __m512 scaled = _mm512_div_ps(x, scale_vec);
                    __m512 rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                    __m512 shifted = _mm512_add_ps(rounded, zp_vec);

                    // Clamp to INT8 range
                    __m512 min_val = _mm512_set1_ps(-128.0f);
                    __m512 max_val = _mm512_set1_ps(127.0f);
                    __m512 clamped = _mm512_min_ps(_mm512_max_ps(shifted, min_val), max_val);

                    // Convert to INT8 and store
                    __m512i int_vals = _mm512_cvtps_epi32(clamped);
                    __m128i packed = _mm512_cvtepi32_epi8(int_vals);
                    _mm_storeu_si128((__m128i *)&output[i], packed);
                }
            }

            /**
             * Dequantize INT32 accumulator to FP32
             */
            void DequantizeFP32(const int32_t *input, float *output, size_t size, float scale)
            {
                __m512 scale_vec = _mm512_set1_ps(scale);

                for (size_t i = 0; i < size; i += 16)
                {
                    // Load 16 INT32 values
                    __m512i int_vals = _mm512_loadu_si512((__m512i *)&input[i]);

                    // Convert to FP32 and scale
                    __m512 float_vals = _mm512_cvtepi32_ps(int_vals);
                    __m512 result = _mm512_mul_ps(float_vals, scale_vec);

                    // Store result
                    _mm512_storeu_ps(&output[i], result);
                }
            }

            /**
             * Per-channel quantization for convolutional layers
             */
            void PerChannelQuantize(const float *input, int8_t *output,
                                    const float *scales, const int8_t *zero_points,
                                    size_t channels, size_t size_per_channel)
            {

                for (size_t ch = 0; ch < channels; ++ch)
                {
                    __m512 scale_vec = _mm512_set1_ps(scales[ch]);
                    __m512 zp_vec = _mm512_set1_ps(static_cast<float>(zero_points[ch]));

                    const float *ch_input = &input[ch * size_per_channel];
                    int8_t *ch_output = &output[ch * size_per_channel];

                    for (size_t i = 0; i < size_per_channel; i += 16)
                    {
                        // Load 16 floats for this channel
                        __m512 x = _mm512_loadu_ps(&ch_input[i]);

                        // Quantize with channel-specific scale/zp
                        __m512 scaled = _mm512_div_ps(x, scale_vec);
                        __m512 rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
                        __m512 shifted = _mm512_add_ps(rounded, zp_vec);

                        // Clamp and convert
                        __m512 min_val = _mm512_set1_ps(-128.0f);
                        __m512 max_val = _mm512_set1_ps(127.0f);
                        __m512 clamped = _mm512_min_ps(_mm512_max_ps(shifted, min_val), max_val);

                        __m512i int_vals = _mm512_cvtps_epi32(clamped);
                        __m128i packed = _mm512_cvtepi32_epi8(int_vals);
                        _mm_storeu_si128((__m128i *)&ch_output[i], packed);
                    }
                }
            }

            /**
             * Mixed precision matrix multiplication: INT8 weights × FP16 activations
             */
            void MixedPrecisionMatMul(const int8_t *weights, const _Float16 *activations,
                                      float *output, size_t M, size_t N, size_t K,
                                      const float *weight_scales, const float *act_scales)
            {

                for (size_t m = 0; m < M; ++m)
                {
                    for (size_t n = 0; n < N; ++n)
                    {
                        __m512i sum = _mm512_setzero_si512();

                        for (size_t k = 0; k < K; k += 64)
                        {
                            // Load INT8 weights and convert to INT16 for VNNI
                            __m512i w_int8 = _mm512_loadu_si512((__m512i *)&weights[m * K + k]);
                            __m512i w_int16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(w_int8, 0));

                            // Load FP16 activations and convert to INT8
                            __m256i act_fp16 = _mm256_loadu_si256((__m256i *)&activations[n * K + k]);
                            __m512i act_int8 = _mm512_cvtepi16_epi8(_mm512_cvtepu16_epi32(act_fp16));

                            // VNNI dot product
                            sum = _mm512_dpbusds_epi32(sum, w_int16, act_int8);
                        }

                        // Dequantize and store
                        __m512 sum_fp32 = _mm512_cvtepi32_ps(sum);
                        __m512 scale = _mm512_mul_ps(_mm512_set1_ps(weight_scales[m]),
                                                     _mm512_set1_ps(act_scales[n]));
                        __m512 result = _mm512_mul_ps(sum_fp32, scale);

                        _mm512_storeu_ps(&output[m * N + n], result);
                    }
                }
            }
        };

    private:
        // TODO: Add quantization parameters and FP16 support
    };

} // namespace avx512
} // namespace ryzen_llm
