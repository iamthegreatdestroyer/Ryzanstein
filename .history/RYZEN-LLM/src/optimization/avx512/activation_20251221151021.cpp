// Copyright (c) 2024 RYZEN-LLM Project
// Licensed under MIT License
//
// AVX-512 SIMD Activation Functions
// [REF:OL-005b] - Optimization Layer: AVX-512 SIMD Primitives
//
// This file implements vectorized activation functions (GELU, SiLU, etc.)
// using AVX-512 for maximum throughput.
//
// Key Features:
// - GELU (Gaussian Error Linear Unit)
// - SiLU/Swish activation
// - ReLU and variants
// - Softmax with numerical stability

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ryzen_llm
{
    namespace avx512
    {

        class Activation
        {
        public:
            Activation() = default;
            ~Activation() = default;

            /**
             * AVX-512 GELU Activation (Tanh approximation)
             * GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
             */
            void GELU(const float *input, float *output, size_t size)
            {
                const __m512 sqrt_2_pi = _mm512_set1_ps(0.7978845608028654f); // √(2/π)
                const __m512 coeff = _mm512_set1_ps(0.044715f);

                for (size_t i = 0; i < size; i += 16)
                {
                    // Load 16 floats
                    __m512 x = _mm512_loadu_ps(&input[i]);

                    // Compute x³
                    __m512 x2 = _mm512_mul_ps(x, x);
                    __m512 x3 = _mm512_mul_ps(x2, x);

                    // Compute inner = √(2/π) * (x + 0.044715 * x³)
                    __m512 inner = _mm512_fmadd_ps(coeff, x3, x);
                    inner = _mm512_mul_ps(sqrt_2_pi, inner);

                    // Compute tanh(inner) using approximation
                    __m512 tanh_inner = tanh_avx512(inner);

                    // Compute GELU: 0.5 * x * (1 + tanh(inner))
                    __m512 one = _mm512_set1_ps(1.0f);
                    __m512 half = _mm512_set1_ps(0.5f);
                    __m512 result = _mm512_add_ps(one, tanh_inner);
                    result = _mm512_mul_ps(result, x);
                    result = _mm512_mul_ps(result, half);

                    // Store result
                    _mm512_storeu_ps(&output[i], result);
                }
            }

            /**
             * AVX-512 SiLU/Swish Activation
             * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
             */
            void SiLU(const float *input, float *output, size_t size)
            {
                for (size_t i = 0; i < size; i += 16)
                {
                    // Load 16 floats
                    __m512 x = _mm512_loadu_ps(&input[i]);

                    // Compute sigmoid(x) = 1 / (1 + exp(-x))
                    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
                    __m512 exp_neg_x = exp_avx512(neg_x);
                    __m512 one = _mm512_set1_ps(1.0f);
                    __m512 sigmoid = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));

                    // Compute SiLU: x * sigmoid(x)
                    __m512 result = _mm512_mul_ps(x, sigmoid);

                    // Store result
                    _mm512_storeu_ps(&output[i], result);
                }
            }

            /**
             * AVX-512 ReLU Activation
             * ReLU(x) = max(0, x)
             */
            void ReLU(const float *input, float *output, size_t size)
            {
                __m512 zero = _mm512_setzero_ps();

                for (size_t i = 0; i < size; i += 16)
                {
                    __m512 x = _mm512_loadu_ps(&input[i]);
                    __m512 result = _mm512_max_ps(x, zero);
                    _mm512_storeu_ps(&output[i], result);
                }
            }

            /**
             * AVX-512 Leaky ReLU
             * LeakyReLU(x) = x if x > 0 else alpha * x
             */
            void LeakyReLU(const float *input, float *output, size_t size, float alpha = 0.01f)
            {
                __m512 zero = _mm512_setzero_ps();
                __m512 alpha_vec = _mm512_set1_ps(alpha);

                for (size_t i = 0; i < size; i += 16)
                {
                    __m512 x = _mm512_loadu_ps(&input[i]);
                    __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
                    __m512 positive = x;
                    __m512 negative = _mm512_mul_ps(x, alpha_vec);
                    __m512 result = _mm512_mask_blend_ps(mask, negative, positive);
                    _mm512_storeu_ps(&output[i], result);
                }
            }

            /**
             * AVX-512 Numerically Stable Softmax
             */
            void Softmax(const float *input, float *output, size_t size)
            {
                // Find max value for numerical stability
                float max_val = *std::max_element(input, input + size);
                __m512 max_vec = _mm512_set1_ps(max_val);

                // Compute exp(x - max) and sum
                __m512 sum_exp = _mm512_setzero_ps();
                for (size_t i = 0; i < size; i += 16)
                {
                    __m512 x = _mm512_loadu_ps(&input[i]);
                    __m512 shifted = _mm512_sub_ps(x, max_vec);
                    __m512 exp_vals = exp_avx512(shifted);
                    sum_exp = _mm512_add_ps(sum_exp, exp_vals);
                    _mm512_storeu_ps(&output[i], exp_vals);
                }

                // Horizontal sum of sum_exp
                float total_sum = _mm512_reduce_add_ps(sum_exp);
                __m512 sum_vec = _mm512_set1_ps(total_sum);

                // Normalize
                for (size_t i = 0; i < size; i += 16)
                {
                    __m512 exp_vals = _mm512_loadu_ps(&output[i]);
                    __m512 normalized = _mm512_div_ps(exp_vals, sum_vec);
                    _mm512_storeu_ps(&output[i], normalized);
                }
            }
        };

    private:
            /**
             * AVX-512 tanh approximation
             * tanh(x) ≈ x * (27 + x²) / (27 + 9*x²) for |x| < 3
             */
            __m512 tanh_avx512(__m512 x)
            {
                __m512 x2 = _mm512_mul_ps(x, x);
                __m512 nine = _mm512_set1_ps(9.0f);
                __m512 twenty_seven = _mm512_set1_ps(27.0f);

                __m512 numerator = _mm512_fmadd_ps(x2, nine, twenty_seven);
                numerator = _mm512_fmadd_ps(x, numerator, x2);

                __m512 denominator = _mm512_fmadd_ps(x2, nine, twenty_seven);

                return _mm512_div_ps(numerator, denominator);
            }

            /**
             * AVX-512 exp approximation using Taylor series
             * exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! for |x| < 1
             */
            __m512 exp_avx512(__m512 x)
            {
                __m512 one = _mm512_set1_ps(1.0f);
                __m512 result = one;

                // x
                result = _mm512_add_ps(result, x);

                // x²/2!
                __m512 x2 = _mm512_mul_ps(x, x);
                __m512 term = _mm512_div_ps(x2, _mm512_set1_ps(2.0f));
                result = _mm512_add_ps(result, term);

                // x³/3!
                __m512 x3 = _mm512_mul_ps(x2, x);
                term = _mm512_div_ps(x3, _mm512_set1_ps(6.0f));
                result = _mm512_add_ps(result, term);

                // x⁴/4!
                __m512 x4 = _mm512_mul_ps(x3, x);
                term = _mm512_div_ps(x4, _mm512_set1_ps(24.0f));
                result = _mm512_add_ps(result, term);

                return result;
            }
        };

    private:
        // TODO: Add constant tables for approximations
    };

} // namespace avx512
} // namespace ryzen_llm
