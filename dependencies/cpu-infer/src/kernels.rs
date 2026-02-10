//! SIMD-accelerated compute kernels for CPU inference.

/// Dot product of two f32 slices — auto-vectorized by LLVM.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Softmax in-place over a mutable f32 slice.
#[inline]
pub fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// ReLU activation in-place.
#[inline]
pub fn relu(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.max(0.0);
    }
}

/// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(data: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.7978845608;
    for v in data.iter_mut() {
        let x = *v;
        let inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
        *v = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Stub batch matmul: generates `max_tokens` output values from input.
pub fn batch_matmul(input: &[f32], max_tokens: usize) -> Vec<f32> {
    use rayon::prelude::*;
    (0..max_tokens)
        .into_par_iter()
        .map(|i| {
            let offset = (i % input.len().max(1)) as f32;
            (offset * 0.1).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_empty() {
        assert!((dot_product(&[], &[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_relu() {
        let mut data = vec![-1.0, 0.0, 1.0, -0.5, 2.0];
        relu(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_gelu_zero() {
        let mut data = vec![0.0];
        gelu(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_matmul() {
        let input = vec![1.0, 2.0, 3.0];
        let out = batch_matmul(&input, 10);
        assert_eq!(out.len(), 10);
    }
}
