//! Quantization utilities for reduced-precision inference.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantFormat {
    /// Full 32-bit floating point.
    F32,
    /// 16-bit floating point (half precision).
    F16,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization (for 1.58-bit style).
    Int4,
    /// 1-bit ternary (BitNet style: -1, 0, +1).
    Ternary,
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub zero_point: i8,
    pub format: QuantFormat,
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Quantize f32 data to int8 using min-max scaling.
    pub fn quantize_int8(data: &[f32]) -> Self {
        if data.is_empty() {
            return QuantizedTensor {
                data: vec![],
                scale: 1.0,
                zero_point: 0,
                format: QuantFormat::Int8,
                shape: vec![0],
            };
        }
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-8);
        let scale = range / 255.0;
        let zero_point = (-min / scale).round().clamp(-128.0, 127.0) as i8;

        let quantized: Vec<i8> = data
            .iter()
            .map(|&v| {
                ((v / scale) + zero_point as f32)
                    .round()
                    .clamp(-128.0, 127.0) as i8
            })
            .collect();

        QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            format: QuantFormat::Int8,
            shape: vec![data.len()],
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&v| (v as f32 - self.zero_point as f32) * self.scale)
            .collect()
    }

    /// Quantize to ternary (-1, 0, +1) for BitNet-style inference.
    pub fn quantize_ternary(data: &[f32], threshold: f32) -> Self {
        let quantized: Vec<i8> = data
            .iter()
            .map(|&v| {
                if v > threshold {
                    1
                } else if v < -threshold {
                    -1
                } else {
                    0
                }
            })
            .collect();

        QuantizedTensor {
            data: quantized,
            scale: 1.0,
            zero_point: 0,
            format: QuantFormat::Ternary,
            shape: vec![data.len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_round_trip() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let q = QuantizedTensor::quantize_int8(&data);
        let deq = q.dequantize();
        for (orig, restored) in data.iter().zip(deq.iter()) {
            assert!(
                (orig - restored).abs() < 0.05,
                "orig={} restored={}",
                orig,
                restored
            );
        }
    }

    #[test]
    fn test_ternary_quantization() {
        let data = vec![0.5, -0.5, 0.01, -0.01, 1.0];
        let q = QuantizedTensor::quantize_ternary(&data, 0.1);
        assert_eq!(q.data, vec![1, -1, 0, 0, 1]);
    }

    #[test]
    fn test_empty_quantize() {
        let q = QuantizedTensor::quantize_int8(&[]);
        assert!(q.data.is_empty());
    }

    #[test]
    fn test_format_variants() {
        assert_ne!(QuantFormat::F32, QuantFormat::Int8);
        assert_eq!(QuantFormat::Ternary, QuantFormat::Ternary);
    }
}
