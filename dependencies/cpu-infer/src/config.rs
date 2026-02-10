use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInferConfig {
    pub ryzanstein_url: String,
    pub num_threads: Option<usize>,
    pub max_batch_size: usize,
    pub use_simd: bool,
    pub default_max_tokens: usize,
    pub default_temperature: f32,
}

impl Default for CpuInferConfig {
    fn default() -> Self {
        CpuInferConfig {
            ryzanstein_url: "http://localhost:8000".into(),
            num_threads: None, // auto-detect
            max_batch_size: 32,
            use_simd: true,
            default_max_tokens: 256,
            default_temperature: 0.7,
        }
    }
}
