//! # σ-index — Succinct Semantic Code Index
//!
//! Combines an FM-index built on wavelet trees with ΣLANG-compressed code
//! embeddings to create a code search index occupying space comparable to
//! gzip-compressed source while supporting O(m) pattern matching.
//!
//! ## Architecture
//! - **FM-Index Layer**: Structural/textual search via wavelet tree BWT
//! - **HNSW Layer**: Semantic search via ΣLANG meaning vectors
//! - **Incremental Update**: tree-sitter incremental parsing for live updates
//!
//! ## Complexity Guarantees
//! - Pattern matching: O(m) where m = query length
//! - Semantic search: O(log n) via HNSW
//! - Index size: ~12-15 MB for 500K-line codebase

pub mod fm_index;
pub mod hnsw_layer;
pub mod incremental;
pub mod query;
pub mod ryzanstein_integration;
pub mod config;
pub mod error;

pub use config::IndexConfig;
pub use error::{IndexError, IndexResult};
pub use query::{SearchQuery, SearchResult, SearchMode};

use std::path::PathBuf;

/// Main entry point for the σ-index system
pub struct SigmaIndex {
    config: IndexConfig,
    fm_index: fm_index::FMIndex,
    hnsw: hnsw_layer::HNSWLayer,
    file_registry: Vec<IndexedFile>,
}

/// Metadata for an indexed file
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexedFile {
    pub path: PathBuf,
    pub hash: String,
    pub size_bytes: u64,
    pub language: Language,
    pub last_indexed: u64,
}

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Language {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Go,
    C,
    Cpp,
    Java,
    Unknown,
}

/// Index statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStats {
    pub total_files: usize,
    pub total_lines: usize,
    pub index_size_bytes: u64,
    pub original_size_bytes: u64,
    pub compression_ratio: f64,
    pub languages: std::collections::HashMap<Language, usize>,
    pub last_updated: u64,
}

impl SigmaIndex {
    /// Create a new σ-index with given configuration
    pub fn new(config: IndexConfig) -> IndexResult<Self> {
        Ok(Self {
            fm_index: fm_index::FMIndex::new(config.fm_index_config.clone())?,
            hnsw: hnsw_layer::HNSWLayer::new(config.hnsw_config.clone())?,
            file_registry: Vec::new(),
            config,
        })
    }

    /// Index an entire directory recursively
    pub async fn index_directory(&mut self, path: &std::path::Path) -> IndexResult<IndexStats> {
        let files = self.discover_files(path)?;
        for file in &files {
            self.index_file(file).await?;
        }
        Ok(self.stats())
    }

    /// Index a single file
    pub async fn index_file(&mut self, path: &std::path::Path) -> IndexResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| IndexError::IoError(e.to_string()))?;
        let language = Language::detect(path);

        // Parse with tree-sitter for structural tokens
        let tokens = incremental::parse_file(&content, language)?;

        // Update FM-index with structural tokens
        self.fm_index.add_document(path, &tokens)?;

        // Generate and store semantic embeddings via Ryzanstein
        let embeddings = self.generate_embeddings(&content, &tokens).await?;
        self.hnsw.add_embeddings(path, &embeddings)?;

        self.file_registry.push(IndexedFile {
            path: path.to_path_buf(),
            hash: Self::hash_content(&content),
            size_bytes: content.len() as u64,
            language,
            last_indexed: Self::now(),
        });

        Ok(())
    }

    /// Search the index with a query
    pub async fn search(&self, query: &SearchQuery) -> IndexResult<Vec<SearchResult>> {
        match query.mode {
            SearchMode::Exact => self.fm_index.search(&query.pattern),
            SearchMode::Semantic => self.hnsw.search(&query.pattern, query.top_k).await,
            SearchMode::Hybrid => {
                let exact = self.fm_index.search(&query.pattern)?;
                let semantic = self.hnsw.search(&query.pattern, query.top_k).await?;
                Ok(Self::merge_results(exact, semantic, query.top_k))
            }
        }
    }

    /// Incrementally update the index when a file changes
    pub async fn update_file(&mut self, path: &std::path::Path) -> IndexResult<()> {
        self.remove_file(path)?;
        self.index_file(path).await
    }

    /// Remove a file from the index
    pub fn remove_file(&mut self, path: &std::path::Path) -> IndexResult<()> {
        self.fm_index.remove_document(path)?;
        self.hnsw.remove_document(path)?;
        self.file_registry.retain(|f| f.path != path);
        Ok(())
    }

    /// Get current index statistics
    pub fn stats(&self) -> IndexStats {
        let total_lines: usize = self.file_registry.iter()
            .map(|f| f.size_bytes as usize / 40) // rough estimate
            .sum();
        let original_size: u64 = self.file_registry.iter()
            .map(|f| f.size_bytes)
            .sum();
        let index_size = self.fm_index.size_bytes() + self.hnsw.size_bytes();
        let mut languages = std::collections::HashMap::new();
        for file in &self.file_registry {
            *languages.entry(file.language).or_insert(0) += 1;
        }

        IndexStats {
            total_files: self.file_registry.len(),
            total_lines,
            index_size_bytes: index_size,
            original_size_bytes: original_size,
            compression_ratio: if index_size > 0 {
                original_size as f64 / index_size as f64
            } else {
                0.0
            },
            languages,
            last_updated: Self::now(),
        }
    }

    // -- Private helpers --

    fn discover_files(&self, path: &std::path::Path) -> IndexResult<Vec<PathBuf>> {
        let mut files = Vec::new();
        if path.is_file() {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            for entry in walkdir(path)? {
                if Language::detect(&entry) != Language::Unknown {
                    files.push(entry);
                }
            }
        }
        Ok(files)
    }

    async fn generate_embeddings(
        &self,
        content: &str,
        tokens: &[incremental::Token],
    ) -> IndexResult<Vec<f32>> {
        // Delegate to Ryzanstein ΣLANG encoder (mocked in tests)
        ryzanstein_integration::encode_semantic(content, tokens).await
    }

    fn merge_results(
        exact: Vec<SearchResult>,
        semantic: Vec<SearchResult>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut combined = exact;
        combined.extend(semantic);
        combined.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        combined.dedup_by(|a, b| a.file_path == b.file_path && a.line == b.line);
        combined.truncate(top_k);
        combined
    }

    fn hash_content(content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl Language {
    pub fn detect(path: &std::path::Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("py") => Language::Python,
            Some("rs") => Language::Rust,
            Some("js") => Language::JavaScript,
            Some("ts") | Some("tsx") => Language::TypeScript,
            Some("go") => Language::Go,
            Some("c") | Some("h") => Language::C,
            Some("cpp") | Some("cc") | Some("hpp") => Language::Cpp,
            Some("java") => Language::Java,
            _ => Language::Unknown,
        }
    }
}

fn walkdir(path: &std::path::Path) -> IndexResult<Vec<PathBuf>> {
    let mut results = Vec::new();
    fn walk_recursive(dir: &std::path::Path, results: &mut Vec<PathBuf>) -> IndexResult<()> {
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir)
                .map_err(|e| IndexError::IoError(e.to_string()))? {
                let entry = entry.map_err(|e| IndexError::IoError(e.to_string()))?;
                let path = entry.path();
                if path.is_dir() {
                    walk_recursive(&path, results)?;
                } else {
                    results.push(path);
                }
            }
        }
        Ok(())
    }
    walk_recursive(path, &mut results)?;
    Ok(results)
}
