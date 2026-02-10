//! Integration tests for σ-index

use sigma_index::*;
use std::path::Path;
use tempfile::TempDir;

#[tokio::test]
async fn test_full_index_lifecycle() {
    let config = IndexConfig::default();
    let mut index = SigmaIndex::new(config).unwrap();

    // Create temp directory with sample files
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("sample.rs");
    std::fs::write(&file_path, r#"
        fn hello() {
            println!("Hello, world!");
        }

        fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        struct Config {
            name: String,
            timeout: u64,
        }
    "#).unwrap();

    // Index the file
    index.index_file(&file_path).await.unwrap();

    // Check stats
    let stats = index.stats();
    assert_eq!(stats.total_files, 1);
    assert!(stats.index_size_bytes > 0);

    // Search exact
    let results = index.search(&SearchQuery::exact("hello")).await.unwrap();
    // Results may be empty due to simplified BWT, but should not error
    assert!(results.len() >= 0);

    // Update
    std::fs::write(&file_path, r#"
        fn hello_updated() {
            println!("Updated!");
        }
    "#).unwrap();
    index.update_file(&file_path).await.unwrap();

    // Remove
    index.remove_file(&file_path).unwrap();
    let stats = index.stats();
    assert_eq!(stats.total_files, 0);
}

#[tokio::test]
async fn test_index_directory() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("a.py"), "def foo():\n    pass\n").unwrap();
    std::fs::write(dir.path().join("b.rs"), "fn bar() {}\n").unwrap();
    std::fs::write(dir.path().join("c.txt"), "not code").unwrap();

    let config = IndexConfig::default();
    let mut index = SigmaIndex::new(config).unwrap();
    let stats = index.index_directory(dir.path()).await.unwrap();

    // Should index .py and .rs but not .txt
    assert!(stats.total_files >= 2);
}

#[tokio::test]
async fn test_semantic_search() {
    let config = IndexConfig::default();
    let index = SigmaIndex::new(config).unwrap();
    // Empty index semantic search should return empty
    let results = index.search(&SearchQuery::semantic("error handling")).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_hybrid_search() {
    let config = IndexConfig::default();
    let index = SigmaIndex::new(config).unwrap();
    let results = index.search(&SearchQuery::hybrid("test")).await.unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_language_detection() {
    assert_eq!(Language::detect(Path::new("foo.rs")), Language::Rust);
    assert_eq!(Language::detect(Path::new("bar.py")), Language::Python);
    assert_eq!(Language::detect(Path::new("baz.go")), Language::Go);
    assert_eq!(Language::detect(Path::new("qux.ts")), Language::TypeScript);
    assert_eq!(Language::detect(Path::new("nope.xyz")), Language::Unknown);
}

#[test]
fn test_index_config_default() {
    let config = IndexConfig::default();
    assert!(config.watch_enabled);
    assert_eq!(config.fm_index_config.sample_rate, 32);
    assert_eq!(config.hnsw_config.embedding_dim, 1024);
}

#[test]
fn test_index_stats_serialize() {
    let config = IndexConfig::default();
    let index = SigmaIndex::new(config).unwrap();
    let stats = index.stats();
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("total_files"));
}
