//! σ-index CLI tool

use clap::Parser;
use sigma_index::{IndexConfig, SigmaIndex, SearchQuery};

#[derive(Parser, Debug)]
#[command(name = "sigma-index", about = "Succinct semantic code index", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Build index for a directory
    Build {
        #[arg(short, long)]
        path: String,
        #[arg(short, long, default_value = "sigma-index.db")]
        output: String,
    },
    /// Search the index
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "hybrid")]
        mode: String,
        #[arg(short, long, default_value = "sigma-index.db")]
        index: String,
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
    },
    /// Show index statistics
    Stats {
        #[arg(short, long, default_value = "sigma-index.db")]
        index: String,
    },
    /// Watch directory for changes
    Watch {
        #[arg(short, long)]
        path: String,
        #[arg(short, long, default_value = "sigma-index.db")]
        index: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { path, output: _ } => {
            let config = IndexConfig::default();
            let mut index = SigmaIndex::new(config)?;
            let stats = index.index_directory(std::path::Path::new(&path)).await?;
            println!("Indexed {} files ({} lines)", stats.total_files, stats.total_lines);
            println!("Index size: {} bytes (compression ratio: {:.2}x)",
                stats.index_size_bytes, stats.compression_ratio);
        }
        Commands::Search { query, mode, index: _, top_k } => {
            let search_mode = match mode.as_str() {
                "exact" => sigma_index::query::SearchMode::Exact,
                "semantic" => sigma_index::query::SearchMode::Semantic,
                _ => sigma_index::query::SearchMode::Hybrid,
            };
            let config = IndexConfig::default();
            let idx = SigmaIndex::new(config)?;
            let q = SearchQuery { pattern: query, mode: search_mode, top_k, file_filter: None, language_filter: None };
            let results = idx.search(&q).await?;
            for (i, r) in results.iter().enumerate() {
                println!("{}. {}:{} (score: {:.4}) - {}", i + 1, r.file_path.display(), r.line, r.score, r.snippet);
            }
        }
        Commands::Stats { index: _ } => {
            let config = IndexConfig::default();
            let idx = SigmaIndex::new(config)?;
            let stats = idx.stats();
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        Commands::Watch { path: _, index: _ } => {
            println!("File watching mode not yet implemented (requires notify integration)");
        }
    }

    Ok(())
}
