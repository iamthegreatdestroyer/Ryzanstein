use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// ann-hybrid: Unified sub-linear search engine.
#[derive(Parser)]
#[command(name = "ann-hybrid", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an index from a JSONL corpus.
    Index {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Search an existing index.
    Search {
        #[arg(short, long)]
        index: PathBuf,
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "10")]
        top_k: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { input, output } => {
            println!("Building index from {} → {}", input.display(), output.display());
            println!("(index build not yet implemented in CLI)");
        }
        Commands::Search { index, query, top_k } => {
            println!("Searching {} for \"{}\" (top {})", index.display(), query, top_k);
            println!("(CLI search not yet implemented)");
        }
    }
}
