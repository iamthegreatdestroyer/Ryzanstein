use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod compress;
mod drain;
mod pattern;
mod query;
mod semantic;

/// semlog — Semantic Log Compression
#[derive(Parser)]
#[command(name = "semlog", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a log file into .semlog format
    Compress {
        /// Input log file
        #[arg(short, long)]
        input: PathBuf,
        /// Output .semlog file
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Query a compressed .semlog file
    Query {
        /// The .semlog file to query
        #[arg(short, long)]
        file: PathBuf,
        /// Query expression
        query: String,
    },
    /// Stream-compress from stdin
    Stream {
        /// Output .semlog file
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output } => {
            println!("Compressing {} → {}", input.display(), output.display());
            match compress::compress_file(&input, &output) {
                Ok(stats) => println!(
                    "Done: {} lines, {:.1}× compression",
                    stats.lines_processed, stats.compression_ratio
                ),
                Err(e) => eprintln!("Error: {e}"),
            }
        }
        Commands::Query { file, query: q } => {
            println!("Querying {} for: {}", file.display(), q);
            match query::query_file(&file, &q) {
                Ok(results) => {
                    for r in &results {
                        println!("{}", r);
                    }
                    println!("({} results)", results.len());
                }
                Err(e) => eprintln!("Error: {e}"),
            }
        }
        Commands::Stream { output } => {
            println!("Streaming to {}", output.display());
            if let Err(e) = compress::stream_compress(&output) {
                eprintln!("Error: {e}");
            }
        }
    }
}
