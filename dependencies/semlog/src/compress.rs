/// Compression and streaming entry points for semlog.
///
/// Orchestrates the Drain parser → Semantic classifier → Pattern detector → Storage pipeline.

use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use thiserror::Error;

use crate::drain::{DrainConfig, DrainParser};
use crate::pattern::{PatternConfig, PatternDetector};
use crate::semantic::classify_token;

#[derive(Error, Debug)]
pub enum CompressError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Statistics from a compression run.
#[derive(Debug, Clone)]
pub struct CompressStats {
    pub lines_processed: u64,
    pub templates_discovered: usize,
    pub patterns_detected: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub compression_ratio: f64,
}

/// Compress a log file into .semlog format.
pub fn compress_file(input: &Path, output: &Path) -> Result<CompressStats, CompressError> {
    let content = fs::read_to_string(input)?;
    let input_bytes = content.len() as u64;

    let mut parser = DrainParser::new(DrainConfig::default());
    let mut detector = PatternDetector::new(PatternConfig::default());

    let mut template_ids = Vec::new();
    let mut lines_processed: u64 = 0;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let tid = parser.parse_line(line);
        template_ids.push(tid);
        detector.feed(tid);
        lines_processed += 1;
    }

    let patterns = detector.finish();
    let templates = parser.templates();

    // Serialize compressed format
    let compressed = serde_json::json!({
        "version": "0.1.0",
        "templates": templates.iter().map(|t| serde_json::json!({
            "id": t.id,
            "pattern": t.to_string_repr(),
            "count": t.count,
        })).collect::<Vec<_>>(),
        "template_sequence": template_ids,
        "patterns": patterns.iter().map(|p| serde_json::json!({
            "kind": format!("{:?}", p.kind),
            "start_line": p.start_line,
            "end_line": p.end_line,
            "line_count": p.line_count,
        })).collect::<Vec<_>>(),
        "stats": {
            "lines": lines_processed,
            "templates": templates.len(),
            "patterns": patterns.len(),
        },
    });

    let output_data = serde_json::to_vec(&compressed)?;
    fs::write(output, &output_data)?;
    let output_bytes = output_data.len() as u64;

    Ok(CompressStats {
        lines_processed,
        templates_discovered: templates.len(),
        patterns_detected: patterns.len(),
        input_bytes,
        output_bytes,
        compression_ratio: if output_bytes > 0 {
            input_bytes as f64 / output_bytes as f64
        } else {
            0.0
        },
    })
}

/// Stream-compress from stdin to output file.
pub fn stream_compress(output: &Path) -> Result<(), CompressError> {
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    let mut parser = DrainParser::new(DrainConfig::default());
    let mut detector = PatternDetector::new(PatternConfig::default());
    let mut lines: u64 = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let tid = parser.parse_line(&line);
        detector.feed(tid);
        lines += 1;

        // Periodic flush every 10,000 lines
        if lines % 10_000 == 0 {
            eprintln!("[semlog] Processed {} lines, {} templates", lines, parser.templates().len());
        }
    }

    // Final output
    let patterns = detector.finish();
    let templates = parser.templates();

    let compressed = serde_json::json!({
        "version": "0.1.0",
        "templates": templates.iter().map(|t| serde_json::json!({
            "id": t.id,
            "pattern": t.to_string_repr(),
            "count": t.count,
        })).collect::<Vec<_>>(),
        "stats": {
            "lines": lines,
            "templates": templates.len(),
            "patterns": patterns.len(),
        },
    });

    let output_data = serde_json::to_vec_pretty(&compressed)?;
    fs::write(output, &output_data)?;
    eprintln!("[semlog] Done: {} lines → {} bytes", lines, output_data.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compress_file() {
        let mut input = NamedTempFile::new().unwrap();
        writeln!(input, "GET /api/users 200 42ms").unwrap();
        writeln!(input, "GET /api/items 200 35ms").unwrap();
        writeln!(input, "POST /api/orders 201 120ms").unwrap();
        writeln!(input, "GET /api/users 500 1ms").unwrap();

        let output = NamedTempFile::new().unwrap();

        let stats = compress_file(input.path(), output.path()).unwrap();
        assert_eq!(stats.lines_processed, 4);
        assert!(stats.templates_discovered > 0);
        assert!(stats.output_bytes > 0);
    }
}
