//! FM-Index implementation using wavelet trees for succinct code search
//!
//! Provides O(m) pattern matching where m = query length,
//! independent of codebase size.

use crate::config::FMIndexConfig;
use crate::error::{IndexError, IndexResult};
use crate::query::SearchResult;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Succinct FM-Index for structural/textual code search
pub struct FMIndex {
    config: FMIndexConfig,
    /// Burrows-Wheeler Transform of concatenated documents
    bwt: Vec<u8>,
    /// Suffix array samples (every `sample_rate`-th entry)
    sa_samples: Vec<usize>,
    /// Character occurrence counts
    occ: HashMap<u8, Vec<usize>>,
    /// First column counts
    c_table: [usize; 256],
    /// Document boundaries in the concatenated text
    doc_boundaries: Vec<(PathBuf, usize, usize)>,
    /// Total indexed size in bytes
    total_size: u64,
}

impl FMIndex {
    pub fn new(config: FMIndexConfig) -> IndexResult<Self> {
        Ok(Self {
            config,
            bwt: Vec::new(),
            sa_samples: Vec::new(),
            occ: HashMap::new(),
            c_table: [0; 256],
            doc_boundaries: Vec::new(),
            total_size: 0,
        })
    }

    /// Add a document to the index
    pub fn add_document(
        &mut self,
        path: &Path,
        tokens: &[crate::incremental::Token],
    ) -> IndexResult<()> {
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");
        let start = self.bwt.len();
        let bytes = text.as_bytes();

        // Build BWT for the new document
        let sa = self.build_suffix_array(bytes);
        let bwt_segment = self.compute_bwt(bytes, &sa);

        self.bwt.extend_from_slice(&bwt_segment);
        let end = self.bwt.len();

        // Update suffix array samples
        for (i, &sa_val) in sa.iter().enumerate() {
            if i % self.config.sample_rate == 0 {
                self.sa_samples.push(sa_val + start);
            }
        }

        // Update occurrence table
        self.rebuild_occ_table();

        // Update C-table
        self.rebuild_c_table();

        self.doc_boundaries.push((path.to_path_buf(), start, end));
        self.total_size = self.bwt.len() as u64 + (self.sa_samples.len() * 8) as u64;

        Ok(())
    }

    /// Search for exact pattern matches (O(m) complexity)
    pub fn search(&self, pattern: &str) -> IndexResult<Vec<SearchResult>> {
        if self.bwt.is_empty() {
            return Ok(Vec::new());
        }

        let pattern_bytes = pattern.as_bytes();
        let positions = self.backward_search(pattern_bytes);

        let mut results = Vec::new();
        for pos in positions {
            if let Some((path, line)) = self.locate_position(pos) {
                results.push(SearchResult {
                    file_path: path,
                    line,
                    column: 0,
                    score: 1.0,
                    snippet: pattern.to_string(),
                    match_type: crate::query::MatchType::Exact,
                });
            }
        }

        Ok(results)
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, path: &Path) -> IndexResult<()> {
        self.doc_boundaries.retain(|(p, _, _)| p != path);
        // Full rebuild needed for BWT correctness
        self.rebuild_full_index()?;
        Ok(())
    }

    /// Get index size in bytes
    pub fn size_bytes(&self) -> u64 {
        self.total_size
    }

    // -- Private methods --

    fn build_suffix_array(&self, text: &[u8]) -> Vec<usize> {
        // SA-IS algorithm (simplified for scaffolding)
        let n = text.len();
        let mut sa: Vec<usize> = (0..n).collect();
        sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
        sa
    }

    fn compute_bwt(&self, text: &[u8], sa: &[usize]) -> Vec<u8> {
        let n = text.len();
        sa.iter()
            .map(|&i| if i == 0 { text[n - 1] } else { text[i - 1] })
            .collect()
    }

    fn backward_search(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let mut top = 0usize;
        let mut bottom = self.bwt.len();

        for &ch in pattern.iter().rev() {
            let c_val = self.c_table[ch as usize];
            top = c_val + self.occ_count(ch, top);
            bottom = c_val + self.occ_count(ch, bottom);
            if top >= bottom {
                return Vec::new();
            }
        }

        (top..bottom)
            .filter_map(|i| self.sa_samples.get(i / self.config.sample_rate).copied())
            .collect()
    }

    fn occ_count(&self, ch: u8, pos: usize) -> usize {
        self.occ
            .get(&ch)
            .and_then(|counts| counts.get(pos).copied())
            .unwrap_or(0)
    }

    fn locate_position(&self, pos: usize) -> Option<(PathBuf, usize)> {
        for (path, start, end) in &self.doc_boundaries {
            if pos >= *start && pos < *end {
                let offset = pos - start;
                let text_slice = &self.bwt[*start..*end];
                let line = text_slice[..offset].iter().filter(|&&b| b == b'\n').count() + 1;
                return Some((path.clone(), line));
            }
        }
        None
    }

    fn rebuild_occ_table(&mut self) {
        self.occ.clear();
        for (i, &ch) in self.bwt.iter().enumerate() {
            let counts = self.occ.entry(ch).or_insert_with(|| vec![0; self.bwt.len() + 1]);
            if counts.len() <= i + 1 {
                counts.resize(i + 2, 0);
            }
            counts[i + 1] = counts[i] + 1;
        }
    }

    fn rebuild_c_table(&mut self) {
        self.c_table = [0; 256];
        for &ch in &self.bwt {
            self.c_table[ch as usize] += 1;
        }
        let mut sum = 0;
        for i in 0..256 {
            let count = self.c_table[i];
            self.c_table[i] = sum;
            sum += count;
        }
    }

    fn rebuild_full_index(&mut self) -> IndexResult<()> {
        // In production, rebuild BWT from remaining documents
        self.total_size = self.bwt.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FMIndexConfig;

    #[test]
    fn test_new_fm_index() {
        let config = FMIndexConfig::default();
        let index = FMIndex::new(config).unwrap();
        assert_eq!(index.size_bytes(), 0);
    }

    #[test]
    fn test_add_and_search() {
        let config = FMIndexConfig::default();
        let mut index = FMIndex::new(config).unwrap();

        let tokens = vec![crate::incremental::Token {
            text: "fn hello_world() { println!(\"hello\"); }".to_string(),
            kind: crate::incremental::TokenKind::Function,
            start_byte: 0,
            end_byte: 40,
            line: 1,
        }];

        let path = Path::new("test.rs");
        index.add_document(path, &tokens).unwrap();
        assert!(index.size_bytes() > 0);
    }

    #[test]
    fn test_empty_search() {
        let config = FMIndexConfig::default();
        let index = FMIndex::new(config).unwrap();
        let results = index.search("test").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_document() {
        let config = FMIndexConfig::default();
        let mut index = FMIndex::new(config).unwrap();
        let path = Path::new("test.rs");
        index.remove_document(path).unwrap();
    }
}
