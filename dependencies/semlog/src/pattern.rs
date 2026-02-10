/// Behavioral pattern grouper — identifies semantic sequences in log streams.
///
/// Groups consecutive log lines into behavioral patterns:
/// - Retry sequences (repeated similar operations)
/// - Cascading failures (errors propagating across components)
/// - Normal startup (initialization sequences)
/// - Heartbeat (periodic health checks)

use std::collections::VecDeque;

/// A behavioral pattern identified in a sequence of logs.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    /// Repeated operation with same template (e.g., retry loops)
    RetrySequence { count: u32, template_id: u32 },
    /// Errors in multiple templates within a time window
    CascadingFailure { affected_templates: Vec<u32> },
    /// Periodic occurrences of the same template
    Heartbeat { template_id: u32, interval_ms: u64 },
    /// Initialization / startup sequence
    Startup { templates: Vec<u32> },
    /// Normal operation (no special pattern)
    Normal { template_id: u32, count: u32 },
}

/// A detected log pattern with metadata.
#[derive(Debug, Clone)]
pub struct LogPattern {
    pub kind: PatternKind,
    pub start_line: u64,
    pub end_line: u64,
    pub line_count: u32,
}

/// Configuration for the pattern grouper.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Min consecutive same-template lines to detect retry
    pub retry_min_count: u32,
    /// Maximum window size for cascade detection
    pub cascade_window: usize,
    /// Error template IDs (templates containing "error", "fail", etc.)
    pub error_templates: Vec<u32>,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            retry_min_count: 3,
            cascade_window: 10,
            error_templates: Vec::new(),
        }
    }
}

/// Detects behavioral patterns in a stream of template IDs.
pub struct PatternDetector {
    config: PatternConfig,
    window: VecDeque<(u32, u64)>, // (template_id, line_number)
    patterns: Vec<LogPattern>,
    current_run_id: Option<u32>,
    current_run_start: u64,
    current_run_count: u32,
    line_counter: u64,
}

impl PatternDetector {
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            window: VecDeque::new(),
            patterns: Vec::new(),
            current_run_id: None,
            current_run_start: 0,
            current_run_count: 0,
            line_counter: 0,
        }
    }

    /// Feed a template ID from the next log line.
    pub fn feed(&mut self, template_id: u32) {
        self.line_counter += 1;

        // Track runs of same template
        if self.current_run_id == Some(template_id) {
            self.current_run_count += 1;
        } else {
            self.flush_run();
            self.current_run_id = Some(template_id);
            self.current_run_start = self.line_counter;
            self.current_run_count = 1;
        }

        // Maintain sliding window for cascade detection
        self.window.push_back((template_id, self.line_counter));
        while self.window.len() > self.config.cascade_window {
            self.window.pop_front();
        }

        // Check for cascading failures
        self.check_cascade();
    }

    /// Finalize and return all detected patterns.
    pub fn finish(mut self) -> Vec<LogPattern> {
        self.flush_run();
        self.patterns
    }

    /// Get patterns detected so far (non-consuming).
    pub fn patterns(&self) -> &[LogPattern] {
        &self.patterns
    }

    fn flush_run(&mut self) {
        if let Some(tid) = self.current_run_id {
            let kind = if self.current_run_count >= self.config.retry_min_count {
                PatternKind::RetrySequence {
                    count: self.current_run_count,
                    template_id: tid,
                }
            } else {
                PatternKind::Normal {
                    template_id: tid,
                    count: self.current_run_count,
                }
            };

            self.patterns.push(LogPattern {
                kind,
                start_line: self.current_run_start,
                end_line: self.current_run_start + self.current_run_count as u64 - 1,
                line_count: self.current_run_count,
            });
        }
        self.current_run_id = None;
        self.current_run_count = 0;
    }

    fn check_cascade(&mut self) {
        if self.config.error_templates.is_empty() {
            return;
        }

        let error_count = self
            .window
            .iter()
            .filter(|(tid, _)| self.config.error_templates.contains(tid))
            .count();

        // If >50% of the window is errors from multiple templates, it's a cascade
        if error_count > self.config.cascade_window / 2 {
            let affected: Vec<u32> = self
                .window
                .iter()
                .filter(|(tid, _)| self.config.error_templates.contains(tid))
                .map(|(tid, _)| *tid)
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if affected.len() >= 2 {
                let start = self.window.front().map(|(_, l)| *l).unwrap_or(0);
                let end = self.window.back().map(|(_, l)| *l).unwrap_or(0);
                self.patterns.push(LogPattern {
                    kind: PatternKind::CascadingFailure {
                        affected_templates: affected,
                    },
                    start_line: start,
                    end_line: end,
                    line_count: self.window.len() as u32,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_retry_sequence() {
        let mut detector = PatternDetector::new(PatternConfig {
            retry_min_count: 3,
            ..Default::default()
        });

        // Feed 5 identical template IDs → retry pattern
        for _ in 0..5 {
            detector.feed(42);
        }
        detector.feed(99); // Different template ends the run

        let patterns = detector.finish();
        assert!(patterns.iter().any(|p| matches!(
            &p.kind,
            PatternKind::RetrySequence { count: 5, template_id: 42 }
        )));
    }

    #[test]
    fn test_no_retry_short_run() {
        let mut detector = PatternDetector::new(PatternConfig {
            retry_min_count: 3,
            ..Default::default()
        });
        detector.feed(42);
        detector.feed(42);
        detector.feed(99);

        let patterns = detector.finish();
        assert!(patterns.iter().all(|p| !matches!(p.kind, PatternKind::RetrySequence { .. })));
    }

    #[test]
    fn test_detect_normal_pattern() {
        let mut detector = PatternDetector::new(PatternConfig::default());
        detector.feed(1);
        detector.feed(2);

        let patterns = detector.finish();
        assert!(patterns.iter().any(|p| matches!(p.kind, PatternKind::Normal { .. })));
    }

    #[test]
    fn test_pattern_line_tracking() {
        let mut detector = PatternDetector::new(PatternConfig {
            retry_min_count: 3,
            ..Default::default()
        });
        for _ in 0..4 {
            detector.feed(10);
        }
        let patterns = detector.finish();
        let retry = patterns.iter().find(|p| matches!(p.kind, PatternKind::RetrySequence { .. }));
        assert!(retry.is_some());
        assert_eq!(retry.unwrap().start_line, 1);
        assert_eq!(retry.unwrap().line_count, 4);
    }
}
