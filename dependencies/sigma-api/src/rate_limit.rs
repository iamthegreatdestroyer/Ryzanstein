//! Token-bucket rate limiter using DashMap for concurrent access.

use dashmap::DashMap;
use std::time::Instant;

pub struct RateLimiter {
    requests: DashMap<String, Vec<Instant>>,
    max_requests: usize,
    window_secs: u64,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        RateLimiter {
            requests: DashMap::new(),
            max_requests,
            window_secs,
        }
    }

    /// Returns true if the request is allowed under the rate limit.
    pub fn allow(&self, client_id: &str) -> bool {
        let now = Instant::now();
        let mut entry = self.requests.entry(client_id.to_string()).or_default();
        let cutoff = now - std::time::Duration::from_secs(self.window_secs);

        // Remove expired entries
        entry.retain(|t| *t > cutoff);

        if entry.len() < self.max_requests {
            entry.push(now);
            true
        } else {
            false
        }
    }

    pub fn remaining(&self, client_id: &str) -> usize {
        let now = Instant::now();
        let cutoff = now - std::time::Duration::from_secs(self.window_secs);
        match self.requests.get(client_id) {
            Some(entry) => {
                let active = entry.iter().filter(|t| **t > cutoff).count();
                self.max_requests.saturating_sub(active)
            }
            None => self.max_requests,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows() {
        let limiter = RateLimiter::new(3, 60);
        assert!(limiter.allow("user-1"));
        assert!(limiter.allow("user-1"));
        assert!(limiter.allow("user-1"));
        assert!(!limiter.allow("user-1")); // 4th blocked
    }

    #[test]
    fn test_rate_limiter_separate_clients() {
        let limiter = RateLimiter::new(1, 60);
        assert!(limiter.allow("a"));
        assert!(limiter.allow("b"));
        assert!(!limiter.allow("a"));
    }

    #[test]
    fn test_remaining() {
        let limiter = RateLimiter::new(5, 60);
        assert_eq!(limiter.remaining("new-user"), 5);
        limiter.allow("new-user");
        assert_eq!(limiter.remaining("new-user"), 4);
    }
}
