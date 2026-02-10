//! Query engine for causal inference.

use crate::{CausalEvent, EventType};

/// Query filter for causal events
#[derive(Debug, Clone)]
pub struct EventQuery {
    pub event_type: Option<EventType>,
    pub source: Option<String>,
    pub min_confidence: Option<f64>,
    pub limit: Option<usize>,
}

impl EventQuery {
    pub fn new() -> Self {
        Self {
            event_type: None,
            source: None,
            min_confidence: None,
            limit: None,
        }
    }

    pub fn with_type(mut self, et: EventType) -> Self {
        self.event_type = Some(et);
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if an event matches this query
    pub fn matches(&self, event: &CausalEvent) -> bool {
        if let Some(ref et) = self.event_type {
            if event.event_type != *et {
                return false;
            }
        }
        if let Some(ref source) = self.source {
            if event.source != *source {
                return false;
            }
        }
        true
    }
}

impl Default for EventQuery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn sample_event(et: EventType, source: &str) -> CausalEvent {
        CausalEvent {
            id: Uuid::new_v4(),
            event_type: et,
            timestamp: Utc::now(),
            source: source.to_string(),
            description: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_query_matches_all() {
        let q = EventQuery::new();
        let event = sample_event(EventType::ModelInference, "engine");
        assert!(q.matches(&event));
    }

    #[test]
    fn test_query_matches_type() {
        let q = EventQuery::new().with_type(EventType::ModelInference);
        assert!(q.matches(&sample_event(EventType::ModelInference, "a")));
        assert!(!q.matches(&sample_event(EventType::UserAction, "a")));
    }

    #[test]
    fn test_query_matches_source() {
        let q = EventQuery::new().with_source("engine");
        assert!(q.matches(&sample_event(EventType::ModelInference, "engine")));
        assert!(!q.matches(&sample_event(EventType::ModelInference, "ui")));
    }
}
