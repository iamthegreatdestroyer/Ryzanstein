//! # causedb
//!
//! Causal inference database for the Ryzanstein LLM ecosystem.
//! Tracks cause-and-effect relationships between system events,
//! model outputs, pipeline stages, and user actions.

pub mod config;
pub mod error;
pub mod graph;
pub mod query;
pub mod ryzanstein_integration;

use std::collections::HashMap;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use config::CauseDbConfig;
use error::CauseDbError;
use uuid::Uuid;

/// A causal event in the system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalEvent {
    pub id: Uuid,
    pub event_type: EventType,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub description: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Event categories
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EventType {
    ModelInference,
    TokenGeneration,
    CacheOperation,
    PipelineStage,
    UserAction,
    SystemEvent,
    ErrorEvent,
    Custom(String),
}

/// A causal link between two events
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalLink {
    pub id: Uuid,
    pub cause_id: Uuid,
    pub effect_id: Uuid,
    pub link_type: LinkType,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Types of causal relationships
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LinkType {
    Direct,
    Indirect,
    Correlation,
    Temporal,
    Counterfactual,
}

/// Query result for causal analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct CausalChain {
    pub root: Uuid,
    pub chain: Vec<CausalEvent>,
    pub links: Vec<CausalLink>,
    pub total_confidence: f64,
}

/// Database statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct DbStats {
    pub total_events: usize,
    pub total_links: usize,
    pub event_types: HashMap<String, usize>,
}

/// Main causal inference database
pub struct CauseDb {
    config: CauseDbConfig,
    events: Mutex<HashMap<Uuid, CausalEvent>>,
    links: Mutex<Vec<CausalLink>>,
}

impl CauseDb {
    /// Create a new CauseDb instance
    pub fn new(config: CauseDbConfig) -> Self {
        Self {
            config,
            events: Mutex::new(HashMap::new()),
            links: Mutex::new(Vec::new()),
        }
    }

    /// Record a new causal event
    pub fn record_event(&self, event_type: EventType, source: &str, description: &str) -> Result<Uuid, CauseDbError> {
        let event = CausalEvent {
            id: Uuid::new_v4(),
            event_type,
            timestamp: Utc::now(),
            source: source.to_string(),
            description: description.to_string(),
            metadata: HashMap::new(),
        };
        let id = event.id;
        self.events.lock().unwrap().insert(id, event);
        Ok(id)
    }

    /// Record a causal event with metadata
    pub fn record_event_with_metadata(
        &self,
        event_type: EventType,
        source: &str,
        description: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Uuid, CauseDbError> {
        let event = CausalEvent {
            id: Uuid::new_v4(),
            event_type,
            timestamp: Utc::now(),
            source: source.to_string(),
            description: description.to_string(),
            metadata,
        };
        let id = event.id;
        self.events.lock().unwrap().insert(id, event);
        Ok(id)
    }

    /// Link two events with a causal relationship
    pub fn link_events(
        &self,
        cause_id: Uuid,
        effect_id: Uuid,
        link_type: LinkType,
        confidence: f64,
    ) -> Result<Uuid, CauseDbError> {
        if confidence < 0.0 || confidence > 1.0 {
            return Err(CauseDbError::InvalidConfidence(confidence));
        }
        let events = self.events.lock().unwrap();
        if !events.contains_key(&cause_id) {
            return Err(CauseDbError::EventNotFound(cause_id));
        }
        if !events.contains_key(&effect_id) {
            return Err(CauseDbError::EventNotFound(effect_id));
        }
        drop(events);

        let link = CausalLink {
            id: Uuid::new_v4(),
            cause_id,
            effect_id,
            link_type,
            confidence,
            metadata: HashMap::new(),
        };
        let id = link.id;
        self.links.lock().unwrap().push(link);
        Ok(id)
    }

    /// Get an event by ID
    pub fn get_event(&self, id: Uuid) -> Option<CausalEvent> {
        self.events.lock().unwrap().get(&id).cloned()
    }

    /// Find direct effects of an event
    pub fn find_effects(&self, cause_id: Uuid) -> Vec<CausalEvent> {
        let links = self.links.lock().unwrap();
        let events = self.events.lock().unwrap();
        links.iter()
            .filter(|l| l.cause_id == cause_id)
            .filter_map(|l| events.get(&l.effect_id).cloned())
            .collect()
    }

    /// Find direct causes of an event
    pub fn find_causes(&self, effect_id: Uuid) -> Vec<CausalEvent> {
        let links = self.links.lock().unwrap();
        let events = self.events.lock().unwrap();
        links.iter()
            .filter(|l| l.effect_id == effect_id)
            .filter_map(|l| events.get(&l.cause_id).cloned())
            .collect()
    }

    /// Trace full causal chain from root event
    pub fn trace_chain(&self, root_id: Uuid) -> Result<CausalChain, CauseDbError> {
        let events = self.events.lock().unwrap();
        let links = self.links.lock().unwrap();

        let root = events.get(&root_id)
            .ok_or(CauseDbError::EventNotFound(root_id))?
            .clone();

        let mut chain = vec![root];
        let mut chain_links = Vec::new();
        let mut current_ids = vec![root_id];
        let mut visited = std::collections::HashSet::new();
        visited.insert(root_id);

        loop {
            let mut next_ids = Vec::new();
            for cid in &current_ids {
                for link in links.iter() {
                    if link.cause_id == *cid && !visited.contains(&link.effect_id) {
                        visited.insert(link.effect_id);
                        if let Some(event) = events.get(&link.effect_id) {
                            chain.push(event.clone());
                            chain_links.push(link.clone());
                            next_ids.push(link.effect_id);
                        }
                    }
                }
            }
            if next_ids.is_empty() {
                break;
            }
            current_ids = next_ids;
        }

        let total_confidence = if chain_links.is_empty() {
            1.0
        } else {
            chain_links.iter().map(|l| l.confidence).product()
        };

        Ok(CausalChain {
            root: root_id,
            chain,
            links: chain_links,
            total_confidence,
        })
    }

    /// Get database statistics
    pub fn stats(&self) -> DbStats {
        let events = self.events.lock().unwrap();
        let links = self.links.lock().unwrap();
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for event in events.values() {
            let key = format!("{:?}", event.event_type);
            *type_counts.entry(key).or_insert(0) += 1;
        }
        DbStats {
            total_events: events.len(),
            total_links: links.len(),
            event_types: type_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> CauseDb {
        CauseDb::new(CauseDbConfig::default())
    }

    #[test]
    fn test_record_event() {
        let db = test_db();
        let id = db.record_event(EventType::ModelInference, "engine", "inference started").unwrap();
        let event = db.get_event(id).unwrap();
        assert_eq!(event.source, "engine");
    }

    #[test]
    fn test_link_events() {
        let db = test_db();
        let a = db.record_event(EventType::UserAction, "ui", "user prompt").unwrap();
        let b = db.record_event(EventType::ModelInference, "engine", "inference").unwrap();
        let link_id = db.link_events(a, b, LinkType::Direct, 0.95).unwrap();
        assert_ne!(link_id, Uuid::nil());
    }

    #[test]
    fn test_invalid_confidence() {
        let db = test_db();
        let a = db.record_event(EventType::SystemEvent, "sys", "a").unwrap();
        let b = db.record_event(EventType::SystemEvent, "sys", "b").unwrap();
        assert!(db.link_events(a, b, LinkType::Direct, 1.5).is_err());
    }

    #[test]
    fn test_find_effects() {
        let db = test_db();
        let root = db.record_event(EventType::UserAction, "ui", "prompt").unwrap();
        let e1 = db.record_event(EventType::ModelInference, "engine", "infer").unwrap();
        let e2 = db.record_event(EventType::TokenGeneration, "engine", "tokens").unwrap();
        db.link_events(root, e1, LinkType::Direct, 0.9).unwrap();
        db.link_events(root, e2, LinkType::Direct, 0.8).unwrap();

        let effects = db.find_effects(root);
        assert_eq!(effects.len(), 2);
    }

    #[test]
    fn test_find_causes() {
        let db = test_db();
        let a = db.record_event(EventType::CacheOperation, "cache", "miss").unwrap();
        let b = db.record_event(EventType::ModelInference, "engine", "slow inference").unwrap();
        db.link_events(a, b, LinkType::Direct, 0.85).unwrap();

        let causes = db.find_causes(b);
        assert_eq!(causes.len(), 1);
        assert_eq!(causes[0].id, a);
    }

    #[test]
    fn test_trace_chain() {
        let db = test_db();
        let a = db.record_event(EventType::UserAction, "ui", "prompt").unwrap();
        let b = db.record_event(EventType::ModelInference, "engine", "infer").unwrap();
        let c = db.record_event(EventType::TokenGeneration, "engine", "tokens").unwrap();
        db.link_events(a, b, LinkType::Direct, 0.9).unwrap();
        db.link_events(b, c, LinkType::Direct, 0.8).unwrap();

        let chain = db.trace_chain(a).unwrap();
        assert_eq!(chain.chain.len(), 3);
        assert!((chain.total_confidence - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let db = test_db();
        db.record_event(EventType::ModelInference, "a", "a1").unwrap();
        db.record_event(EventType::ModelInference, "a", "a2").unwrap();
        db.record_event(EventType::UserAction, "b", "b1").unwrap();

        let stats = db.stats();
        assert_eq!(stats.total_events, 3);
    }

    #[test]
    fn test_event_not_found() {
        let db = test_db();
        let fake = Uuid::new_v4();
        let real = db.record_event(EventType::SystemEvent, "x", "x").unwrap();
        assert!(db.link_events(fake, real, LinkType::Direct, 0.5).is_err());
    }
}
