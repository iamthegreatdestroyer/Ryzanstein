# zkaudit

Zero-knowledge audit trail system for the Ryzanstein LLM ecosystem.

## Features

- **Hash-Chain Audit Log** — tamper-evident sequential entries with SHA-256 linking
- **Merkle Tree Verification** — efficient proof of audit completeness
- **ZK Proofs** — prove knowledge of audit data without revealing content
- **7 Action Types** — ModelLoad, Inference, ConfigChange, Access, DataExport, SystemAlert

## Quick Start

```rust
use zkaudit::{AuditChain, AuditAction, MerkleTree, ZkProof, ProofVerifier};

let mut chain = AuditChain::new();
chain.append(AuditAction::Inference, "prompt processed", "user-1");
assert!(chain.verify_integrity());

let tree = MerkleTree::from_entries(chain.entries());
println!("Root: {}", tree.root().unwrap());

let proof = ZkProof::generate(b"secret", b"nonce");
assert!(ProofVerifier::verify(&proof));
```

## License

AGPL-3.0
