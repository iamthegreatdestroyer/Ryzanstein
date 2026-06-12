"""
Zero-knowledge audit trail for Ryzanstein LLM.

Python implementation of the Rust zkaudit crate protocol.
Hash chain + Merkle tree + Schnorr sigma protocol ZK proofs.

Chain format is byte-compatible with the Rust zkaudit crate so audit
logs can be verified by either runtime.
"""

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# RFC 3526 Group 5 safe prime (1536-bit) — matches Rust zkaudit crate
# ---------------------------------------------------------------------------
_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF",
    16,
)
_G = 2


# ---------------------------------------------------------------------------
# Audit chain
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    id: str
    timestamp: int
    action: str
    description: str
    actor: str
    hash: str
    prev_hash: str


class AuditChain:
    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []

    def append(self, action: str, description: str, actor: str) -> AuditEntry:
        prev_hash = self._entries[-1].hash if self._entries else "0" * 64
        entry_id = str(uuid.uuid4())
        ts = int(time.time())

        raw = f"{entry_id}|{ts}|{action}|{description}|{actor}|{prev_hash}"
        entry_hash = hashlib.sha256(raw.encode()).hexdigest()

        entry = AuditEntry(
            id=entry_id,
            timestamp=ts,
            action=action,
            description=description,
            actor=actor,
            hash=entry_hash,
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    def verify_integrity(self) -> bool:
        for i, entry in enumerate(self._entries):
            expected_prev = "0" * 64 if i == 0 else self._entries[i - 1].hash
            if entry.prev_hash != expected_prev:
                return False
        return True

    @property
    def entries(self) -> List[AuditEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------

def _sha256_pair(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(a + b).digest()


class MerkleTree:
    def __init__(self, leaves: List[bytes]) -> None:
        self._leaves = leaves
        self._root: Optional[bytes] = self._build(list(leaves))

    @classmethod
    def from_entries(cls, entries: List[AuditEntry]) -> "MerkleTree":
        leaves = [hashlib.sha256(e.hash.encode()).digest() for e in entries]
        return cls(leaves)

    def _build(self, layer: List[bytes]) -> Optional[bytes]:
        if not layer:
            return None
        while len(layer) > 1:
            if len(layer) % 2:
                layer.append(layer[-1])  # duplicate last
            layer = [_sha256_pair(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
        return layer[0]

    def root(self) -> Optional[str]:
        return self._root.hex() if self._root else None


# ---------------------------------------------------------------------------
# Schnorr sigma protocol ZK proof
# ---------------------------------------------------------------------------

@dataclass
class ZkProof:
    commitment: str  # h = g^x mod p (hex)
    challenge: str   # c = H(t || message) (hex)
    response: str    # s = r + c*x (hex)
    t_point: str     # t = g^r mod p (hex)

    def to_dict(self) -> dict:
        return {
            "commitment": self.commitment,
            "challenge": self.challenge,
            "response": self.response,
            "t_point": self.t_point,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZkProof":
        return cls(**d)

    @staticmethod
    def generate(secret: bytes, message: bytes) -> "ZkProof":
        # Derive secret scalar x from secret input
        x = int.from_bytes(hashlib.sha256(secret).digest(), "big") % _P
        # Public key h = g^x mod p
        h = pow(_G, x, _P)
        # Random nonce r
        r_bytes = secrets.token_bytes(64)
        r = int.from_bytes(r_bytes, "big") % _P
        # Commitment t = g^r mod p
        t = pow(_G, r, _P)
        # Fiat-Shamir challenge c = H(t || message)
        h_input = t.to_bytes((t.bit_length() + 7) // 8, "big") + message
        c = int.from_bytes(hashlib.sha256(h_input).digest(), "big")
        # Response s = r + c * x
        s = r + c * x

        def to_hex(n: int) -> str:
            nb = (n.bit_length() + 7) // 8 or 1
            return n.to_bytes(nb, "big").hex()

        return ZkProof(
            commitment=to_hex(h),
            challenge=to_hex(c),
            response=to_hex(s),
            t_point=to_hex(t),
        )

    @staticmethod
    def verify(proof: "ZkProof", message: bytes) -> bool:
        try:
            h = int.from_bytes(bytes.fromhex(proof.commitment), "big")
            t = int.from_bytes(bytes.fromhex(proof.t_point), "big")
            s = int.from_bytes(bytes.fromhex(proof.response), "big")
        except (ValueError, AttributeError):
            return False
        # Recompute challenge
        h_input = t.to_bytes((t.bit_length() + 7) // 8, "big") + message
        c = int.from_bytes(hashlib.sha256(h_input).digest(), "big")
        # Verify g^s mod p == (t * h^c) mod p
        lhs = pow(_G, s, _P)
        rhs = (t * pow(h, c, _P)) % _P
        return lhs == rhs


# ---------------------------------------------------------------------------
# Session-scoped audit manager (in-memory, per-request chain)
# ---------------------------------------------------------------------------

class AuditManager:
    """Per-request audit trail builder. Create one per API request."""

    def __init__(self, request_id: str, api_key_prefix: str) -> None:
        self.request_id = request_id
        self.api_key_prefix = api_key_prefix
        self._chain = AuditChain()

    def record_inference(self, model: str, prompt_tokens: int, output_tokens: int) -> None:
        self._chain.append(
            action="Inference",
            description=f"model={model} prompt_tokens={prompt_tokens} output_tokens={output_tokens}",
            actor=self.api_key_prefix,
        )

    def seal(self) -> dict:
        """
        Seal the audit trail and return a ZK proof.

        Returns a dict with Merkle root, entry count, ZK proof,
        and chain integrity flag.
        """
        tree = MerkleTree.from_entries(self._chain.entries)
        merkle_root = tree.root() or "empty"
        integrity_ok = self._chain.verify_integrity()

        proof = ZkProof.generate(
            secret=merkle_root.encode(),
            message=f"request:{self.request_id}".encode(),
        )

        return {
            "request_id": self.request_id,
            "merkle_root": merkle_root,
            "entry_count": len(self._chain),
            "chain_integrity": integrity_ok,
            "zk_proof": proof.to_dict(),
            "zk_verified": ZkProof.verify(proof, f"request:{self.request_id}".encode()),
        }
