"""
Innovation #8: Glyph Fingerprinting & Cryptographic Proofs — Unit Tests
=======================================================================

Validates:
  - GlyphFingerprint: create, verify, wire format round-trip
  - Bloom filter: correct set membership, false-positive behavior
  - GlyphFingerprintChain: append, merkle_root, prove
  - GlyphProof: verify_inclusion
  - GlyphFingerprintVerifier: full round-trip audit
  - Chain integrity: changing any step invalidates successors
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

from src.fingerprint.glyph_fingerprint import (
    GlyphFingerprint,
    GlyphFingerprintChain,
    GlyphFingerprintVerifier,
    GlyphProof,
)


# ============================================================================
# GlyphFingerprint
# ============================================================================

class TestGlyphFingerprint:

    def _fp(self, step=0, label="ssm_forward", inp=None, out=None, prev=None):
        return GlyphFingerprint.create(
            step              = step,
            transform_label   = label,
            input_primitives  = inp or [0, 1, 2],
            output_primitives = out or [3, 4, 5],
            prev_commitment   = prev,
        )

    def test_commitment_is_32_bytes(self):
        fp = self._fp()
        assert len(fp.commitment) == 32

    def test_deterministic(self):
        fp1 = self._fp()
        fp2 = self._fp()
        assert fp1.commitment == fp2.commitment

    def test_different_inputs_different_commitment(self):
        fp1 = self._fp(inp=[0, 1, 2])
        fp2 = self._fp(inp=[3, 4, 5])
        assert fp1.commitment != fp2.commitment

    def test_different_outputs_different_commitment(self):
        fp1 = self._fp(out=[0, 1])
        fp2 = self._fp(out=[0, 2])
        assert fp1.commitment != fp2.commitment

    def test_different_step_different_commitment(self):
        fp1 = self._fp(step=0)
        fp2 = self._fp(step=1)
        assert fp1.commitment != fp2.commitment

    def test_different_label_different_commitment(self):
        fp1 = self._fp(label="kv_cache")
        fp2 = self._fp(label="ssm_forward")
        assert fp1.commitment != fp2.commitment

    def test_verify_correct_inputs(self):
        fp = self._fp(inp=[0, 1, 2], out=[3, 4, 5])
        assert fp.verify([0, 1, 2], [3, 4, 5])

    def test_verify_wrong_input_fails(self):
        fp = self._fp(inp=[0, 1, 2], out=[3, 4, 5])
        # Different input → commitment won't match
        assert not fp.verify([99], [3, 4, 5])

    def test_verify_wrong_output_fails(self):
        fp = self._fp(inp=[0, 1, 2], out=[3, 4, 5])
        assert not fp.verify([0, 1, 2], [99])

    def test_wire_format_round_trip(self):
        fp = self._fp()
        data = fp.to_bytes()
        assert len(data) >= 112
        fp2 = GlyphFingerprint.from_bytes(data)
        assert fp2.commitment == fp.commitment
        assert fp2.step == fp.step
        assert fp2.transform_label == fp.transform_label

    def test_hex_is_64_chars(self):
        fp = self._fp()
        assert len(fp.hex()) == 64

    def test_prev_commitment_chaining(self):
        fp1 = self._fp(step=0, prev=None)
        fp2 = self._fp(step=1, prev=fp1.commitment)
        assert fp2.prev_commitment == fp1.commitment
        assert fp2.commitment != fp1.commitment

    def test_bloom_different_from_empty_set(self):
        """Non-empty input set should produce non-zero bloom."""
        bloom = GlyphFingerprint._bloom([0, 1, 2])
        assert any(b != 0 for b in bloom)

    def test_bloom_empty_set_all_zeros(self):
        bloom = GlyphFingerprint._bloom([])
        assert bloom == bytes(32)


# ============================================================================
# GlyphFingerprintChain
# ============================================================================

class TestGlyphFingerprintChain:

    def test_empty_chain_length(self):
        chain = GlyphFingerprintChain()
        assert len(chain) == 0

    def test_append_increments_length(self):
        chain = GlyphFingerprintChain()
        chain.append("op_a", [0, 1], [2, 3])
        assert len(chain) == 1

    def test_append_returns_fingerprint(self):
        chain = GlyphFingerprintChain()
        fp = chain.append("op", [0], [1])
        assert isinstance(fp, GlyphFingerprint)

    def test_steps_sequential(self):
        chain = GlyphFingerprintChain()
        for i in range(5):
            fp = chain.append(f"op_{i}", [i], [i+1])
            assert fp.step == i

    def test_prev_commitment_chained(self):
        chain = GlyphFingerprintChain()
        fp0 = chain.append("step0", [0], [1])
        fp1 = chain.append("step1", [1], [2])
        assert fp1.prev_commitment == fp0.commitment

    def test_merkle_root_empty_chain(self):
        chain = GlyphFingerprintChain()
        assert chain.merkle_root() == bytes(32)

    def test_merkle_root_single(self):
        chain = GlyphFingerprintChain()
        fp = chain.append("op", [0], [1])
        assert chain.merkle_root() == fp.commitment

    def test_merkle_root_deterministic(self):
        def build():
            c = GlyphFingerprintChain()
            c.append("a", [0], [1])
            c.append("b", [1], [2])
            return c.merkle_root()
        assert build() == build()

    def test_merkle_root_changes_if_chain_changes(self):
        c1 = GlyphFingerprintChain()
        c1.append("a", [0], [1])
        c2 = GlyphFingerprintChain()
        c2.append("a", [0], [99])   # different output
        assert c1.merkle_root() != c2.merkle_root()

    def test_prove_returns_proof(self):
        chain = GlyphFingerprintChain()
        chain.append("op_a", [0, 1], [2, 3])
        chain.append("op_b", [2, 3], [4, 5])
        proof = chain.prove(step=0)
        assert isinstance(proof, GlyphProof)

    def test_prove_out_of_range_raises(self):
        chain = GlyphFingerprintChain()
        chain.append("op", [0], [1])
        with pytest.raises(IndexError):
            chain.prove(step=5)

    def test_stats(self):
        chain = GlyphFingerprintChain(session_id="sess-001")
        chain.append("op", [0], [1])
        s = chain.stats()
        assert s["chain_length"] == 1
        assert s["session_id"] == "sess-001"
        assert s["merkle_root"] is not None


# ============================================================================
# GlyphProof verification
# ============================================================================

class TestGlyphProof:

    def _build_chain_and_proof(self, n=4, prove_step=0):
        chain = GlyphFingerprintChain()
        inputs  = [[i, i+1] for i in range(n)]
        outputs = [[i+2, i+3] for i in range(n)]
        for i in range(n):
            chain.append(f"step_{i}", inputs[i], outputs[i])
        proof = chain.prove(step=prove_step)
        return chain, proof, inputs[prove_step], outputs[prove_step]

    def test_verify_inclusion_correct(self):
        _, proof, inp, out = self._build_chain_and_proof()
        assert proof.verify_inclusion(inp, out)

    def test_verify_inclusion_wrong_output_fails(self):
        _, proof, inp, out = self._build_chain_and_proof()
        assert not proof.verify_inclusion(inp, [99, 100])

    def test_verify_inclusion_wrong_input_fails(self):
        _, proof, inp, out = self._build_chain_and_proof()
        assert not proof.verify_inclusion([99, 100], out)

    def test_proof_root_matches_chain(self):
        chain, proof, _, _ = self._build_chain_and_proof()
        assert proof.root == chain.merkle_root()

    def test_as_json(self):
        _, proof, _, _ = self._build_chain_and_proof()
        d = proof.as_json()
        assert "step" in d and "commitment" in d and "root" in d


# ============================================================================
# GlyphFingerprintVerifier
# ============================================================================

class TestGlyphFingerprintVerifier:

    def _setup(self, n=3):
        chain = GlyphFingerprintChain(session_id="audit-test")
        inps, outs = [], []
        for i in range(n):
            inp = [i, i+1]
            out = [i+2, i+3]
            chain.append(f"transform_{i}", inp, out)
            inps.append(inp)
            outs.append(out)
        return chain, inps, outs

    def test_verify_proof_ok(self):
        chain, inps, outs = self._setup()
        verifier = GlyphFingerprintVerifier()
        proof = chain.prove(step=0)
        ok, reason = verifier.verify_proof(proof, inps[0], outs[0])
        assert ok, f"Expected ok, got: {reason}"
        assert reason == "ok"

    def test_verify_proof_wrong_input_fails(self):
        chain, inps, outs = self._setup()
        verifier = GlyphFingerprintVerifier()
        proof = chain.prove(step=0)
        ok, reason = verifier.verify_proof(proof, [99], outs[0])
        assert not ok
        assert "mismatch" in reason

    def test_verify_proof_with_expected_root(self):
        chain, inps, outs = self._setup()
        verifier = GlyphFingerprintVerifier()
        proof = chain.prove(step=1)
        ok, reason = verifier.verify_proof(
            proof, inps[1], outs[1], expected_root=chain.merkle_root()
        )
        assert ok

    def test_verify_proof_wrong_expected_root_fails(self):
        chain, inps, outs = self._setup()
        verifier = GlyphFingerprintVerifier()
        proof = chain.prove(step=0)
        wrong_root = bytes(32)
        ok, reason = verifier.verify_proof(proof, inps[0], outs[0], expected_root=wrong_root)
        assert not ok
        assert "root" in reason

    def test_verify_chain_integrity_valid(self):
        chain, _, _ = self._setup(n=5)
        verifier = GlyphFingerprintVerifier()
        ok, reason = verifier.verify_chain_integrity(chain)
        assert ok, reason

    def test_verify_chain_integrity_tampered(self):
        chain, _, _ = self._setup(n=3)
        verifier = GlyphFingerprintVerifier()
        # Tamper: break the chain linkage by manually overriding a fingerprint
        original = chain._chain[1]
        broken = GlyphFingerprint(
            commitment      = original.commitment,
            step            = original.step,
            transform_label = original.transform_label,
            input_bloom     = original.input_bloom,
            output_bloom    = original.output_bloom,
            prev_commitment = bytes(32),   # wrong prev → chain broken
            timestamp       = original.timestamp,
        )
        chain._chain[1] = broken
        ok, reason = verifier.verify_chain_integrity(chain)
        assert not ok
        assert "broken" in reason
