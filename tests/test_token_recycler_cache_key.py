"""Cache-key fix for the Token Recycler: store() and lookup() must agree on
"backend", not just "model".

Before this fix: _TokenRecyclerCache keyed lookups on "model" alone. Once a
second backend (e.g. a future llama.cpp/BitNet backend, see task #228 Thread
C) can serve the same model name as "ollama" does today, a request routed to
one backend could be served a cache hit that was actually stored by a
DIFFERENT backend for the same model name -- a real correctness bug, not
just a theoretical one, since RSU cache entries are otherwise
backend-agnostic once stored.

These tests use bare _TokenRecyclerCache instances (via __new__, mirroring
tests/test_api_server.py's existing pattern) with fake compressor/bank/
retriever doubles -- no live Qdrant needed -- to prove store() writes a
"backend" field and lookup() filters on that same field, using the module's
real BACKEND value.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sigmalang"))

import src.api.server as server


def _make_cache():
    return server._TokenRecyclerCache.__new__(server._TokenRecyclerCache)


def test_store_writes_backend_into_metadata():
    cache = _make_cache()

    captured = {}

    class _CapturingCompressor:
        async def compress(self, prompt, answer, model, metadata=None):
            captured["metadata"] = metadata

            class _FakeRSU:
                id = "fake-id"
                embedding = [0.0]
                prompt_ = prompt

            return _FakeRSU()

    class _NoopBank:
        async def store(self, rsu):
            return rsu.id

    cache.compressor = _CapturingCompressor()
    cache.bank = _NoopBank()

    asyncio.run(cache.store("some prompt", "phi4-mini", "some answer"))

    assert "metadata" in captured, "store() must call compressor.compress() with metadata"
    assert captured["metadata"].get("backend") == server.BACKEND, (
        "store() must stamp the live BACKEND value into RSU metadata so it "
        "reaches VectorBank's top-level Qdrant payload (same mechanism as "
        "_created_ts)"
    )


def test_lookup_filters_on_backend_not_just_model():
    cache = _make_cache()

    captured = {}

    class _CapturingRetriever:
        async def retrieve(self, query_vec, score_threshold=None, filter_dict=None):
            captured["filter_dict"] = filter_dict
            return None  # miss is fine; we only care about the filter shape

    class _StubCompressor:
        async def embed_query(self, text):
            return [0.0]

    cache.compressor = _StubCompressor()
    cache.retriever = _CapturingRetriever()
    cache.misses = 0
    cache.hits = 0

    asyncio.run(cache.lookup("some prompt", "phi4-mini"))

    filter_dict = captured.get("filter_dict")
    assert filter_dict is not None, "lookup() must pass a filter_dict to retriever.retrieve()"
    must_clauses = filter_dict.get("must", [])
    keys_filtered = {clause["key"]: clause["match"]["value"] for clause in must_clauses}

    assert keys_filtered.get("model") == "phi4-mini"
    assert keys_filtered.get("backend") == server.BACKEND, (
        "lookup() must filter on 'backend' in addition to 'model', or a hit "
        "stored by a different backend serving the same model name could be "
        "served as a false cache hit"
    )


def test_store_and_lookup_agree_on_the_same_backend_value():
    """The two filters/writes above must reference the identical BACKEND
    value -- not just any string -- so a real round trip (store under
    BACKEND=X, then lookup while BACKEND=X) actually matches."""
    store_cache = _make_cache()
    lookup_cache = _make_cache()

    captured = {}

    class _CapturingCompressor:
        async def compress(self, prompt, answer, model, metadata=None):
            captured["store_backend"] = metadata.get("backend")

            class _FakeRSU:
                id = "fake-id"
                embedding = [0.0]

            return _FakeRSU()

    class _NoopBank:
        async def store(self, rsu):
            return rsu.id

    class _CapturingRetriever:
        async def retrieve(self, query_vec, score_threshold=None, filter_dict=None):
            for clause in filter_dict.get("must", []):
                if clause["key"] == "backend":
                    captured["lookup_backend"] = clause["match"]["value"]
            return None

    class _StubCompressor:
        async def embed_query(self, text):
            return [0.0]

    store_cache.compressor = _CapturingCompressor()
    store_cache.bank = _NoopBank()
    asyncio.run(store_cache.store("p", "m", "a"))

    lookup_cache.compressor = _StubCompressor()
    lookup_cache.retriever = _CapturingRetriever()
    lookup_cache.misses = 0
    lookup_cache.hits = 0
    asyncio.run(lookup_cache.lookup("p", "m"))

    assert captured["store_backend"] == captured["lookup_backend"] == server.BACKEND


if __name__ == "__main__":
    test_store_writes_backend_into_metadata()
    print("PASS: store() writes backend into metadata")
    test_lookup_filters_on_backend_not_just_model()
    print("PASS: lookup() filters on backend, not just model")
    test_store_and_lookup_agree_on_the_same_backend_value()
    print("PASS: store()/lookup() agree on the same backend value")
    print("\nALL TESTS PASSED")
