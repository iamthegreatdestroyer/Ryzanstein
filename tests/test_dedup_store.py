"""Tests for dedup-on-store (P1) — extracts the REAL _TokenRecyclerCache.store
method from src/api/server.py via AST and binds it to fake collaborators (no
real Qdrant/embeddings). Covers: skip on same-answer near-exact twin, store on
no-twin, store on different-answer twin, fail-open when the dedup check raises.

    pytest tests/test_dedup_store.py   |   python tests/test_dedup_store.py
"""
import ast
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "src", "api", "server.py"),
    os.path.join(HERE, "src", "api", "server.py"),
    os.path.join(HERE, "live_server.py"),
    os.path.join(HERE, "server.py"),
]
SRC_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert SRC_PATH, f"could not locate server source in {CANDIDATES}"
with open(SRC_PATH, encoding="utf-8") as fh:
    SOURCE = fh.read()
TREE = ast.parse(SOURCE)


def _find_class_method(class_name, method_name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {SRC_PATH}")


_store_node = _find_class_method("_TokenRecyclerCache", "store")
_STORE_SRC = ast.get_source_segment(SOURCE, _store_node)
assert _STORE_SRC, "could not extract store() source"
# The extracted method is indented (class body) -- dedent so it compiles standalone.
_lines = _STORE_SRC.splitlines()
_indent = len(_lines[0]) - len(_lines[0].lstrip())
_STORE_SRC = "\n".join(l[_indent:] if l[:_indent].strip() == "" else l for l in _lines)


class _FakeLogger:
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass


class _FakeHttpxClient:
    """Stands in for httpx.AsyncClient so the sigma-index dual-write branch
    (irrelevant to this test, and gated off by default anyway) never makes a
    real network call if somehow reached."""
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, *a, **k):
        raise RuntimeError("dual-write should be unreachable in this test")


NS = {
    "_gw_time": __import__("time"),
    "_gw_bucket": lambda v, size: (f"{v:g}" if size <= 0 else f"{round(round(v/size)*size, 6):g}"),
    "logger": _FakeLogger(),
    "_GW_METRICS": {"store_failures_total": 0, "stores_total": 0, "dedup_skipped_total": 0},
    "BACKEND": "ollama",
    "_SIGMA_INDEX_DUALWRITE": False,   # off by default post step-4-slim; irrelevant here
    "_SIGMA_INDEX_URL": "http://unused",
    "_RECYCLER_TEMP_BUCKET": 0.1,
    "_RECYCLER_TOPP_BUCKET": 0.05,
    "_RECYCLER_EXACT_TWIN": 0.999,
    "_gw_httpx": type("M", (), {"AsyncClient": _FakeHttpxClient}),
    "Optional": Optional,
    "__builtins__": __builtins__,
}
exec(compile("async def store" + _STORE_SRC[len("async def store"):], SRC_PATH, "exec"), NS)
store = NS["store"]


@dataclass
class FakeRSU:
    id: str = "rsu-new-id"
    embedding: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    prompt: str = "q"
    answer: str = "A"
    model: str = "m"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeCandidate:
    rsu_id: str
    score: float
    answer: str
    prompt: str = "q"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FakeCompressor:
    def __init__(self, rsu=None, raises=False):
        self._rsu = rsu or FakeRSU()
        self._raises = raises

    async def compress(self, prompt, answer, model, metadata=None):
        if self._raises:
            raise RuntimeError("compress failed")
        r = self._rsu
        r.prompt, r.answer, r.model = prompt, answer, model
        return r


class FakeRetriever:
    def __init__(self, twins=None, raises=False):
        self._twins = twins or []
        self._raises = raises
        self.calls = []

    async def retrieve_candidates(self, query_embedding, score_threshold=None, filter_dict=None, limit=None):
        self.calls.append({"score_threshold": score_threshold, "filter_dict": filter_dict, "limit": limit})
        if self._raises:
            raise RuntimeError("dedup lookup failed")
        return self._twins


class FakeBank:
    def __init__(self):
        self.stored: List[Any] = []

    async def store(self, rsu):
        self.stored.append(rsu)
        return rsu.id


class FakeCache:
    """Minimal stand-in for _TokenRecyclerCache carrying only what store() touches."""
    def __init__(self, compressor, retriever, bank):
        self.compressor = compressor
        self.retriever = retriever
        self.bank = bank


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _reset_metrics():
    NS["_GW_METRICS"]["store_failures_total"] = 0
    NS["_GW_METRICS"]["stores_total"] = 0
    NS["_GW_METRICS"]["dedup_skipped_total"] = 0


def test_no_twin_stores_normally():
    _reset_metrics()
    bank, retriever = FakeBank(), FakeRetriever(twins=[])
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A"))
    assert len(bank.stored) == 1
    assert NS["_GW_METRICS"]["dedup_skipped_total"] == 0
    assert NS["_GW_METRICS"]["stores_total"] == 1


def test_same_answer_twin_skips_store():
    _reset_metrics()
    twin = FakeCandidate(rsu_id="old", score=0.9995, answer="A")
    bank, retriever = FakeBank(), FakeRetriever(twins=[twin])
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A"))
    assert len(bank.stored) == 0                       # skipped
    assert NS["_GW_METRICS"]["dedup_skipped_total"] == 1
    assert NS["_GW_METRICS"]["stores_total"] == 0       # not counted as a real store


def test_different_answer_twin_still_stores():
    # near-exact score but a DIFFERENT answer -> not a duplicate, store normally
    _reset_metrics()
    twin = FakeCandidate(rsu_id="old", score=0.9995, answer="DIFFERENT")
    bank, retriever = FakeBank(), FakeRetriever(twins=[twin])
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A"))
    assert len(bank.stored) == 1
    assert NS["_GW_METRICS"]["dedup_skipped_total"] == 0


def test_dedup_check_error_falls_through_to_store():
    # fail-open: if the dedup lookup itself raises, the real store must still happen
    _reset_metrics()
    bank, retriever = FakeBank(), FakeRetriever(raises=True)
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A"))
    assert len(bank.stored) == 1
    assert NS["_GW_METRICS"]["dedup_skipped_total"] == 0
    assert NS["_GW_METRICS"]["stores_total"] == 1


def test_dedup_query_uses_exact_twin_threshold_and_param_filter():
    _reset_metrics()
    bank, retriever = FakeBank(), FakeRetriever(twins=[])
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A", temperature=0.73, top_p=0.91))
    assert len(retriever.calls) == 1
    call = retriever.calls[0]
    assert call["score_threshold"] == 0.999
    must = call["filter_dict"]["must"]
    keys = {c["key"]: c["match"]["value"] for c in must}
    assert keys["model"] == "m" and keys["backend"] == "ollama"
    assert keys["temp_bucket"] == "0.7"          # bucketed via _gw_bucket(0.73, 0.1)
    assert keys["top_p_bucket"] == "0.9"         # bucketed via _gw_bucket(0.91, 0.05)


def test_dedup_query_omits_absent_params():
    _reset_metrics()
    bank, retriever = FakeBank(), FakeRetriever(twins=[])
    cache = FakeCache(FakeCompressor(), retriever, bank)
    run(store(cache, "q", "m", "A"))               # no temperature/top_p
    must = retriever.calls[0]["filter_dict"]["must"]
    keys = {c["key"] for c in must}
    assert "temp_bucket" not in keys and "top_p_bucket" not in keys


def test_compress_failure_never_calls_dedup_or_store():
    _reset_metrics()
    bank, retriever = FakeBank(), FakeRetriever(twins=[])
    cache = FakeCache(FakeCompressor(raises=True), retriever, bank)
    run(store(cache, "q", "m", "A"))
    assert len(retriever.calls) == 0
    assert len(bank.stored) == 0
    assert NS["_GW_METRICS"]["store_failures_total"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for t in tests:
        try:
            t(); p += 1
        except AssertionError as e:
            f += 1; print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:
            f += 1; print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{p} passed, {f} failed ({len(tests)} test fns, source: {os.path.relpath(SRC_PATH, HERE)})")
    sys.exit(1 if f else 0)
