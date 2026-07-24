"""Unit tests for the Token Recycler L1 exact-match cache (server.py, PR-1).

Tests the REAL source of three pure units in src/api/server.py --
`_gw_answer_is_complete`, `class _ExactMatchL1Cache`, and `_gw_l1_key` -- by
extracting their exact source segments via AST and exec'ing them in a namespace
with lightweight stubs for the gateway's module-level aliases. This exercises the
shipped code characters without importing the full server (torch / fastapi /
qdrant), so it runs under bare `python` on any node.

Run either way:
    pytest tests/test_l1_cache.py         # pytest-native
    python  tests/test_l1_cache.py        # standalone (exit 0 = all pass)
"""
import ast
import json
import hashlib
import os
import sys
from collections import OrderedDict
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "src", "api", "server.py"),   # repo layout: tests/ next to src/
    os.path.join(HERE, "src", "api", "server.py"),
    os.path.join(HERE, "live_server.py"),                  # scratchpad layout
    os.path.join(HERE, "server.py"),
]
SRC_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert SRC_PATH, f"could not locate server source in {CANDIDATES}"

with open(SRC_PATH, encoding="utf-8") as fh:
    SOURCE = fh.read()
TREE = ast.parse(SOURCE)
WANT = {"_gw_answer_is_complete", "_ExactMatchL1Cache", "_gw_l1_key"}


class _FakeClock:
    """Controllable stand-in for the `time` module (only .time() is used)."""
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now


CLOCK = _FakeClock()

# Namespace mirroring the module-level aliases the extracted units reference.
_NS = {
    "_gw_time": CLOCK,
    "_gw_json": json,
    "_gw_hashlib": hashlib,
    "_GwOrderedDict": OrderedDict,
    "BACKEND": "ollama",
    "Optional": Optional,
    "__builtins__": __builtins__,
}

_found = set()
for _node in TREE.body:
    if isinstance(_node, (ast.FunctionDef, ast.ClassDef)) and _node.name in WANT:
        _seg = ast.get_source_segment(SOURCE, _node)
        assert _seg, f"no source segment for {_node.name}"
        exec(compile(_seg, SRC_PATH, "exec"), _NS)
        _found.add(_node.name)
assert not (WANT - _found), f"units not found in source: {WANT - _found}"

answer_is_complete = _NS["_gw_answer_is_complete"]
L1Cache = _NS["_ExactMatchL1Cache"]
l1_key = _NS["_gw_l1_key"]


class FakeMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeReq:
    """Mimics ChatCompletionRequest: .messages (typed) + .model_dump() (dict)."""
    def __init__(self, messages, model="m", temperature=0.7, max_tokens=256,
                 top_p=0.95, stream=False, **extra):
        self.messages = messages
        self._dump = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p,
            "stream": stream, **extra,
        }

    def model_dump(self):
        return dict(self._dump)


# === _gw_answer_is_complete ==============================================
def test_completeness_gate():
    assert answer_is_complete("stop") is True
    assert answer_is_complete(None) is True            # null-equivalent
    assert answer_is_complete("length") is False       # truncated -> never cache
    assert answer_is_complete("content_filter") is False
    assert answer_is_complete("tool_calls") is False
    assert answer_is_complete("") is False


# === _ExactMatchL1Cache ==================================================
def test_l1_basic_hit_miss():
    CLOCK.now = 1_000_000.0
    c = L1Cache(max_entries=3, ttl=100)
    assert c.get("k") is None and c.misses == 1
    c.put("k", "answer-1")
    assert c.get("k") == "answer-1" and c.hits == 1


def test_l1_ttl_expiry_and_delete():
    c = L1Cache(max_entries=3, ttl=100)
    CLOCK.now = 2_000.0
    c.put("t", "v")
    CLOCK.now = 2_050.0
    assert c.get("t") == "v"                            # within ttl
    CLOCK.now = 2_101.0                                 # 101s > ttl 100
    assert c.get("t") is None
    assert c.expired == 1
    assert "t" not in c._d                              # delete-on-expire


def test_l1_ttl_zero_never_expires():
    c = L1Cache(max_entries=3, ttl=0)
    CLOCK.now = 5_000.0
    c.put("z", "v")
    CLOCK.now = 9_999_999.0
    assert c.get("z") == "v"


def test_l1_lru_eviction():
    CLOCK.now = 10.0
    e = L1Cache(max_entries=2, ttl=1000)
    e.put("a", "1"); e.put("b", "2"); e.put("c", "3")   # evicts oldest "a"
    assert e.get("a") is None
    assert e.get("b") == "2" and e.get("c") == "3"
    assert e.evictions == 1


def test_l1_lru_touch_on_get():
    t = L1Cache(max_entries=2, ttl=1000)
    t.put("a", "1"); t.put("b", "2")
    assert t.get("a") == "1"                             # touch a -> a now MRU
    t.put("c", "3")                                      # evicts b, not a
    assert t.get("a") == "1"
    assert t.get("b") is None


def test_l1_overwrite_no_dup():
    r = L1Cache(max_entries=2, ttl=1000)
    r.put("a", "1"); r.put("a", "2")
    assert r.get("a") == "2"
    assert len(r._d) == 1


def test_l1_stats_shape():
    c = L1Cache(max_entries=2, ttl=1000)
    s = c.stats()
    assert s["enabled"] is True
    assert {"entries", "hits", "misses", "evictions", "expired"} <= set(s)


# === _gw_l1_key ==========================================================
_MSGS = [FakeMsg("system", "be terse"), FakeMsg("user", "hi")]
_BASE = l1_key("served-model", FakeReq(_MSGS))


def test_key_is_sha256_hex_and_deterministic():
    assert isinstance(_BASE, str) and len(_BASE) == 64
    assert l1_key("served-model", FakeReq(list(_MSGS))) == _BASE


def test_key_param_sensitivity():
    assert l1_key("served-model", FakeReq(_MSGS, temperature=0.0)) != _BASE
    assert l1_key("served-model", FakeReq(_MSGS, max_tokens=2048)) != _BASE
    assert l1_key("served-model", FakeReq(_MSGS, top_p=0.5)) != _BASE
    assert l1_key("OTHER-model", FakeReq(_MSGS)) != _BASE


def test_key_future_param_auto_folds():
    # a sampling field not present today (reflected from model_dump) still keys
    assert l1_key("served-model", FakeReq(_MSGS, seed=42)) != _BASE


def test_key_stream_excluded():
    assert l1_key("served-model", FakeReq(_MSGS, stream=True)) == _BASE


def test_key_message_order_and_content_exact():
    rev = [FakeMsg("user", "hi"), FakeMsg("system", "be terse")]
    assert l1_key("served-model", FakeReq(rev)) != _BASE                 # order is semantic
    ws = [FakeMsg("system", "be terse"), FakeMsg("user", "hi ")]         # trailing space
    assert l1_key("served-model", FakeReq(ws)) != _BASE                  # byte-exact content
    role = [FakeMsg("system", "be terse"), FakeMsg("assistant", "hi")]
    assert l1_key("served-model", FakeReq(role)) != _BASE                # role matters


def test_key_lone_surrogate_does_not_crash():
    # regression (adversarial review, PR-1): a lone UTF-16 surrogate in content
    # must NOT raise UnicodeEncodeError (surrogatepass), stays stable + distinct.
    k_sur = l1_key("served-model", FakeReq([FakeMsg("system", "be terse"), FakeMsg("user", "\ud800")]))
    assert isinstance(k_sur, str) and len(k_sur) == 64
    k_sur2 = l1_key("served-model", FakeReq([FakeMsg("system", "be terse"), FakeMsg("user", "\ud800")]))
    assert k_sur == k_sur2
    assert k_sur != _BASE


# === standalone runner (no pytest needed) ================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:  # pragma: no cover
            failed += 1
            print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed  ({len(tests)} test fns, source: {os.path.relpath(SRC_PATH, HERE)})")
    sys.exit(1 if failed else 0)
