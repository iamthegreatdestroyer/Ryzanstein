"""Tests for step-4 candidate selection — extracts the REAL functions from
src/api/server.py via AST and execs them with lightweight stubs, so it exercises
the shipped code (`_gw_select_recyclable`, `_gw_digit_multiset`,
`_gw_digits_compatible`, `_GwSelectionOutcome`) without importing torch/fastapi.

    pytest tests/test_step4_select.py   |   python tests/test_step4_select.py
"""
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

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
WANT = {"_gw_digit_multiset", "_gw_digits_compatible", "_GwSelectionOutcome", "_gw_select_recyclable"}

NS = {
    "_gw_re": re,
    "_GW_DIGIT_RE": re.compile(r"\d+"),      # module-level assign, provided directly
    "_gw_dataclass": dataclass,
    "_gw_field": field,
    "Any": Any, "Dict": Dict, "List": List,
    "__builtins__": __builtins__,
}
_LINES = SOURCE.splitlines()


def _node_source(node):
    # ast.get_source_segment EXCLUDES decorators; extract by line range from the
    # first decorator so @_gw_dataclass is preserved (else the class isn't a dataclass).
    start = node.lineno - 1
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list) - 1
    return "\n".join(_LINES[start:node.end_lineno])


_found = set()
for _n in TREE.body:
    if isinstance(_n, (ast.FunctionDef, ast.ClassDef)) and _n.name in WANT:
        exec(compile(_node_source(_n), SRC_PATH, "exec"), NS)
        _found.add(_n.name)
assert not (WANT - _found), f"units not found: {WANT - _found}"

select = NS["_gw_select_recyclable"]
digit_multiset = NS["_gw_digit_multiset"]
digits_compatible = NS["_gw_digits_compatible"]

NOW, TTL, TWIN, MARGIN, QP = 1_000_000.0, 86_400, 0.999, 0.01, "user: hello"


@dataclass
class Cand:
    rsu_id: str
    score: float
    answer: str = "A"
    prompt: str = QP
    metadata: Dict[str, Any] = field(default_factory=lambda: {"_created_ts": NOW})


def sel(cands, qp=QP, mt=None):
    return select(cands, qp, NOW, TTL, TWIN, MARGIN, request_max_tokens=mt)


def test_lexical_helpers():
    assert digit_multiset("who won 2020") == ("2020",)
    assert digits_compatible("won 2020", "who won 2020") is True
    assert digits_compatible("won 2020", "won 2024") is False
    assert digits_compatible("top 10", "ten best") is False


def test_empty():
    assert sel([]).reason == "no_candidates"


def test_single_lone_accepts():
    assert sel([Cand("a", 0.992)]).chosen.rsu_id == "a"


def test_exact_twin_beats_margin():
    o = sel([Cand("a", 0.9995, answer="X"), Cand("b", 0.9994, answer="Y")])
    assert o.chosen.rsu_id == "a"


def test_clear_margin_accepts():
    assert sel([Cand("a", 0.993, answer="X"), Cand("b", 0.980, answer="Y")]).chosen.rsu_id == "a"


def test_ambiguous_rejects_and_skips_l2():
    o = sel([Cand("a", 0.9925, answer="X"), Cand("b", 0.9920, answer="Y")])
    assert o.chosen is None and o.reason == "ambiguous" and o.allow_l2_store is False


def test_same_answer_twins_still_hit():
    o = sel([Cand("a", 0.9925, answer="S"), Cand("b", 0.9920, answer="S")])
    assert o.chosen.rsu_id == "a" and o.reason == "hit"


def test_margin_vs_first_different_answer():
    o = sel([Cand("a", 0.9930, answer="X"), Cand("b", 0.9928, answer="X"), Cand("c", 0.9700, answer="Y")])
    assert o.chosen.rsu_id == "a"


def test_close_different_after_twin_rejects():
    o = sel([Cand("a", 0.9930, answer="X"), Cand("b", 0.9928, answer="X"), Cand("c", 0.9925, answer="Y")])
    assert o.chosen is None and o.reason == "ambiguous"


def test_expired_falls_through():
    o = sel([Cand("a", 0.995, metadata={"_created_ts": NOW - 2 * TTL}), Cand("b", 0.991)])
    assert o.chosen.rsu_id == "b" and o.expired_ids == ["a"]


def test_all_expired_miss():
    o = sel([Cand("a", 0.995, metadata={"_created_ts": 0}), Cand("b", 0.994, metadata={"_created_ts": 0})])
    assert o.chosen is None and set(o.expired_ids) == {"a", "b"} and o.counts["expired_filtered"] == 2


def test_ttl_zero_never_expires():
    o = select([Cand("a", 0.9995, metadata={"_created_ts": 0})], QP, NOW, 0, TWIN, MARGIN)
    assert o.chosen.rsu_id == "a" and o.expired_ids == []


def test_bounded_deletes():
    cands = [Cand(f"e{i}", 0.99, metadata={"_created_ts": 0}) for i in range(20)]
    o = select(cands, QP, NOW, TTL, TWIN, MARGIN, max_deletes=8)
    assert len(o.expired_ids) == 8 and o.counts["expired_filtered"] == 20


def test_digit_swap_rejected():
    o = sel([Cand("a", 0.9995, prompt="who won in 2020")], qp="who won in 2024")
    assert o.chosen is None and o.counts["lexical_rejected"] == 1


def test_matching_digits_pass():
    assert sel([Cand("a", 0.992, prompt="revenue in 2024")], qp="revenue in 2024").chosen.rsu_id == "a"


def test_query_digit_vs_none_rejected():
    o = sel([Cand("a", 0.9995, prompt="capital city")], qp="capital in 2024")
    assert o.chosen is None and o.counts["lexical_rejected"] == 1


def test_maxtok_smaller_rejected():
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": 512})], mt=128)
    assert o.chosen is None and o.counts["maxtok_rejected"] == 1


def test_maxtok_ge_ok():
    assert sel([Cand("a", 0.992, metadata={"_created_ts": NOW, "max_tokens": 128})], mt=512).chosen.rsu_id == "a"


def test_maxtok_absent_no_constraint():
    assert sel([Cand("a", 0.992, metadata={"_created_ts": NOW})], mt=16).chosen.rsu_id == "a"


# --- R1 sentinel / non-int hardening (step-4 review findings) --------------
def test_maxtok_unlimited_request_serves_any():
    # Ollama num_predict=-1 (unlimited) must serve any complete answer, not reject
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": 256})], mt=-1)
    assert o.chosen.rsu_id == "a"


def test_maxtok_unlimited_stored_not_served_to_finite():
    # a -1-stored (unlimited-budget) answer must NOT be served to a finite budget
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": -1})], mt=50)
    assert o.chosen is None and o.counts["maxtok_rejected"] == 1


def test_maxtok_unlimited_both_serves():
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": -1})], mt=-1)
    assert o.chosen.rsu_id == "a"


def test_maxtok_noninteger_request_no_crash():
    # a string num_predict must not raise (would 500); treated as unlimited
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": 256})], mt="256")
    assert o.chosen.rsu_id == "a"


def test_maxtok_noninteger_stored_conservative_reject():
    o = sel([Cand("a", 0.9995, metadata={"_created_ts": NOW, "max_tokens": "foo"})], mt=50)
    assert o.chosen is None and o.counts["maxtok_rejected"] == 1


def test_allow_l2_true_on_normal_miss():
    assert sel([]).allow_l2_store is True
    assert sel([Cand("a", 0.995, metadata={"_created_ts": 0})]).allow_l2_store is True


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
