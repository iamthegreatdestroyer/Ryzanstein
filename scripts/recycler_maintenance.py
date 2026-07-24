#!/usr/bin/env python3
"""Token Recycler cache maintenance (Kimi P1: invalidation + TTL enforcement).

Talks directly to Qdrant (no gateway restart needed) to purge RSUs the recycler
can no longer serve, closing two gaps in the live cache:

  * TTL is only enforced lazily: lookup() treats an RSU older than TTL as a miss,
    and step-4 delete-on-expire removes an expired RSU only when it is RETRIEVED
    as a candidate. An RSU that never matches a lookup (e.g. a pre-step-4 entry
    with no temp_bucket, which the param-aware Qdrant filter always excludes) is
    never retrieved, so it is never deleted -- it lives forever. This enforces the
    TTL for those never-retrieved RSUs.
  * Pre-step-4 RSUs have no temp_bucket and are dead weight for the /v1 path.

Scopes:
  expired    (filter) age > TTL (incl. legacy _created_ts=0) -- always safe: these
             are already unservable (lookup() misses on them). RECOMMENDED default.
  legacy     (filter) RSUs with no temp_bucket (pre-step-4 / param-agnostic). More
             aggressive: also removes param-agnostic /api entries that native
             callers without options could still legitimately hit -- opt in only.
  oversized  (id-list) VectorBank.store() is a bare insert with no cap (P1 gap;
             partially mitigated by dedup-on-store, but a cap is still the real
             backstop against unbounded growth). If the collection exceeds --cap
             (default 5000), deletes the OLDEST excess points by _created_ts,
             keeping the most-recently-created --cap. Legacy points with no
             _created_ts sort as epoch 0 (oldest) so they're trimmed first.
  all        (filter) every RSU (full flush).

Dry-run by default (Qdrant `count`/scroll, non-destructive). --apply performs the
delete after writing an audit of the matched (id, prompt, model, _created_ts) to
/tmp/recycler-purge-<scope>-<ts>.json. Cache data is regenerable, so the audit is
a record, not a restore path.

    python3 recycler_maintenance.py                        # dry-run, all scopes
    python3 recycler_maintenance.py --scope expired --apply
    python3 recycler_maintenance.py --scope oversized --cap 5000 --apply
"""
import argparse
import json
import time
import urllib.request

def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)

def scope_filter(scope, ttl, now):
    if scope == "expired":
        return {"must": [{"key": "_created_ts", "range": {"lt": now - ttl}}]}
    if scope == "legacy":
        return {"must": [{"is_empty": {"key": "temp_bucket"}}]}
    if scope == "all":
        # matches every point that has _created_ts (all recycler RSUs do)
        return {"must": [{"key": "_created_ts", "range": {"gte": 0}}]}
    raise SystemExit(f"unknown scope: {scope}")

def count(base, coll, filt):
    return _post(f"{base}/collections/{coll}/points/count", {"filter": filt, "exact": True})["result"]["count"]

def total(base, coll):
    return _get(f"{base}/collections/{coll}")["result"]["points_count"]

def scroll_all_ids_and_ts(base, coll):
    """All (id, _created_ts) pairs in the collection, no filter (unbounded scroll,
    payload-only, no vectors -- cheap). Missing/legacy _created_ts sorts as 0."""
    rows, offset = [], None
    while True:
        body = {"limit": 512, "with_payload": ["_created_ts"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = _post(f"{base}/collections/{coll}/points/scroll", body)["result"]
        for p in res["points"]:
            rows.append((p["id"], (p.get("payload") or {}).get("_created_ts", 0) or 0))
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return rows

def oversized_ids(base, coll, cap):
    """Ids of the oldest points beyond --cap (empty list if under cap)."""
    rows = scroll_all_ids_and_ts(base, coll)
    if len(rows) <= cap:
        return []
    rows.sort(key=lambda r: r[1])              # oldest (smallest _created_ts) first
    return [rid for rid, _ts in rows[: len(rows) - cap]]

def audit(base, coll, filt, path):
    # scroll matched points (no vectors) and record what will be deleted
    rows, offset = [], None
    while True:
        body = {"filter": filt, "limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = _post(f"{base}/collections/{coll}/points/scroll", body)["result"]
        for p in res["points"]:
            pl = p.get("payload", {})
            rows.append({"id": p["id"], "prompt": (pl.get("prompt", "") or "")[:200],
                         "model": pl.get("model", ""), "_created_ts": pl.get("_created_ts"),
                         "has_temp_bucket": "temp_bucket" in pl})
        offset = res.get("next_page_offset")
        if offset is None:
            break
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return len(rows)

def resolve_filter(scope, ttl, now, base, coll, cap):
    if scope == "oversized":
        ids = oversized_ids(base, coll, cap)
        return {"must": [{"has_id": ids}]} if ids else None   # None = nothing to do
    return scope_filter(scope, ttl, now)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["expired", "legacy", "oversized", "all"], default=None)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run count)")
    ap.add_argument("--ttl", type=int, default=86400)
    ap.add_argument("--cap", type=int, default=5000, help="oversized: max points to keep (newest wins)")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--collection", default="rsu_bank")
    ap.add_argument("--now", type=float, default=None)
    a = ap.parse_args()
    now = a.now if a.now is not None else time.time()
    base, coll = a.qdrant.rstrip("/"), a.collection

    print(f"collection={coll}  total points={total(base, coll)}  ttl={a.ttl}s  cap={a.cap}")
    if not a.apply:
        print("DRY-RUN (no deletes). Counts per scope:")
        for sc in ("expired", "legacy", "oversized", "all"):
            filt = resolve_filter(sc, a.ttl, now, base, coll, a.cap)
            n = count(base, coll, filt) if filt is not None else 0
            print(f"  {sc:9s} -> {n}")
        print("Re-run with --scope <s> --apply to delete.")
        return

    if a.scope is None:
        raise SystemExit("--apply requires --scope")
    filt = resolve_filter(a.scope, a.ttl, now, base, coll, a.cap)
    if filt is None:
        print(f"APPLY scope={a.scope}: nothing to do (collection within bounds)")
        return
    n = count(base, coll, filt)
    apath = f"/tmp/recycler-purge-{a.scope}-{int(now)}.json"
    audited = audit(base, coll, filt, apath)
    print(f"APPLY scope={a.scope}: matched {n} points, audit written to {apath} ({audited} rows)")
    _post(f"{base}/collections/{coll}/points/delete", {"filter": filt})
    print(f"deleted. total points now: {total(base, coll)}")

if __name__ == "__main__":
    main()
