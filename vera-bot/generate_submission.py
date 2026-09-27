#!/usr/bin/env python3
"""
generate_submission.py
=======================
Produces `submission.jsonl` (one line per test pair), per challenge-brief.md §7.2.

IMPORTANT: the real challenge dataset defines a canonical set of 30 (merchant, trigger)
test pairs that every participant must produce a message for (see challenge-brief.md §6:
"A canonical 'submission test set' is 30 specific (merchant, trigger) pairs"). That
specific 30-pair list was NOT part of what's been uploaded to this session yet — only
the base brief/testing-brief/design docs and judge_simulator.py came through, not the
actual dataset/ folder contents.

This script currently runs against the PLACEHOLDER dataset in dataset/ (10 merchants,
15 triggers, 6 customers) and emits one line per trigger (matched to its merchant, plus
customer where the trigger scope is "customer"). That's enough to prove the full
pipeline works end-to-end.

TO PRODUCE YOUR REAL SUBMISSION:
1. Replace dataset/categories/*.json, dataset/merchants_seed.json,
   dataset/customers_seed.json, dataset/triggers_seed.json with the real files from
   the challenge package.
2. If the real package includes an explicit "canonical test set" file (e.g.
   `test_pairs.json` or similar, listing the 30 required (merchant_id, trigger_id)
   pairs with test_id like "T01".."T30") drop it in dataset/ and this script will
   prefer it automatically (see `load_canonical_pairs()` below — update the filename
   there once you see what it's actually called).
3. Re-run: python generate_submission.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm_composer  # noqa: E402

DATASET_DIR = Path(__file__).parent / "dataset"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    categories = {}
    for f in (DATASET_DIR / "categories").glob("*.json"):
        data = load_json(f)
        categories[data["slug"]] = data

    merchants = {m["merchant_id"]: m for m in load_json(DATASET_DIR / "merchants_seed.json")["merchants"]}
    customers = {c["customer_id"]: c for c in load_json(DATASET_DIR / "customers_seed.json")["customers"]}
    triggers = {t["id"]: t for t in load_json(DATASET_DIR / "triggers_seed.json")["triggers"]}
    return categories, merchants, customers, triggers


def load_canonical_pairs():
    """
    Looks for an explicit canonical-test-set file. Update this filename once you know
    what the real package calls it (it wasn't part of what was uploaded to this
    session). Returns None if not found, in which case we fall back to "one line per
    trigger currently loaded".
    """
    for candidate in ("test_pairs.json", "canonical_test_set.json", "submission_test_set.json"):
        p = DATASET_DIR / candidate
        if p.exists():
            return load_json(p)
    return None


def main():
    categories, merchants, customers, triggers = load_dataset()
    canonical = load_canonical_pairs()

    if canonical:
        pairs = [(p["test_id"], p["merchant_id"], p["trigger_id"], p.get("customer_id")) for p in canonical]
        print(f"Using canonical test set: {len(pairs)} pairs")
    else:
        pairs = []
        for i, (tid, trig) in enumerate(sorted(triggers.items()), start=1):
            mid = trig.get("merchant_id")
            cid = trig.get("customer_id") if trig.get("scope") == "customer" else None
            pairs.append((f"T{i:02d}", mid, tid, cid))
        print(f"WARNING: no canonical test set file found — using all {len(pairs)} placeholder "
              f"triggers as a stand-in. Replace with the real 30-pair set before final submission.")

    out_path = Path(__file__).parent / "submission.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for test_id, mid, tid, cid in pairs:
            merchant = merchants.get(mid)
            trigger = triggers.get(tid)
            if not merchant or not trigger:
                print(f"  skip {test_id}: missing merchant={mid} or trigger={tid}")
                continue
            category = categories.get(merchant.get("category_slug"))
            if not category:
                print(f"  skip {test_id}: missing category for merchant={mid}")
                continue
            customer = customers.get(cid) if cid else None

            composed = llm_composer.compose_message(category, merchant, trigger, customer)

            line = {
                "test_id": test_id,
                "body": composed["body"],
                "cta": composed["cta"],
                "send_as": composed["send_as"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
            print(f"  {test_id}: {composed['body'][:70]!r}")

    print(f"\nWrote {written} lines to {out_path}")
    if llm_composer.DEV_MODE:
        print("NOTE: ANTHROPIC_API_KEY was not set, so these were generated by the DEV_MODE "
              "templated fallback, NOT the real LLM composer. Set ANTHROPIC_API_KEY and re-run "
              "before actually submitting — the placeholder text will score very poorly.")


if __name__ == "__main__":
    main()
