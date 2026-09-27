#!/usr/bin/env python3
"""
test_bot_local.py
==================
Fast, offline smoke test using FastAPI's TestClient (in-process ASGI calls —
no real HTTP, no network, no API key needed). Runs the DEV_MODE templated
composer. This is NOT a substitute for running judge_simulator.py with a real
ANTHROPIC_API_KEY and a real deployed URL — it only proves the endpoints are
wired correctly and return well-shaped responses.

Run: python test_bot_local.py
"""

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import bot  # noqa: E402

client = TestClient(bot.app)
DATASET_DIR = Path(__file__).parent / "dataset"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        FAILURES.append(msg)


FAILURES = []


def main():
    print("=== healthz / metadata before any context ===")
    r = client.get("/v1/healthz")
    check(r.status_code == 200, "GET /v1/healthz returns 200")
    check(r.json()["status"] == "ok", "healthz status == ok")

    r = client.get("/v1/metadata")
    check(r.status_code == 200, "GET /v1/metadata returns 200")
    check("team_name" in r.json(), "metadata has team_name")

    print("\n=== pushing base contexts ===")
    for f in (DATASET_DIR / "categories").glob("*.json"):
        data = load_json(f)
        r = client.post("/v1/context", json={
            "scope": "category", "context_id": data["slug"], "version": 1,
            "payload": data, "delivered_at": "2026-04-26T10:00:00Z",
        })
        check(r.status_code == 200 and r.json()["accepted"], f"pushed category {data['slug']}")

    merchants = load_json(DATASET_DIR / "merchants_seed.json")["merchants"]
    for m in merchants:
        r = client.post("/v1/context", json={
            "scope": "merchant", "context_id": m["merchant_id"], "version": 1,
            "payload": m, "delivered_at": "2026-04-26T10:00:00Z",
        })
        check(r.status_code == 200 and r.json()["accepted"], f"pushed merchant {m['merchant_id']}")

    customers = load_json(DATASET_DIR / "customers_seed.json")["customers"]
    for c in customers:
        r = client.post("/v1/context", json={
            "scope": "customer", "context_id": c["customer_id"], "version": 1,
            "payload": c, "delivered_at": "2026-04-26T10:00:00Z",
        })
        check(r.status_code == 200 and r.json()["accepted"], f"pushed customer {c['customer_id']}")

    print("\n=== idempotency check (re-post same version) ===")
    m0 = merchants[0]
    r = client.post("/v1/context", json={
        "scope": "merchant", "context_id": m0["merchant_id"], "version": 1,
        "payload": m0, "delivered_at": "2026-04-26T10:00:00Z",
    })
    check(r.json()["accepted"] is False and r.json()["reason"] == "stale_version",
          "re-posting same version is rejected as stale_version")

    print("\n=== healthz reflects loaded contexts ===")
    r = client.get("/v1/healthz")
    counts = r.json()["contexts_loaded"]
    check(counts["category"] == 5, f"5 categories loaded (got {counts['category']})")
    check(counts["merchant"] == len(merchants), f"{len(merchants)} merchants loaded (got {counts['merchant']})")
    check(counts["customer"] == len(customers), f"{len(customers)} customers loaded (got {counts['customer']})")

    triggers = load_json(DATASET_DIR / "triggers_seed.json")["triggers"]
    for t in triggers:
        client.post("/v1/context", json={
            "scope": "trigger", "context_id": t["id"], "version": 1,
            "payload": t, "delivered_at": "2026-04-26T10:00:00Z",
        })

    print("\n=== /v1/tick ===")
    trig_ids = [t["id"] for t in triggers]
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": trig_ids})
    check(r.status_code == 200, "POST /v1/tick returns 200")
    actions = r.json().get("actions", [])
    check(len(actions) > 0, f"tick produced {len(actions)} actions (expected > 0)")
    for a in actions:
        for key in ("conversation_id", "merchant_id", "send_as", "body", "cta", "suppression_key", "rationale"):
            check(key in a, f"action has key '{key}'")

    print("\n=== duplicate tick (same triggers) should NOT re-fire suppressed ones ===")
    r2 = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": trig_ids})
    actions2 = r2.json().get("actions", [])
    check(len(actions2) == 0, f"second tick with same triggers produced {len(actions2)} actions (expected 0, dedup)")

    if actions:
        conv = actions[0]
        print("\n=== /v1/reply — normal flow ===")
        r = client.post("/v1/reply", json={
            "conversation_id": conv["conversation_id"], "merchant_id": conv["merchant_id"],
            "customer_id": conv.get("customer_id"), "from_role": "merchant",
            "message": "Tell me more please", "received_at": "2026-04-26T10:45:00Z", "turn_number": 2,
        })
        check(r.status_code == 200, "POST /v1/reply returns 200")
        check(r.json().get("action") in ("send", "wait", "end"), "reply action is one of send/wait/end")

        print("\n=== /v1/reply — auto-reply detection (3x verbatim) ===")
        canned = "Thank you for contacting us, our team will get back to you shortly."
        conv_id2 = "conv_test_autoreply"
        client.post("/v1/tick", json={"now": "2026-04-26T11:00:00Z", "available_triggers": []})
        # Manually seed a conversation to test auto-reply logic in isolation
        import conversation_handlers as ch
        with bot._lock:
            bot.conversations[conv_id2] = ch.new_state(conv_id2, conv["merchant_id"], None, "", "dentists", "")
        for i in range(3):
            r = client.post("/v1/reply", json={
                "conversation_id": conv_id2, "merchant_id": conv["merchant_id"], "customer_id": None,
                "from_role": "merchant", "message": canned,
                "received_at": "2026-04-26T11:0{}:00Z".format(i), "turn_number": i + 1,
            })
        check(r.json().get("action") == "end", f"3rd verbatim repeat triggers action=end (got {r.json().get('action')})")

        print("\n=== /v1/reply — hostile message ===")
        conv_id3 = "conv_test_hostile"
        with bot._lock:
            bot.conversations[conv_id3] = ch.new_state(conv_id3, conv["merchant_id"], None, "", "dentists", "")
        r = client.post("/v1/reply", json={
            "conversation_id": conv_id3, "merchant_id": conv["merchant_id"], "customer_id": None,
            "from_role": "merchant", "message": "Stop messaging me. This is useless spam.",
            "received_at": "2026-04-26T11:10:00Z", "turn_number": 1,
        })
        check(r.json().get("action") == "end", f"hostile message triggers action=end (got {r.json().get('action')})")

        print("\n=== /v1/reply — intent transition ===")
        conv_id4 = "conv_test_intent"
        with bot._lock:
            st = ch.new_state(conv_id4, conv["merchant_id"], None, "", "dentists", "")
            st["stage"] = "qualifying"
            bot.conversations[conv_id4] = st
        r = client.post("/v1/reply", json={
            "conversation_id": conv_id4, "merchant_id": conv["merchant_id"], "customer_id": None,
            "from_role": "merchant", "message": "Yes let's do it, go ahead",
            "received_at": "2026-04-26T11:15:00Z", "turn_number": 1,
        })
        check(r.json().get("action") in ("send", "end"), "intent-transition reply is a valid action")

    print("\n=== /v1/teardown ===")
    r = client.post("/v1/teardown")
    check(r.status_code == 200, "POST /v1/teardown returns 200")
    r = client.get("/v1/healthz")
    check(sum(r.json()["contexts_loaded"].values()) == 0, "contexts cleared after teardown")

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED (DEV_MODE templated composer — set ANTHROPIC_API_KEY")
        print("and re-run against a live deployment before actually submitting).")
        sys.exit(0)


if __name__ == "__main__":
    main()
