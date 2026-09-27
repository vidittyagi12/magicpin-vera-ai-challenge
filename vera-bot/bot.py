"""
bot.py — Vera challenge bot (magicpin AI Challenge)
====================================================
Implements the 5-endpoint contract from challenge-testing-brief.md:

    POST /v1/context   — idempotent context push (category/merchant/trigger/customer)
    POST /v1/tick       — periodic wake-up; bot may proactively initiate conversations
    POST /v1/reply      — synchronous reply to an inbound merchant/customer message
    GET  /v1/healthz    — liveness probe
    GET  /v1/metadata   — bot identity

Run locally:
    export ANTHROPIC_API_KEY=sk-ant-...           # omit to run in DEV_MODE (templated, offline)
    uvicorn bot:app --host 0.0.0.0 --port 8080

Deploy on Render: see render.yaml / README.md.
"""

import os
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import llm_composer
import conversation_handlers as ch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vera.bot")

app = FastAPI(title="Vera Challenge Bot")
START_TIME = time.time()

MAX_ACTIONS_PER_TICK = 20  # per testing-brief §5

# ---------------------------------------------------------------------------
# In-memory state. Per the spec this is acceptable ("Storing in memory is
# fine; just don't restart between calls."). Guarded by a single lock since
# the judge harness caps at 10 req/s — no need for anything fancier.
# ---------------------------------------------------------------------------

_lock = threading.RLock()

# (scope, context_id) -> {"version": int, "payload": dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id -> ConversationState (see conversation_handlers.py)
conversations: dict[str, dict] = {}

# merchant_id -> set of currently-open (unended) conversation_ids, to avoid
# spamming the same merchant with a second thread while one is already live.
open_conversations_by_merchant: dict[str, set] = {}

# suppression_key -> last-used-at (epoch seconds); used for tick-level dedup so
# we don't re-fire the same trigger repeatedly across ticks.
used_suppression_keys: dict[str, float] = {}


def _get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _resolve_category_for_merchant(merchant: dict) -> Optional[dict]:
    slug = merchant.get("category_slug")
    if not slug:
        return None
    return _get_ctx("category", slug)


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    with _lock:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _cid) in contexts.keys():
            counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Team Vera"),
        "team_members": os.environ.get("TEAM_MEMBERS", "").split(",") if os.environ.get("TEAM_MEMBERS") else [],
        "model": llm_composer.ANTHROPIC_MODEL if not llm_composer.DEV_MODE else "DEV_MODE (no LLM configured)",
        "approach": (
            "Single Claude-backed composer (temperature=0) driven by a shared rubric-and-voice "
            "system prompt covering all 5 judged dimensions. A lightweight deterministic layer "
            "handles auto-reply detection (verbatim-repeat counting), hostile-message handling, "
            "and intent-transition detection (commitment-phrase regex) before falling through to "
            "the LLM for actual message composition and reply decisions. Anti-repetition is "
            "enforced by passing already-sent bodies into every prompt plus a one-shot retry."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", ""),
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    with _lock:
        cur = contexts.get(key)
        if cur and cur["version"] >= body.version:
            return {"accepted": False, "reason": "stale_version",
                     "current_version": cur["version"]}
        contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    with _lock:
        # Rank by urgency (desc); highest-value sends first, respecting the cap.
        candidates = []
        for trig_id in body.available_triggers:
            trigger = _get_ctx("trigger", trig_id)
            if not trigger:
                continue
            candidates.append((trigger.get("urgency", 1), trig_id, trigger))
        candidates.sort(key=lambda t: t[0], reverse=True)

        for _urgency, trig_id, trigger in candidates:
            if len(actions) >= MAX_ACTIONS_PER_TICK:
                break

            suppression_key = trigger.get("suppression_key", "") or trig_id
            expires_at = trigger.get("expires_at")
            if expires_at:
                try:
                    exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) > exp:
                        continue  # trigger stale, skip
                except ValueError:
                    pass

            if suppression_key in used_suppression_keys:
                continue  # already fired this trigger before — dedup

            merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
            if not merchant_id:
                continue
            merchant = _get_ctx("merchant", merchant_id)
            if not merchant:
                continue
            category = _resolve_category_for_merchant(merchant)
            if not category:
                continue

            customer = None
            if trigger.get("scope") == "customer":
                customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
                if customer_id:
                    customer = _get_ctx("customer", customer_id)
                if not customer:
                    continue  # can't do a customer-facing send without customer context

            # Restraint: don't open a second simultaneous thread with a merchant
            # that already has one in flight (spam avoidance, rewarded per rubric).
            open_convs = open_conversations_by_merchant.get(merchant_id, set())
            if open_convs and customer is None:
                continue

            composed = llm_composer.compose_message(
                category=category, merchant=merchant, trigger=trigger, customer=customer,
                already_sent=[],
            )
            if not composed.get("body"):
                continue  # nothing worth sending

            conv_id = f"conv_{merchant_id}_{trig_id}"
            state = ch.new_state(
                conversation_id=conv_id, merchant_id=merchant_id,
                customer_id=customer.get("customer_id") if customer else None,
                trigger_id=trig_id, category_slug=merchant.get("category_slug", ""),
                suppression_key=suppression_key,
            )
            ch.record_bot_send(state, composed["body"], composed["cta"])
            conversations[conv_id] = state
            open_conversations_by_merchant.setdefault(merchant_id, set()).add(conv_id)
            used_suppression_keys[suppression_key] = time.time()

            name = (merchant.get("identity") or {}).get("name", merchant_id)
            actions.append({
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer.get("customer_id") if customer else None,
                "send_as": composed["send_as"],
                "trigger_id": trig_id,
                "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
                "template_params": [name, str(trigger.get("kind", "")), composed["body"][:60]],
                "body": composed["body"],
                "cta": composed["cta"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    with _lock:
        state = conversations.get(body.conversation_id)
        if state is None:
            # Judge referenced a conversation we don't know about (shouldn't happen
            # per spec, but be defensive) — bootstrap minimal state so we can still
            # respond sensibly instead of erroring out.
            merchant = _get_ctx("merchant", body.merchant_id) if body.merchant_id else None
            state = ch.new_state(
                conversation_id=body.conversation_id,
                merchant_id=body.merchant_id or "",
                customer_id=body.customer_id,
                trigger_id="",
                category_slug=(merchant or {}).get("category_slug", ""),
                suppression_key="",
            )
            conversations[body.conversation_id] = state

        merchant = _get_ctx("merchant", state["merchant_id"]) or {}
        category = _resolve_category_for_merchant(merchant) or {}
        trigger = _get_ctx("trigger", state.get("trigger_id", "")) or {}
        customer = _get_ctx("customer", state["customer_id"]) if state.get("customer_id") else None

        result = ch.respond(state, body.message, category, merchant, trigger, customer)

        if state.get("ended"):
            open_conversations_by_merchant.get(state["merchant_id"], set()).discard(body.conversation_id)

    # Shape the response strictly to the 3 documented action variants.
    if result["action"] == "send":
        return {"action": "send", "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"), "rationale": result.get("rationale", "")}
    elif result["action"] == "wait":
        return {"action": "wait", "wait_seconds": result.get("wait_seconds", 1800),
                "rationale": result.get("rationale", "")}
    else:
        return {"action": "end", "rationale": result.get("rationale", "")}


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional, per testing-brief §11 privacy requirement)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    with _lock:
        contexts.clear()
        conversations.clear()
        open_conversations_by_merchant.clear()
        used_suppression_keys.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
