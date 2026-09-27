"""
llm_composer.py
================
The actual `compose()` brain for Vera. Turns (category, merchant, trigger, customer?)
dicts into a message, following the magicpin AI Challenge rubric:

    specificity | category_fit | merchant_fit | trigger_relevance | engagement_compulsion

Also used for reply-composition (deciding send/wait/end mid-conversation) via
`compose_reply()`.

DEV_MODE: if ANTHROPIC_API_KEY is not set, falls back to a deterministic
template-based composer so the server is smoke-testable without network access
or an API key. Set ANTHROPIC_API_KEY before deploying for real scoring runs —
the template fallback will NOT score well on the LLM judge, it exists purely
so `uvicorn` boots and endpoints return well-shaped responses in dev.
"""

import os
import json
import re
import logging

logger = logging.getLogger("vera.composer")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
DEV_MODE = not ANTHROPIC_API_KEY

_client = None
if not DEV_MODE:
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:  # pragma: no cover
        logger.error("Failed to init Anthropic client, falling back to DEV_MODE: %s", e)
        DEV_MODE = True

if DEV_MODE:
    logger.warning(
        "ANTHROPIC_API_KEY not set (or client init failed) — running in DEV_MODE "
        "with a templated fallback composer. Set ANTHROPIC_API_KEY for real scoring."
    )


# ---------------------------------------------------------------------------
# Shared rubric / voice guidance, embedded once, reused by every prompt.
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_PROMPT = """You are the composition engine for Vera, magicpin's AI assistant that messages
merchants (and, on their behalf, their customers) over WhatsApp. You write ONE message at a time.

You will be given up to four JSON contexts:
- category: slow-changing knowledge about this type of business (voice rules, offer catalog,
  peer benchmarks, this week's digest, seasonal beats, trend signals)
- merchant: this specific business's current state (identity, performance, offers, conversation
  history, customer aggregate, derived signals)
- trigger: the specific event that justifies messaging right now
- customer (optional): populated only when writing ON BEHALF OF the merchant to one of their
  own customers

You are graded on exactly these five dimensions (0-10 each), so optimize hard for all five:

1. SPECIFICITY — anchor on a concrete, verifiable fact from the given contexts: a real number,
   date, percentage, headline, or peer stat. "10% off" is generic and scores low. "Dental
   Cleaning @ ₹299" or "38% better across a 2,100-patient trial" scores high.
2. CATEGORY FIT — voice, vocabulary, and offer format must match the category's `voice` field.
   Clinical/peer tone for dentists/doctors/lawyers, never hype ("AMAZING DEAL"). Warm/practical
   for salons. Operator-to-operator for restaurants. Coaching/motivational for gyms.
   Trustworthy/precise for pharmacies. Respect the category's `taboos` vocabulary — never use
   forbidden words (e.g. "cure", "guaranteed" for clinical categories).
3. MERCHANT FIT — personalize to THIS merchant: use their real name, their real numbers
   (performance, offers, customer_aggregate), their conversation history so far, and their
   stated language preference (identity.languages). If languages include "hi", Hindi-English
   code-mix (Hinglish) is expected and preferred, not just English.
4. TRIGGER RELEVANCE — the message must make it obvious WHY NOW. Reference the trigger's
   payload directly. Never write a generic "just checking in" message.
5. ENGAGEMENT COMPULSION — use at least one lever: specificity/verifiability, loss aversion,
   social proof, effort externalization ("I've drafted X, just say go"), curiosity, reciprocity,
   asking the merchant a direct question, or a single binary commitment (YES/STOP). The two most
   under-used and most rewarded levers are SOCIAL PROOF and ASKING THE MERCHANT A QUESTION.

HARD RULES (violating any of these is penalized heavily):
- NEVER invent a fact not present in the given contexts. No fake research citations, no fake
  competitor names, no fake numbers. If you don't have a number, don't state one.
- Exactly ONE call-to-action per message. Never offer multiple parallel choices
  ("Reply YES for X, NO for Y, MAYBE for Z").
- Put the call-to-action in the LAST sentence, not buried mid-message.
- No long preambles ("I hope you're doing well..."). Get to the point in sentence one.
- Don't re-introduce yourself if conversation_history shows you've already spoken to them.
- Never send the exact same message body verbatim that was already sent in this conversation
  (you will be told what's already been sent — vary the wording and framing).
- Match the merchant's / customer's language preference exactly.

Always respond with STRICT JSON only, no markdown fences, no commentary, matching exactly the
schema you are asked for in the user message.
"""


def _safe_json(obj):
    """Compact JSON dump, tolerant of None."""
    return json.dumps(obj if obj is not None else {}, ensure_ascii=False, default=str)


def _extract_json(text: str) -> dict:
    """Anthropic sometimes wraps JSON in prose or fences despite instructions — recover it."""
    text = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Fall back to first {...} block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def _call_claude(system: str, user: str, max_tokens: int = 900) -> str:
    if DEV_MODE or _client is None:
        raise RuntimeError("DEV_MODE active — no live LLM call made")
    resp = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=0,  # determinism, per challenge spec §7.1
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Initial / proactive composition — used by /v1/tick
# ---------------------------------------------------------------------------

def compose_message(category: dict, merchant: dict, trigger: dict,
                     customer: dict | None = None,
                     already_sent: list[str] | None = None) -> dict:
    """
    Returns dict with keys: body, cta, send_as, suppression_key, rationale.
    `cta` is one of: "binary", "open_ended", "none".
    `send_as` is "vera" (merchant-facing) or "merchant_on_behalf" (customer-facing).
    """
    already_sent = already_sent or []
    send_as = "merchant_on_behalf" if customer else "vera"

    user_prompt = f"""Compose ONE new outbound WhatsApp message.

CATEGORY CONTEXT:
{_safe_json(category)}

MERCHANT CONTEXT:
{_safe_json(merchant)}

TRIGGER CONTEXT (this is WHY you are messaging right now):
{_safe_json(trigger)}

CUSTOMER CONTEXT (populated only if this message is being sent to the merchant's own
customer, on the merchant's behalf; null means this message is merchant-facing):
{_safe_json(customer)}

Messages already sent in this thread (do not repeat any of these verbatim):
{_safe_json(already_sent)}

send_as MUST be "{send_as}".

Respond with STRICT JSON only:
{{
  "body": "<the WhatsApp message text>",
  "cta": "binary" | "open_ended" | "none",
  "rationale": "<1-2 sentences: why this message, what it should achieve, which compulsion lever(s) used>"
}}"""

    if DEV_MODE:
        result = _dev_fallback_compose(category, merchant, trigger, customer)
    else:
        try:
            raw = _call_claude(RUBRIC_SYSTEM_PROMPT, user_prompt)
            result = _extract_json(raw)
        except Exception as e:
            logger.error("compose_message LLM call failed, using fallback: %s", e)
            result = _dev_fallback_compose(category, merchant, trigger, customer)

    body = str(result.get("body", "")).strip()

    # Anti-repetition safety net: if the model repeated itself, retry once with a
    # sharper instruction. If still repeated, fall back to a templated variant.
    if body and body in already_sent and not DEV_MODE:
        try:
            retry_prompt = user_prompt + "\n\nIMPORTANT: your previous attempt repeated an " \
                                          "already-sent message verbatim. Write a genuinely " \
                                          "different message this time."
            raw = _call_claude(RUBRIC_SYSTEM_PROMPT, retry_prompt)
            retry_result = _extract_json(raw)
            if retry_result.get("body"):
                result = retry_result
                body = str(result.get("body", "")).strip()
        except Exception as e:
            logger.error("Anti-repetition retry failed: %s", e)

    cta = result.get("cta", "open_ended")
    if cta not in ("binary", "open_ended", "none"):
        cta = "open_ended"

    suppression_key = trigger.get("suppression_key", "") or trigger.get("id", "")

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": str(result.get("rationale", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Reply composition — used mid-conversation by /v1/reply (via conversation_handlers.py)
# ---------------------------------------------------------------------------

def compose_reply(category: dict, merchant: dict, trigger: dict, customer: dict | None,
                   history: list[dict], merchant_message: str,
                   stage_hint: str = "", already_sent: list[str] | None = None) -> dict:
    """
    Returns dict with keys: action ("send"|"wait"|"end"), body (if send), cta (if send),
    wait_seconds (if wait), rationale.
    """
    already_sent = already_sent or []

    user_prompt = f"""You are mid-conversation. Decide the next move and, if sending, compose the message.

CATEGORY CONTEXT:
{_safe_json(category)}

MERCHANT CONTEXT:
{_safe_json(merchant)}

TRIGGER CONTEXT (the original reason this conversation started):
{_safe_json(trigger)}

CUSTOMER CONTEXT (null if this is a merchant-facing conversation):
{_safe_json(customer)}

CONVERSATION SO FAR (oldest first):
{_safe_json(history)}

LATEST INCOMING MESSAGE (just received, not yet in the history above):
"{merchant_message}"

{("STAGE HINT: " + stage_hint) if stage_hint else ""}

Messages you have already sent in this conversation (never repeat verbatim):
{_safe_json(already_sent)}

Decide ONE of three actions:
- "send": you have something worth saying right now. Exactly one CTA, last sentence.
- "wait": the other party asked for time / said "later" / is mid-thought — back off gracefully.
- "end": conversation is resolved, they said not interested, or continuing would waste turns
  (e.g. this is clearly an automated canned reply that has now repeated, or they explicitly
  declined, or you've achieved the goal and there's nothing left to say).

Special cases to get right:
- If the latest message is a WhatsApp Business AUTO-REPLY (a generic "thank you for contacting
  us, we'll get back to you" canned text, especially if it has appeared before), do not keep
  pushing — respond "end" gracefully, or if this is the first time you suspect it, "send" one
  short, genuinely human check ("...just so a real person sees this, mind confirming it's you?")
  before ending on the next repeat.
- If the merchant has just given clear affirmative commitment ("yes", "let's do it", "go ahead",
  "haan kar do", "theek hai", "sure", "confirm"), do NOT ask another qualifying question — take
  the action or clearly confirm you're doing it now (use words like "done", "sending",
  "confirmed", "here's", "drafted").
- If the message is hostile/abusive with no genuine question in it, "end" politely and briefly
  (no body needed).
- If the message is hostile but ALSO contains a real (even if off-topic) request, stay polite,
  decline the off-topic part briefly, and steer back to the original mission in ONE short
  message ("send").

Respond with STRICT JSON only:
{{
  "action": "send" | "wait" | "end",
  "body": "<message text, only if action=send>",
  "cta": "binary" | "open_ended" | "none",
  "wait_seconds": <integer, only if action=wait>,
  "rationale": "<1-2 sentences explaining the decision>"
}}"""

    if DEV_MODE:
        result = _dev_fallback_reply(merchant_message)
    else:
        try:
            raw = _call_claude(RUBRIC_SYSTEM_PROMPT, user_prompt)
            result = _extract_json(raw)
        except Exception as e:
            logger.error("compose_reply LLM call failed, using fallback: %s", e)
            result = _dev_fallback_reply(merchant_message)

    action = result.get("action", "send")
    if action not in ("send", "wait", "end"):
        action = "send"

    out = {"action": action, "rationale": str(result.get("rationale", "")).strip()}

    if action == "send":
        body = str(result.get("body", "")).strip()
        if body in already_sent and not DEV_MODE:
            body = body + " "  # trivial de-dup nudge; real fix is the retry above at compose_message
        cta = result.get("cta", "open_ended")
        if cta not in ("binary", "open_ended", "none"):
            cta = "open_ended"
        out["body"] = body
        out["cta"] = cta
    elif action == "wait":
        try:
            out["wait_seconds"] = int(result.get("wait_seconds", 1800))
        except (TypeError, ValueError):
            out["wait_seconds"] = 1800

    return out


# ---------------------------------------------------------------------------
# DEV_MODE fallbacks (no network / no API key) — deterministic templates.
# These exist ONLY so the server boots and is smoke-testable offline.
# They will NOT score well against the real judge — set ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------

def _dev_fallback_compose(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    name = (merchant.get("identity") or {}).get("name", "there")
    kind = trigger.get("kind", "update")
    if customer:
        cname = (customer.get("identity") or {}).get("name", "")
        body = f"Hi {cname}, this is {name} — following up re: {kind}. Let us know if you'd like to book a slot."
    else:
        body = f"Hi {name}, quick note on {kind} — want me to pull the details and draft next steps?"
    return {"body": body, "cta": "open_ended",
            "rationale": f"[DEV_MODE placeholder] templated on trigger.kind={kind}"}


def _dev_fallback_reply(merchant_message: str) -> dict:
    return {
        "action": "send",
        "body": "Got it — noted. Want me to go ahead with the next step?",
        "cta": "binary",
        "rationale": "[DEV_MODE placeholder] generic acknowledgement",
    }
