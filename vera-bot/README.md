# Vera Challenge Bot — magicpin AI Challenge submission

## Approach

A single Claude-backed composer (`llm_composer.py`), called at `temperature=0` for
determinism, driven by one shared system prompt that encodes all five judged
dimensions (specificity, category fit, merchant fit, trigger relevance, engagement
compulsion) plus the brief's hard rules (no fabrication, one CTA, no long preambles,
etc.) directly as instructions — rather than five separate prompts or a fine-tuned
classifier per dimension.

A thin deterministic layer (`conversation_handlers.py`) sits in front of the LLM for
the three behaviors the judge explicitly tests in isolation (`judge_simulator.py`:
`_auto_reply`, `_intent`, `_hostile`):

- **Auto-reply detection**: normalized-text repeat counting. 2nd verbatim repeat →
  one genuine human-check probe. 3rd+ repeat → `end` immediately, no further turns
  wasted (matches the brief's explicit hint: "same message verbatim 3+ times =
  auto-reply").
- **Hostile handling**: regex-based hostility detection. Pure hostility with no
  embedded ask → `end` immediately, no apology theatre. Hostility with a genuine
  (even off-topic) request folded in → one polite redirect back to the original
  mission, per Phase 4's "stay on-mission" requirement.
- **Intent transition**: commitment-phrase regex (English + Hindi: "yes", "let's do
  it", "haan", "theek hai", "kar do", etc.) fired only when the bot's last message was
  itself a qualifying question. When it fires, the LLM prompt is given an explicit
  instruction to switch to action-mode language ("done", "sending", "confirmed")
  instead of asking another qualifying question.

Everything else — the actual message composition, and the `send`/`wait`/`end`
decision in ordinary (non-auto-reply, non-hostile, non-just-committed) turns — goes
to the LLM, because that's genuinely a judgment call that benefits from full context
rather than a hand-rolled heuristic.

**Anti-repetition** is enforced two ways: every prompt is given the list of bodies
already sent in that conversation with an explicit "never repeat these" instruction,
and if the model repeats anyway, one automatic retry fires with a sharper instruction.

**Restraint**: `/v1/tick` won't open a second simultaneous merchant-facing thread with
a merchant that already has one in flight, and tracks `suppression_key` usage across
ticks so the same trigger never fires twice.

## Tradeoffs

- The heuristic pre-checks (auto-reply/hostile/intent) are regex-based, not
  LLM-classified, to keep them instant and free of an extra round-trip inside the 30s
  budget. This trades some recall (a very unusual phrasing might slip past) for speed
  and determinism.
- `/v1/tick` composes one LLM call per candidate trigger, sequentially. Fine for
  the challenge's rate limits (10 req/s, 20 actions/tick cap) but would need
  parallelization (`asyncio.gather`) for a higher-throughput production version.
- Customer-facing (`send_as: merchant_on_behalf`) and merchant-facing composition
  share one prompt/model rather than two specialized prompts — simpler to maintain
  and audit, at some cost to how sharply each voice could be tuned.

## What additional context would have helped most

The real 30-pair **canonical test set** (referenced in challenge-brief.md §6 but not
part of what was uploaded to this session) — without it, `generate_submission.py`
currently falls back to emitting one line per placeholder trigger rather than the
actual required 30 lines. See the big comment at the top of that script for exactly
what to drop in once you have it.

---

## Project layout

```
bot.py                     — FastAPI app, all 5 required endpoints + optional /v1/teardown
llm_composer.py             — Claude-backed compose_message() / compose_reply()
conversation_handlers.py    — optional deliverable: respond(state, merchant_message) state machine
generate_submission.py       — produces submission.jsonl
test_bot_local.py            — offline smoke test (DEV_MODE, no API key/network needed)
judge_simulator.py           — magicpin's own local judge (copied from your upload, unmodified)
requirements.txt
render.yaml / Procfile       — Render deploy config
dataset/                     — PLACEHOLDER data (see below) — replace with the real challenge dataset
  categories/*.json           (5 categories)
  merchants_seed.json          (10 merchants — real dataset has 50)
  customers_seed.json          (6 customers — real dataset has 200)
  triggers_seed.json           (15 triggers — real dataset has 100)
```

⚠️ **The `dataset/` folder here is placeholder data I generated to match the shapes
in the brief**, so the bot and test scripts are runnable end-to-end right now. Before
your actual submission, replace these 4 files/folder with the real ones from the
challenge package (same filenames/shapes, so no code changes needed) — I didn't
receive the real dataset in this session, only the two briefs, the design docs, and
`judge_simulator.py`.

---

## 1. Setup

```bash
cd vera-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...           # required for real composition
# optional overrides:
export ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
export TEAM_NAME="Your Team Name"
export CONTACT_EMAIL=you@example.com
```

Without `ANTHROPIC_API_KEY` set, the bot runs in **DEV_MODE**: all 5 endpoints work
and return well-shaped responses, but messages are generated by a dumb template, not
the real composer. This exists purely so you can smoke-test the plumbing without an
API key or network. **Do not submit while in DEV_MODE** — set the key before
deploying for real.

## 2. Run locally

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## 3. Offline smoke test (no API key needed)

```bash
python test_bot_local.py
```

This runs entirely in-process (FastAPI `TestClient`, no real HTTP, no network) using
DEV_MODE, and checks: context push + idempotency, `/v1/tick` produces actions with
all required keys, tick-level suppression dedup, `/v1/reply` auto-reply detection
(3x verbatim → `end`), hostile-message handling (→ `end`), intent-transition handling,
and `/v1/teardown`. Fix anything it flags before moving on.

## 4. Real test with the judge simulator

Once you're happy with the offline smoke test **and** have set `ANTHROPIC_API_KEY`:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080 &
```

Edit the `CONFIGURATION` section at the top of `judge_simulator.py` (this is how that
script is designed to be used — it has no CLI flags):

```python
BOT_URL = "http://localhost:8080"
LLM_PROVIDER = "anthropic"     # the JUDGE's own scoring LLM — can differ from your bot's
LLM_API_KEY = "sk-ant-..."     # can be the same key or a separate one
LLM_MODEL = ""                 # leave blank for default
TEST_SCENARIO = "all"          # or "full" to score against the whole placeholder dataset
```

Then:

```bash
python judge_simulator.py
```

Iterate on `llm_composer.py`'s `RUBRIC_SYSTEM_PROMPT` based on the scores/hints it
gives you.

## 5. Generate `submission.jsonl`

```bash
python generate_submission.py
```

Reads `dataset/`, calls the real composer (needs `ANTHROPIC_API_KEY` set) for every
test pair, writes `submission.jsonl` to the project root. **Replace the placeholder
dataset with the real one first** (see the warning above) — otherwise this produces
lines for the wrong (placeholder) merchants/triggers.

## 6. Deploy to Render

1. Push this folder to a git repo (GitHub/GitLab).
2. In Render: **New → Blueprint**, point it at the repo — it'll pick up `render.yaml`
   automatically. (Or **New → Web Service** manually, if you'd rather not use the
   blueprint: build command `pip install -r requirements.txt`, start command
   `uvicorn bot:app --host 0.0.0.0 --port $PORT`.)
3. In the Render dashboard, set the `ANTHROPIC_API_KEY` environment variable (it's
   deliberately left out of `render.yaml` / not committed to git — never commit API
   keys).
4. Once deployed, confirm:
   ```bash
   curl https://<your-app>.onrender.com/v1/healthz
   curl https://<your-app>.onrender.com/v1/metadata
   ```
5. **Free-tier note**: Render's free web services spin down after ~15 min idle and
   take ~30-60s to wake on the next request. The judge's warmup phase (§4 Phase 1 of
   the testing brief) should wake it, but if your test window has long gaps, consider
   a paid "always-on" instance, or a cheap external uptime-pinger, so `/v1/healthz`
   doesn't fail 3x-in-a-row and get you disqualified for that slot.

## 7. Pre-flight checklist (from challenge-testing-brief.md §12)

- [ ] Endpoint reachable from the public internet (Render URL) — verify after step 6
- [ ] All 5 endpoints implemented — done (`bot.py`)
- [ ] `/v1/context` idempotent on `(scope, context_id, version)` — done, tested in
      `test_bot_local.py`
- [ ] `/v1/tick` returns within 30s even with nothing to send — done (`{"actions": []}`)
- [ ] `/v1/reply` returns within 30s — done (single LLM call per turn)
- [ ] Bot persists context across calls, no restarts — done (in-memory, single process)
- [ ] `judge_simulator.py` passes locally with non-zero scores — **run it yourself
      with a real API key (step 4)**; I couldn't run it in my sandbox (no network
      access, no `anthropic`/`fastapi` packages available to install)
- [ ] Replace placeholder `dataset/` with the real challenge dataset
- [ ] Regenerate `submission.jsonl` against the real dataset + real 30-pair test set
- [ ] Set `ANTHROPIC_API_KEY` on Render (not DEV_MODE) before the actual test window
- [ ] Submit your Render URL via the submission portal

## A note on what I could and couldn't verify myself

My sandbox has no network access, so I could not: install `fastapi`/`anthropic` to
actually run `uvicorn` or `test_bot_local.py`, call the real Anthropic API, or hit a
deployed URL. I did verify every `.py` file compiles cleanly (`py_compile`) and every
dataset `.json` file parses. Please run `python test_bot_local.py` yourself as the
first real check — if anything fails, share the output and I'll fix it.
