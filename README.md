# memory-lab

A memory layer for a support agent on a grocery delivery platform. One customer, Ali, chats with the agent over six months, and the agent has to remember the right things about him.

## What we save

We save facts, not messages.

Ali writes: *"Hi! Just so you know, I moved to Dubai last week. Anyway, where's my order?"*

We save one short fact: **"Ali lives in Dubai (since 2026-03-10)."** "I" became "Ali", and "last week" became a real date, so the fact still makes sense six months later.

| Save | Skip | Never save |
| --- | --- | --- |
| Preferences: "I get home at 6pm, so deliver after that" | Small talk: "hello", "thanks", "ok" | Passwords, API keys |
| Facts about the user: "nut allergy", "lives in Dubai" | One-off requests: "deliver after 8pm today" | Card or bank numbers |
| Decisions: "chose weekly delivery" | Things you can look up again: today's order status | Personal data the task doesn't need |
| Corrections: "Actually I'm vegan, not vegetarian" | One-off moods: "I'm annoyed today" | |

Corrections matter most. They tell us the agent had something wrong.

"Please deliver after 6pm" on its own is a request for one order, not a preference. It only becomes a preference when there's a pattern behind it, like "I get home at 6pm".

## How a turn gets saved

```
turn -> redact -> extract -> gate -> compare -> save
```

### 1. Redact (`src/gate.py`)

Before anything goes to the LLM, regex swaps secrets for placeholders. The card number never leaves our server.

```
"My card is 4111 1111 1111 1111"  ->  "My card is [CARD_NUMBER]"
```

It catches API keys (`sk-`, `AKIA`, `ghp_`), IBANs, and card numbers (13 to 19 digits that pass the Luhn checksum, so a long order id isn't mistaken for a card).

### 2. Extract (`src/extractor.py`)

An LLM (`gpt-4o-mini` through LiteLLM) reads the turn and returns candidate facts as Pydantic objects:

| Field | Example |
| --- | --- |
| `subject` | Ali |
| `predicate` | lives_in |
| `value` | Dubai |
| `text` | Ali lives in Dubai (since 2026-03-10). |

We pass the user's name and today's date with every turn. That's how "I" becomes "Ali" and "last week" becomes a real date.

The prompt also says to keep words that change the meaning (not, cancelled, paused, anymore). Without that rule, "I cancelled my weekly delivery" came back as "Ali has a weekly delivery".

### 3. Gate (`src/gate.py`)

The LLM doesn't always follow its prompt, so code checks what it returned:

| Check | Example dropped |
| --- | --- |
| Secrets again, as a backup | a key the regex missed but the LLM cleaned up |
| Small talk | "Ali is annoyed today" |
| Not about the user | "Omar loves spicy food" (Ali's brother) |

A rule in the prompt is a suggestion. A check in code always runs.

### 4. Compare (`src/compare.py`)

For each fact, we look up what's already saved: rows with the **same subject and predicate**, plus the **5 closest by meaning**. An LLM picks one action:

| Action | When | Example |
| --- | --- | --- |
| ADD | Nothing similar saved | "Ali has a nut allergy" (first time) |
| UPDATE | Replaces an old fact | "Lives in Dubai" replaces "lives in Lahore" |
| DELETE | Says an old fact is no longer true | "I'm not vegetarian anymore" |
| IGNORE | Already saved | "I'm vegetarian" said again |

The LLM returns a decision (structured output, not tools), and our code runs it. We show the LLM short ids (1, 2, 3) instead of UUIDs, then map them back. Code checks the id is real before acting.

UPDATE never erases. The old row gets `valid_to` set and `replaced_by` pointing at the new row, so history stays.

### 5. Save (`src/db.py`)

Postgres with pgvector on Supabase. One table, `memories`:

| Column | Example |
| --- | --- |
| `tenant_id`, `user_id` | freshcart, ali |
| `subject`, `predicate`, `value` | Ali, lives_in, Dubai |
| `text` | Ali lives in Dubai |
| `source` | user_said |
| `valid_from`, `valid_to` | 2026-03-17, empty |
| `replaced_by` | id of the newer fact |
| `derived_from` | mar-01 |
| `embedding` | `text-embedding-3-small`, 1536 dims |

Row-level security is on with no policies. Supabase exposes tables through a public API, and this closes that door. Our backend connects as `postgres`, so it isn't affected.

## Evals (Module 0)

Before trusting any of this, we measure it.

**Data:**
- `evals/ali_history.yaml`: 15 chats with Ali, Jan to Jun 2026. Real facts mixed with small talk, one-off requests, a card number and his brother's tastes.
- `evals/cases.yaml`: 42 questions, 7 for each type.

| Type | Checks | Example |
| --- | --- | --- |
| recall | Finds one fact | "What's my cat's name?" -> Biscuit |
| update | Uses the newest fact | "Which city do I live in?" -> Dubai, not Lahore |
| time | Reasons about when | "Where did I live in January?" -> Lahore |
| across_chats | Combines facts | "Suggest a snack" -> vegan and nut-free |
| abstain | Admits it was never told | "What's my dog's name?" -> "you haven't told me" |
| old_premise | Catches an outdated question | "Send it to my Lahore address" -> "you moved to Dubai" |

**Grading:** code checks first (`must_contain`, `must_not_contain`). An LLM judge (`gpt-4o`) only for cases code can't check, like "is this snack vegan and nut-free?".

**Systems compared** (`src/run_evals.py`):

| System | How it answers |
| --- | --- |
| no_memory | Sees only the question |
| full_history | Every past message pasted into the prompt |
| simple_search | Every raw message saved with a vector, top 5 pulled in |
| memory | Our pipeline above, top 5 facts pulled in |

Every run goes to Langfuse (one trace per case, with a pass/fail score) and is saved to `evals/results/<system>.json`.

**First results:**

| Type | no_memory | full_history | simple_search | memory |
| --- | --- | --- | --- | --- |
| recall | 0% | 100% | 100% | 86% |
| update | 0% | 86% | 71% | 57% |
| time | 0% | 86% | 86% | 0% |
| across_chats | 0% | 57% | 14% | 29% |
| abstain | 100% | 86% | 100% | 86% |
| old_premise | 0% | 86% | 86% | 57% |
| **overall** | 17% | 83% | 76% | 52% |
| tokens / question | 114 | 906 | 220 | 163 |
| p95 latency | 1.7s | 1.5s | 2.5s | 3.9s |

Our memory system uses the fewest tokens of the three systems that have any memory, but it's behind simple search on accuracy for now.

## Run it

```bash
uv sync
```

`.env` needs:

```
OPENAI_API_KEY=...
DATABASE_HOST=aws-0-<region>.pooler.supabase.com
DATABASE_USER=postgres.<project-ref>
DATABASE_PASSWORD=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...
```

Use Supabase's pooler host. The direct host is IPv6 only.

```bash
uv run python src/extractor.py                          # try the extractor
uv run python src/gate.py                               # try redact + gate
uv run python src/compare.py                            # run 7 Ali turns through the full pipeline
uv run python src/run_evals.py --system full_history    # run the evals
uv run python src/run_evals.py --system memory --fresh  # wipe stored eval data first
```

Systems: `no_memory`, `full_history`, `simple_search`, `memory`. Use `--only recall` to run one type.
