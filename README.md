# memory-lab

A memory layer for a support agent on a grocery delivery platform. One customer, Ali, chats with the agent over six months, and the agent has to remember the right things about him.

## What we save

We save facts, not messages.

Ali writes: *"Hi! Just so you know, I moved to Dubai last week. Anyway, where's my order?"*

We save one short fact: **"Ali lives in Dubai (since 2026-03-10)."** "I" became "Ali", and "last week" became a real date, so the fact still makes sense six months later.

| Save | Skip | Never save |
| --- | --- | --- |
| Preferences: "deliver after 6pm" | Small talk: "hello", "thanks", "ok" | Passwords, API keys |
| Facts about the user: "nut allergy", "lives in Dubai" | Things you can look up again: today's order status | Card or bank numbers |
| Decisions: "chose weekly delivery" | One-off moods: "I'm annoyed today" | Personal data the task doesn't need |
| Corrections: "Actually I'm vegan, not vegetarian" | | |

Corrections matter most. They tell us the agent had something wrong.

## How it works so far

### Step 1: Extract

A turn comes in, and an LLM (`gpt-4o-mini` through LiteLLM) pulls out a list of candidate facts. Each one comes back as a Pydantic object:

| Field | Example |
| --- | --- |
| `subject` | Ali |
| `predicate` | lives_in |
| `value` | Dubai |
| `text` | Ali lives in Dubai (since 2026-03-10). |

We pass the user's name and today's date with every turn. That's how the model turns "I" into "Ali" and "last week" into a real date.

| Turn | Facts |
| --- | --- |
| "I moved to Dubai last week. Where's my order?" | `Ali lives_in Dubai` |
| "thanks!" | none |
| "I'm vegetarian and allergic to nuts" | `Ali diet vegetarian`, `Ali allergy nuts` |

Code: `src/extractor.py`

## Run it

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." >> .env
uv run python src/extractor.py
```
