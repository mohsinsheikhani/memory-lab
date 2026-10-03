from datetime import date
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"

# The tags the LLM must pick from. "one": a new value closes the old one (one city at a time).
# "many": values pile up (he can like shawarma and mangoes). Anything else is "other".
PREDICATES = {
    "lives_in": "one",
    "diet": "one",
    "household_size": "one",
    "delivery_plan": "one",
    "delivery_time": "one",
    "allergy": "many",
    "likes_food": "many",
    "dislikes_food": "many",
}

# How fast each tag goes stale. A city changes over years, a delivery time over weeks.
SPEED = {
    "lives_in": "slow",
    "diet": "slow",
    "household_size": "slow",
    "allergy": "never",
    "delivery_plan": "medium",
    "delivery_time": "medium",
    "likes_food": "medium",
    "dislikes_food": "medium",
    "other": "medium",
}
STALE_AFTER_DAYS = {"never": None, "slow": 365, "medium": 60}


class CandidateFact(BaseModel):
    subject: str
    predicate: Literal[(*PREDICATES, "other")]
    value: str
    text: str


class ExtractionResult(BaseModel):
    facts: list[CandidateFact]


PROMPT = """You pull lasting facts about the user out of one chat message.

Rules:
- Use the user's name as the subject, never "I" or "the user".
- Turn relative dates ("last week", "yesterday") into real dates using today's date.
- One fact per item. predicate must be one of the listed tags. Use "other" if none fit.
  A move is lives_in. Vegetarian or vegan is diet. Starting, pausing or cancelling a delivery is delivery_plan.
- The fact must keep the meaning of the message. Never drop words that change it, like not, no longer, stopped, cancelled, paused, resumed, anymore.
- Only save what the user says outright. Never guess a fact from a passing remark.
- Only save things that will still be true next month. Skip weather, moods and passing comments.
- value is the current state, in a few words.
- text is one short sentence that makes sense on its own, like "Ali lives in Dubai (since 2026-03-10)."
- Small talk, greetings, thanks and questions with no facts give an empty list.
"""


def extract_facts(turn: str, user_name: str, turn_date: date) -> list[CandidateFact]:
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"User: {user_name}\nToday: {turn_date.isoformat()}\nMessage: {turn}"},
        ],
        response_format=ExtractionResult,
        temperature=0,
    )
    return ExtractionResult.model_validate_json(resp.choices[0].message.content).facts


if __name__ == "__main__":
    for turn in [
        "Hi! I moved to Dubai last week. Where's my order?",
        "thanks!",
        "I'm vegetarian and allergic to nuts",
        "My card is 4111 1111 1111 1111",
    ]:
        print(f"\n{turn}")
        for fact in extract_facts(turn, "Ali", date(2026, 3, 17)):
            print("  ", fact)
