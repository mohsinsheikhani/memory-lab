from datetime import date

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "gpt-4o-mini"


class CandidateFact(BaseModel):
    subject: str
    predicate: str
    value: str
    text: str


class ExtractionResult(BaseModel):
    facts: list[CandidateFact]


PROMPT = """You pull lasting facts about the user out of one chat message.

Rules:
- Use the user's name as the subject, never "I" or "the user".
- Turn relative dates ("last week", "yesterday") into real dates using today's date.
- One fact per item. Predicates are short snake_case, like lives_in, diet, allergy, delivery_time.
- value is just the fact's value, like "Dubai" or "nuts".
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
