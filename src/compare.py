from datetime import date
from typing import Literal

import litellm
import psycopg
from pydantic import BaseModel

from db import embed, save_fact
from extractor import MODEL, CandidateFact, extract_facts
from gate import gate, redact


class Decision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "IGNORE"]
    target_id: int | None
    reason: str


PROMPT = """You keep a user's memory clean. You get one new fact and a numbered list of facts already saved.
Pick one action:

- ADD: nothing saved covers this yet.
- UPDATE: the new fact replaces or changes a saved one (moved city, vegetarian -> vegan). target_id = that fact's number.
- DELETE: the new fact says a saved one is no longer true, and there's nothing new to save in its place. target_id = that fact's number.
- IGNORE: it's already saved, same meaning.

target_id must be a number from the list, or null for ADD and IGNORE.
"""


def find_related(conn: psycopg.Connection, fact: CandidateFact, vec: list[float],
                 tenant_id: str, user_id: str, k: int = 5) -> list[tuple]:
    """Current facts that might clash: same subject + predicate, or close in meaning."""
    return conn.execute(
        """
        (select id, text from memories
         where tenant_id = %s and user_id = %s and valid_to is null
           and subject = %s and predicate = %s)
        union
        (select id, text from memories
         where tenant_id = %s and user_id = %s and valid_to is null
         order by embedding <=> %s::vector
         limit %s)
        """,
        (tenant_id, user_id, fact.subject, fact.predicate, tenant_id, user_id, vec, k),
    ).fetchall()


def decide(fact: CandidateFact, related: list[tuple]) -> Decision:
    saved = "\n".join(f"{i}: {text}" for i, (_, text) in enumerate(related, 1)) or "(nothing saved yet)"
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Saved facts:\n{saved}\n\nNew fact: {fact.text}"},
        ],
        response_format=Decision,
        temperature=0,
    )
    return Decision.model_validate_json(resp.choices[0].message.content)


def apply(conn: psycopg.Connection, decision: Decision, fact: CandidateFact, vec: list[float],
          related: list[tuple], tenant_id: str, user_id: str, turn_date: date, derived_from: str) -> str:
    action = decision.action

    # Our code checks the LLM's answer before acting on it.
    target = None
    if action in ("UPDATE", "DELETE"):
        if decision.target_id is None or not 1 <= decision.target_id <= len(related):
            action = "ADD" if action == "UPDATE" else "IGNORE"
        else:
            target = related[decision.target_id - 1]

    if action == "IGNORE":
        return "IGNORE"

    if action == "DELETE":
        conn.execute("update memories set valid_to = %s where id = %s", (turn_date, target[0]))
        return f"DELETE  '{target[1]}'"

    new_id = save_fact(conn, fact, tenant_id, user_id, "user_said", turn_date, derived_from, embedding=vec)
    if action == "ADD":
        return "ADD"

    conn.execute("update memories set valid_to = %s, replaced_by = %s where id = %s", (turn_date, new_id, target[0]))
    return f"UPDATE  '{target[1]}' ->"


def remember(conn: psycopg.Connection, turn: str, user_name: str, tenant_id: str,
             user_id: str, turn_date: date, derived_from: str) -> None:
    """The full write step: redact -> extract -> gate -> compare -> save."""
    facts = gate(extract_facts(redact(turn), user_name, turn_date), user_name)
    for fact in facts:
        vec = embed(fact.text)
        related = find_related(conn, fact, vec, tenant_id, user_id)
        decision = decide(fact, related)
        result = apply(conn, decision, fact, vec, related, tenant_id, user_id, turn_date, derived_from)
        print(f"  {result} {fact.text}   ({decision.reason})")


if __name__ == "__main__":
    from db import connect, init_schema

    turns = [
        (date(2026, 1, 5), "jan-01", "Hi! I'm vegetarian and I live in Lahore."),
        (date(2026, 1, 9), "jan-02", "Please deliver after 6pm, thanks!"),
        (date(2026, 1, 20), "jan-03", "Just a reminder, I'm vegetarian."),
        (date(2026, 2, 2), "feb-01", "I have a nut allergy. Also I signed up for weekly delivery."),
        (date(2026, 3, 17), "mar-01", "I moved from Lahore to Dubai last week."),
        (date(2026, 4, 3), "apr-01", "Actually I'm vegan now, not just vegetarian."),
        (date(2026, 4, 20), "apr-02", "I cancelled my weekly delivery."),
    ]

    with connect() as conn:
        init_schema(conn)
        conn.execute("delete from memories where tenant_id = 'freshcart' and user_id = 'ali-demo'")
        for turn_date, conv_id, turn in turns:
            print(f"\n[{turn_date}] {turn}")
            remember(conn, turn, "Ali", "freshcart", "ali-demo", turn_date, conv_id)

        print("\nFinal memory (current facts):")
        for (text,) in conn.execute(
            "select text from memories where user_id = 'ali-demo' and valid_to is null order by created_at"
        ):
            print("  ", text)
        print("\nHistory (replaced or deleted):")
        for text, valid_to in conn.execute(
            "select text, valid_to from memories where user_id = 'ali-demo' and valid_to is not null order by created_at"
        ):
            print(f"   {text}  (until {valid_to})")
