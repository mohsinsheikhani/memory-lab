from datetime import date
from typing import Literal

import litellm
import psycopg
from pydantic import BaseModel

from db import confirm, embed, facts_as_of, replace, save_fact, scoped, search
from extractor import MODEL, PREDICATES, CandidateFact, extract_facts
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


def find_related(conn: psycopg.Connection, fact: CandidateFact, vec: list[float], k: int = 5) -> list[tuple]:
    """Current facts that might clash: same subject + predicate, or close in meaning."""
    with scoped(conn) as (tenant_id, user_id):
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
          related: list[tuple], turn_date: date, derived_from: str) -> str:
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
        with scoped(conn) as (tenant_id, user_id):
            conn.execute("update memories set valid_to = %s where id = %s and tenant_id = %s and user_id = %s",
                         (turn_date, target[0], tenant_id, user_id))
        return f"DELETE  '{target[1]}'"

    new_id = save_fact(conn, fact, "user_said", turn_date, derived_from, embedding=vec)
    if action == "ADD":
        return "ADD"

    replace(conn, target[0], new_id, turn_date)
    return f"UPDATE  '{target[1]}' ->"


def by_rule(conn: psycopg.Connection, fact: CandidateFact, vec: list[float], turn_date: date, derived_from: str) -> str | None:
    """Known tags need no LLM. "one": the new value closes the old one. "many": keep them all."""
    kind = PREDICATES.get(fact.predicate)
    if kind is None:
        return None

    with scoped(conn) as (tenant_id, user_id):
        current = conn.execute(
            """
            select id, text, value from memories
            where tenant_id = %s and user_id = %s and valid_to is null and subject = %s and predicate = %s
            """,
            (tenant_id, user_id, fact.subject, fact.predicate),
        ).fetchall()

    for old_id, _, value in current:
        if value.strip().lower() == fact.value.strip().lower():
            confirm(conn, old_id, turn_date)
            return "CONFIRM"

    new_id = save_fact(conn, fact, "user_said", turn_date, derived_from, embedding=vec)
    if kind == "one" and current:
        for old_id, old_text, _ in current:
            replace(conn, old_id, new_id, turn_date)
        return f"REPLACE  '{old_text}' ->"
    return "ADD"


def remember(conn: psycopg.Connection, turn: str, user_name: str, turn_date: date, derived_from: str) -> None:
    """The full write step: redact -> extract -> gate -> compare -> save."""
    facts = gate(extract_facts(redact(turn), user_name, turn_date), user_name)
    for fact in facts:
        vec = embed(fact.text)
        result = by_rule(conn, fact, vec, turn_date, derived_from)
        if result:
            print(f"  {result} {fact.text}   (rule: {fact.predicate} is {PREDICATES[fact.predicate]})")
            continue
        related = find_related(conn, fact, vec)
        decision = decide(fact, related)
        result = apply(conn, decision, fact, vec, related, turn_date, derived_from)
        print(f"  {result} {fact.text}   ({decision.reason})")


if __name__ == "__main__":
    from auth import login, make_token
    from db import connect, init_schema

    turns = [
        (date(2026, 1, 5), "jan-01", "Hi! I'm vegetarian and I live in Lahore."),
        (date(2026, 1, 9), "jan-02", "Please deliver after 6pm, thanks!"),
        (date(2026, 1, 20), "jan-03", "Just a reminder, I'm vegetarian."),
        (date(2026, 2, 2), "feb-01", "I have a nut allergy. Also I signed up for weekly delivery."),
        (date(2026, 3, 17), "mar-01", "I moved from Lahore to Dubai last week."),
        (date(2026, 3, 25), "mar-02", "I love shawarma."),
        (date(2026, 3, 28), "mar-03", "Mangoes are my favourite too."),
        (date(2026, 4, 3), "apr-01", "Actually I'm vegan now, not just vegetarian."),
        (date(2026, 4, 20), "apr-02", "I cancelled my weekly delivery."),
    ]

    with connect() as conn, login(make_token("freshcart", "ali-demo")):
        init_schema(conn)
        with scoped(conn) as (tenant_id, user_id):
            conn.execute("delete from memories where tenant_id = %s and user_id = %s", (tenant_id, user_id))
        for turn_date, conv_id, turn in turns:
            print(f"\n[{turn_date}] {turn}")
            remember(conn, turn, "Ali", turn_date, conv_id)

        with scoped(conn) as (tenant_id, user_id):
            print("\nFinal memory (current facts):")
            for (text,) in conn.execute(
                "select text from memories where tenant_id = %s and user_id = %s and valid_to is null order by created_at",
                (tenant_id, user_id),
            ):
                print("  ", text)
            print("\nHistory (replaced or deleted):")
            for text, valid_to in conn.execute(
                "select text, valid_to from memories where tenant_id = %s and user_id = %s and valid_to is not null order by created_at",
                (tenant_id, user_id),
            ):
                print(f"   {text}  (until {valid_to})")

        for day in (date(2026, 1, 15), date(2026, 3, 20), date(2026, 4, 25)):
            print(f"\nTrue on {day}:")
            for text in facts_as_of(conn, day):
                print("  ", text)

        for today in (date(2026, 4, 25), date(2027, 6, 1)):
            print(f"\nRetrieved on {today}:")
            for text, _ in search(conn, "where do I live, what do I eat, when do you deliver", k=10, today=today):
                print("  ", text)
