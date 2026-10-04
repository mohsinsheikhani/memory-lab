"""The background checker. It reads each new fact next to the old ones and suggests closing
any old fact the new one makes wrong. It only suggests. approve() or reject() decides."""
from datetime import date

import litellm
import psycopg
from pydantic import BaseModel

from db import replace, scoped
from extractor import MODEL


class Proposal(BaseModel):
    target_id: int
    reason: str


class CheckResult(BaseModel):
    close: list[Proposal]


PROMPT = """You keep a user's memory correct. You get one new fact and a numbered list of older saved facts.
The new fact may be about a different topic, but still make an older fact wrong or useless.
Example: "Ali's cat passed away" makes "Add cat food to every order" wrong.

List only the older facts the new fact clearly ends, with a short reason for each.
A short break (a day off, a trip, a few days away) does not end a standing fact.
If nothing is ended, return an empty list. target_id must be a number from the list.
"""


def new_facts(conn: psycopg.Connection, derived_from: str) -> list[tuple]:
    """Current facts that came from one conversation."""
    with scoped(conn) as (tenant_id, user_id):
        return conn.execute(
            "select id, text, embedding from memories"
            " where tenant_id = %s and user_id = %s and derived_from = %s and valid_to is null",
            (tenant_id, user_id, derived_from),
        ).fetchall()


def older_facts(conn: psycopg.Connection, fact_id, vec, derived_from: str, k: int = 10) -> list[tuple]:
    """Current facts from other conversations, closest in meaning first."""
    with scoped(conn) as (tenant_id, user_id):
        return conn.execute(
            """
            select id, text from memories
            where tenant_id = %s and user_id = %s and valid_to is null
              and id <> %s and derived_from is distinct from %s
            order by embedding <=> %s::vector
            limit %s
            """,
            (tenant_id, user_id, fact_id, derived_from, vec, k),
        ).fetchall()


def ask(new_text: str, older: list[tuple]) -> list[Proposal]:
    saved = "\n".join(f"{i}: {text}" for i, (_, text) in enumerate(older, 1))
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Older facts:\n{saved}\n\nNew fact: {new_text}"},
        ],
        response_format=CheckResult,
        temperature=0,
    )
    return CheckResult.model_validate_json(resp.choices[0].message.content).close


def check(conn: psycopg.Connection, derived_from: str) -> list[tuple[str, str, str]]:
    """Look at every fact from one conversation and log proposals. Returns (old text, new text, reason)."""
    logged = []
    for new_id, new_text, vec in new_facts(conn, derived_from):
        older = older_facts(conn, new_id, vec, derived_from)
        if not older:
            continue
        for p in ask(new_text, older):
            if not 1 <= p.target_id <= len(older):
                continue
            old_id, old_text = older[p.target_id - 1]
            with scoped(conn) as (tenant_id, user_id):
                conn.execute(
                    "insert into proposals (tenant_id, user_id, old_id, new_id, reason) values (%s, %s, %s, %s, %s)",
                    (tenant_id, user_id, old_id, new_id, p.reason),
                )
            logged.append((old_text, new_text, p.reason))
    return logged


def pending(conn: psycopg.Connection) -> list[tuple]:
    with scoped(conn) as (tenant_id, user_id):
        return conn.execute(
            """
            select p.id, o.text, n.text, p.reason from proposals p
            join memories o on o.id = p.old_id
            join memories n on n.id = p.new_id
            where p.tenant_id = %s and p.user_id = %s and p.status = 'pending'
            order by p.created_at
            """,
            (tenant_id, user_id),
        ).fetchall()


def approve(conn: psycopg.Connection, proposal_id) -> None:
    """Close the old fact, dated from when the new one became true."""
    with scoped(conn) as (tenant_id, user_id):
        old_id, new_id, on = conn.execute(
            """
            update proposals p set status = 'approved' from memories n
            where p.id = %s and p.tenant_id = %s and p.user_id = %s and n.id = p.new_id
            returning p.old_id, p.new_id, n.valid_from
            """,
            (proposal_id, tenant_id, user_id),
        ).fetchone()
    replace(conn, old_id, new_id, on or date.today())


def reject(conn: psycopg.Connection, proposal_id) -> None:
    with scoped(conn) as (tenant_id, user_id):
        conn.execute("update proposals set status = 'rejected' where id = %s and tenant_id = %s and user_id = %s",
                     (proposal_id, tenant_id, user_id))
