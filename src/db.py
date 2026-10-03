import os
from contextlib import contextmanager
from datetime import date

import litellm
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from auth import current_scope
from extractor import CandidateFact

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

SCHEMA = f"""
create extension if not exists vector;

create table if not exists memories (
    id           uuid primary key default gen_random_uuid(),
    tenant_id    text not null,
    user_id      text not null,
    subject      text not null,
    predicate    text not null,
    value        text not null,
    text         text not null,
    source       text not null,
    valid_from   date,
    valid_to     date,
    derived_from text,
    embedding    vector({EMBED_DIMS}) not null,
    created_at   timestamptz not null default now()
);

-- When a fact is replaced, the old row points to the new one.
alter table memories add column if not exists replaced_by uuid references memories(id);

create index if not exists memories_owner_idx on memories (tenant_id, user_id, subject, predicate);
create index if not exists memories_embedding_idx on memories using hnsw (embedding vector_cosine_ops);

-- Supabase exposes public tables through its API. RLS with no policies blocks that path.
alter table memories enable row level security;

-- Our login role (postgres) has BYPASSRLS, so RLS never applies to it, even with FORCE.
-- Requests run as memory_app instead, which can't bypass it.
do $$ begin
    if not exists (select from pg_roles where rolname = 'memory_app') then
        create role memory_app nologin;
    end if;
end $$;
grant memory_app to current_user;
grant usage on schema public, extensions to memory_app;
grant select, insert, update, delete on memories to memory_app;

-- Only the logged-in tenant + user's rows. No settings means no rows.
drop policy if exists owner_only on memories;
create policy owner_only on memories to memory_app
    using (tenant_id = current_setting('app.tenant_id', true) and user_id = current_setting('app.user_id', true))
    with check (tenant_id = current_setting('app.tenant_id', true) and user_id = current_setting('app.user_id', true));
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ["DATABASE_HOST"],
        port=5432,
        dbname="postgres",
        user=os.environ["DATABASE_USER"],
        password=os.environ["DATABASE_PASSWORD"],
        autocommit=True,
    )
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


@contextmanager
def scoped(conn: psycopg.Connection):
    """Run as the logged-in user. Postgres hides every row that isn't theirs."""
    tenant_id, user_id = current_scope()
    with conn.transaction():
        conn.execute("set local role memory_app")
        conn.execute("select set_config('app.tenant_id', %s, true), set_config('app.user_id', %s, true)",
                     (tenant_id, user_id))
        yield tenant_id, user_id


def embed(text: str) -> list[float]:
    return litellm.embedding(model=EMBED_MODEL, input=[text]).data[0]["embedding"]


def save_fact(
    conn: psycopg.Connection,
    fact: CandidateFact,
    source: str,
    valid_from: date,
    derived_from: str | None = None,
    embedding: list[float] | None = None,
) -> str:
    vec = embedding or embed(fact.text)
    with scoped(conn) as (tenant_id, user_id):
        row = conn.execute(
            """
            insert into memories (tenant_id, user_id, subject, predicate, value, text, source, valid_from, derived_from, embedding)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            returning id
            """,
            (tenant_id, user_id, fact.subject, fact.predicate, fact.value, fact.text,
             source, valid_from, derived_from, vec),
        ).fetchone()
    return str(row[0])


def search(conn: psycopg.Connection, query: str, k: int = 5) -> list[tuple[str, float]]:
    """Current facts for the logged-in user, closest first."""
    vec = embed(query)
    with scoped(conn) as (tenant_id, user_id):
        return conn.execute(
            """
            select text, 1 - (embedding <=> %s::vector) as similarity
            from memories
            where tenant_id = %s and user_id = %s and valid_to is null
            order by embedding <=> %s::vector
            limit %s
            """,
            (vec, tenant_id, user_id, vec, k),
        ).fetchall()


if __name__ == "__main__":
    from auth import login, make_token
    from extractor import extract_facts
    from gate import gate, redact

    turn_date = date(2026, 2, 3)
    with connect() as conn, login(make_token("freshcart", "ali")):
        init_schema(conn)
        for turn in ["I'm vegetarian and I have a nut allergy", "Please deliver after 6pm, thanks!"]:
            for fact in gate(extract_facts(redact(turn), "Ali", turn_date), "Ali"):
                save_fact(conn, fact, "user_said", turn_date, derived_from="demo")
                print("saved:", fact.text)

        print("\nsearch: 'suggest a snack'")
        for text, sim in search(conn, "suggest a snack"):
            print(f"  {sim:.2f}  {text}")
