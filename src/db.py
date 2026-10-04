import os
from contextlib import contextmanager
from datetime import date

import litellm
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from auth import current_scope
from extractor import SPEED, STALE_AFTER_DAYS, CandidateFact

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

-- The last day the user told us this. Saying it again moves it forward.
alter table memories add column if not exists last_confirmed date;

-- Words of each fact, so keyword search can find exact names like "Sara" or "halloumi".
alter table memories add column if not exists words tsvector
    generated always as (to_tsvector('english', text)) stored;

create index if not exists memories_owner_idx on memories (tenant_id, user_id, subject, predicate);
create index if not exists memories_embedding_idx on memories using hnsw (embedding vector_cosine_ops);
create index if not exists memories_words_idx on memories using gin (words);

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

-- The checker's suggestions to close an old fact. Nothing closes until one is approved.
create table if not exists proposals (
    id         uuid primary key default gen_random_uuid(),
    tenant_id  text not null,
    user_id    text not null,
    old_id     uuid not null references memories(id) on delete cascade,
    new_id     uuid not null references memories(id) on delete cascade,
    reason     text not null,
    status     text not null default 'pending',
    created_at timestamptz not null default now()
);
alter table proposals enable row level security;
grant select, insert, update, delete on proposals to memory_app;
drop policy if exists owner_only on proposals;
create policy owner_only on proposals to memory_app
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
            insert into memories (tenant_id, user_id, subject, predicate, value, text, source, valid_from, last_confirmed, derived_from, embedding)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            returning id
            """,
            (tenant_id, user_id, fact.subject, fact.predicate, fact.value, fact.text,
             source, valid_from, valid_from, derived_from, vec),
        ).fetchone()
    return str(row[0])


def replace(conn: psycopg.Connection, old_id: str, new_id: str, on: date) -> None:
    """Close the old fact instead of deleting it. It stays in history and points to the new one."""
    with scoped(conn) as (tenant_id, user_id):
        conn.execute(
            """
            update memories set valid_to = %s, replaced_by = %s
            where id = %s and tenant_id = %s and user_id = %s and valid_to is null
            """,
            (on, new_id, old_id, tenant_id, user_id),
        )


def confirm(conn: psycopg.Connection, fact_id: str, on: date) -> None:
    """The user said it again, so it's fresh again."""
    with scoped(conn) as (tenant_id, user_id):
        conn.execute(
            "update memories set last_confirmed = %s where id = %s and tenant_id = %s and user_id = %s",
            (on, fact_id, tenant_id, user_id),
        )


def with_age(text: str, predicate: str, last_confirmed: date | None, today: date) -> str:
    """Tag a fact that's old for its type, so the agent knows to double check it."""
    days = STALE_AFTER_DAYS[SPEED.get(predicate, "medium")]
    if days is None or last_confirmed is None or (today - last_confirmed).days < days:
        return text
    return f"{text} (last confirmed on {last_confirmed})"


def facts_as_of(conn: psycopg.Connection, day: date) -> list[str]:
    """What was true for the logged-in user on that day."""
    with scoped(conn) as (tenant_id, user_id):
        return [r[0] for r in conn.execute(
            """
            select text from memories
            where tenant_id = %s and user_id = %s
              and (valid_from is null or valid_from <= %s)
              and (valid_to is null or valid_to > %s)
            order by valid_from
            """,
            (tenant_id, user_id, day, day),
        )]


def search(conn: psycopg.Connection, query: str, k: int = 5, today: date | None = None) -> list[tuple[str, float]]:
    """Current facts for the logged-in user, closest first. Old ones get a "last confirmed" tag."""
    vec = embed(query)
    today = today or date.today()
    with scoped(conn) as (tenant_id, user_id):
        rows = conn.execute(
            """
            select text, predicate, last_confirmed, 1 - (embedding <=> %s::vector) as similarity
            from memories
            where tenant_id = %s and user_id = %s and valid_to is null
            order by embedding <=> %s::vector
            limit %s
            """,
            (vec, tenant_id, user_id, vec, k),
        ).fetchall()
    return [(with_age(text, predicate, confirmed, today), sim) for text, predicate, confirmed, sim in rows]


def keyword_search(conn: psycopg.Connection, query: str, k: int = 5, today: date | None = None) -> list[tuple[str, float]]:
    """Current facts that share a word with the question. A fact with any of the words counts, more matches rank higher."""
    today = today or date.today()
    with scoped(conn) as (tenant_id, user_id):
        rows = conn.execute(
            """
            with q as (
                select to_tsquery('english', array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | ')) as words
            )
            select text, predicate, last_confirmed, ts_rank(memories.words, q.words) as rank
            from memories, q
            where tenant_id = %s and user_id = %s and valid_to is null
              and memories.words @@ q.words
            order by rank desc
            limit %s
            """,
            (query, tenant_id, user_id, k),
        ).fetchall()
    return [(with_age(text, predicate, confirmed, today), rank) for text, predicate, confirmed, rank in rows]


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
