"""Run the eval cases against one system and print accuracy per type, tokens and p95 latency.

    uv run python src/run_evals.py --system full_history
    uv run python src/run_evals.py --system memory --fresh   # wipe and re-ingest stored memory first
"""
import argparse
import hashlib
import json
import statistics
import time
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

import litellm
import psycopg
import yaml
from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from pydantic import BaseModel

from auth import login, make_token
from compare import remember
from db import connect, embed, init_schema, search
from extractor import MODEL

load_dotenv()
langfuse = Langfuse()

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
TENANT = "eval"
USER_NAME = "Ali"
DEFAULT_ASKED_ON = date(2026, 6, 15)
JUDGE_MODEL = "gpt-4o"
TOP_K = 5

SYSTEMS = ["no_memory", "full_history", "simple_search", "memory"]

RAW_SCHEMA = """
create table if not exists raw_messages (
    id         uuid primary key default gen_random_uuid(),
    tenant_id  text not null,
    user_id    text not null,
    conv_id    text not null,
    sent_on    date not null,
    role       text not null,
    text       text not null,
    embedding  vector(1536) not null
);
create index if not exists raw_messages_owner_idx on raw_messages (tenant_id, user_id);
alter table raw_messages enable row level security;
"""

ANSWER_PROMPT = """You are the support agent for FreshCart, a grocery delivery app. The customer is {user}.
Today is {today}.

{context}

Answer in 1 to 3 short sentences. Only use what you know about {user} from above.
If you were never told something about {user}, say so. Don't guess."""


# ---------- loading ----------

def load_history() -> dict[str, dict]:
    return {c["id"]: c for c in yaml.safe_load((EVALS / "ali_history.yaml").read_text())}


def load_cases(history: dict) -> list[dict]:
    cases = yaml.safe_load((EVALS / "cases.yaml").read_text())
    for c in cases:
        c["history"] = c.get("history") or list(history)
        c["asked_on"] = c.get("asked_on") or DEFAULT_ASKED_ON
    return cases


def history_key(conv_ids: list[str]) -> str:
    """Each distinct history gets its own user_id, so stored memory never leaks between cases."""
    return "ali-" + hashlib.sha1(",".join(sorted(conv_ids)).encode()).hexdigest()[:8]


def seen(history: dict, conv_ids: list[str]) -> list[dict]:
    return sorted((history[i] for i in conv_ids), key=lambda c: c["date"])


# ---------- ingest (only for systems that store things) ----------

def ingest(conn: psycopg.Connection, system: str, history: dict, cases: list[dict], fresh: bool) -> None:
    table = {"simple_search": "raw_messages", "memory": "memories"}.get(system)
    if not table:
        return
    if fresh:
        conn.execute(f"delete from {table} where tenant_id = %s", (TENANT,))

    for conv_ids in {tuple(sorted(c["history"])) for c in cases}:
        user_id = history_key(list(conv_ids))
        if conn.execute(f"select 1 from {table} where tenant_id = %s and user_id = %s limit 1",
                        (TENANT, user_id)).fetchone():
            continue
        print(f"ingesting {len(conv_ids)} conversations into {table} as {user_id}...")
        for conv in seen(history, list(conv_ids)):
            for msg in conv["messages"]:
                if system == "simple_search":
                    line = f"[{conv['date']}] {msg['role']}: {msg['text']}"
                    conn.execute(
                        "insert into raw_messages (tenant_id, user_id, conv_id, sent_on, role, text, embedding)"
                        " values (%s, %s, %s, %s, %s, %s, %s::vector)",
                        (TENANT, user_id, conv["id"], conv["date"], msg["role"], line, embed(line)),
                    )
                elif msg["role"] == "user":
                    with login(make_token(TENANT, user_id)):
                        remember(conn, msg["text"], USER_NAME, conv["date"], conv["id"])


# ---------- the four systems: build the context for one question ----------

def build_context(conn: psycopg.Connection, system: str, case: dict, history: dict) -> str:
    if system == "no_memory":
        return "You have no memory of past conversations."

    if system == "full_history":
        lines = [f"[{conv['date']}] {m['role']}: {m['text']}"
                 for conv in seen(history, case["history"]) for m in conv["messages"]]
        return "Past conversations:\n" + "\n".join(lines)

    user_id = history_key(case["history"])
    if system == "simple_search":
        rows = conn.execute(
            "select text from raw_messages where tenant_id = %s and user_id = %s"
            " order by embedding <=> %s::vector limit %s",
            (TENANT, user_id, embed(case["question"]), TOP_K),
        ).fetchall()
        return "Relevant past messages:\n" + "\n".join(r[0] for r in rows)

    with login(make_token(TENANT, user_id)):
        facts = search(conn, case["question"], k=TOP_K)
    return "What you know about the customer:\n" + "\n".join(f"- {text}" for text, _ in facts)


def answer(conn, system: str, case: dict, history: dict) -> tuple[str, int, float]:
    start = time.perf_counter()
    context = build_context(conn, system, case, history)
    messages = [
        {"role": "system", "content": ANSWER_PROMPT.format(user=USER_NAME, today=case["asked_on"], context=context)},
        {"role": "user", "content": case["question"]},
    ]
    with langfuse.start_as_current_observation(name="answer", as_type="generation", model=MODEL, input=messages) as gen:
        resp = litellm.completion(model=MODEL, messages=messages, temperature=0)
        reply = resp.choices[0].message.content
        gen.update(output=reply, usage_details={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens})
    latency = time.perf_counter() - start
    return reply, resp.usage.total_tokens, latency


# ---------- grading: code checks first, judge only when the case asks for one ----------

class Verdict(BaseModel):
    passed: bool
    reason: str


def judge(question: str, rule: str, reply: str) -> Verdict:
    resp = litellm.completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You grade a support agent's answer against one rule. Be strict: pass only if the rule is clearly met."},
            {"role": "user", "content": f"Question: {question}\nRule: {rule}\nAnswer: {reply}"},
        ],
        response_format=Verdict,
        temperature=0,
    )
    return Verdict.model_validate_json(resp.choices[0].message.content)


def grade(case: dict, reply: str) -> tuple[bool, list[str]]:
    low = reply.lower()
    fails = [f"missing '{w}'" for w in case.get("must_contain", []) if w.lower() not in low]
    fails += [f"contains '{w}'" for w in case.get("must_not_contain", []) if w.lower() in low]
    if case.get("judge"):
        verdict = judge(case["question"], case["judge"], reply)
        if not verdict.passed:
            fails.append(f"judge: {verdict.reason}")
    return not fails, fails


# ---------- run ----------

def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[-1] if len(values) > 1 else values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--fresh", action="store_true", help="wipe stored eval data and ingest again")
    parser.add_argument("--only", help="run only cases whose id starts with this")
    args = parser.parse_args()

    history = load_history()
    cases = [c for c in load_cases(history) if not args.only or c["id"].startswith(args.only)]
    run_id = f"{args.system}-{uuid.uuid4().hex[:6]}"

    with connect() as conn:
        init_schema(conn)
        conn.execute(RAW_SCHEMA)
        ingest(conn, args.system, history, cases, args.fresh)

        results = []
        for case in cases:
            tags = [args.system, case["type"], case["id"]]
            with propagate_attributes(session_id=run_id, trace_name=f"eval-{args.system}", tags=tags):
                with langfuse.start_as_current_observation(name=case["id"], input=case["question"]) as span:
                    reply, tokens, latency = answer(conn, args.system, case, history)
                    passed, fails = grade(case, reply)
                    span.update(output=reply)
                    span.score_trace(name="passed", value=float(passed), comment="; ".join(fails) or None)
            results.append({"id": case["id"], "type": case["type"], "question": case["question"],
                            "answer": reply, "passed": passed, "fails": fails,
                            "tokens": tokens, "latency_s": round(latency, 3)})
            print(f"{'PASS' if passed else 'FAIL'}  {case['id']:<12} {'; '.join(fails)}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["passed"])

    print(f"\nSystem: {args.system}   (run {run_id})")
    print(f"{'type':<14}{'accuracy':>10}{'cases':>8}")
    for t, passes in by_type.items():
        print(f"{t:<14}{sum(passes) / len(passes):>9.0%}{len(passes):>8}")
    total = [r["passed"] for r in results]
    tokens = [r["tokens"] for r in results]
    latencies = [r["latency_s"] for r in results]
    print(f"{'overall':<14}{sum(total) / len(total):>9.0%}{len(total):>8}")
    print(f"\navg tokens per question: {statistics.mean(tokens):,.0f}")
    print(f"p95 latency: {p95(latencies):.2f}s")

    out = EVALS / "results" / f"{args.system}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "system": args.system, "run_id": run_id,
        "accuracy": {t: sum(p) / len(p) for t, p in by_type.items()} | {"overall": sum(total) / len(total)},
        "avg_tokens": statistics.mean(tokens), "p95_latency_s": p95(latencies),
        "cases": results,
    }, indent=2, default=str))
    print(f"saved {out.relative_to(ROOT)}")
    langfuse.flush()


if __name__ == "__main__":
    main()
