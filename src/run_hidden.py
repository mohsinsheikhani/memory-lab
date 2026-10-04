"""Run the 20 hidden-replacement cases. Each case is its own user: save the old message, then the new one,
then run the checker and see if it proposed closing the old fact.

    uv run python src/run_hidden.py
"""
import json
from pathlib import Path

import yaml

from auth import login, make_token
from checker import check
from compare import remember
from db import connect, init_schema, scoped

ROOT = Path(__file__).resolve().parent.parent
TENANT = "hidden"


def old_rows(conn, case_id: str) -> list[tuple]:
    with scoped(conn) as (tenant_id, user_id):
        return conn.execute(
            "select id, text, valid_to from memories where tenant_id = %s and user_id = %s and derived_from = %s",
            (tenant_id, user_id, f"{case_id}-old"),
        ).fetchall()


def main() -> None:
    cases = yaml.safe_load((ROOT / "evals" / "hidden_cases.yaml").read_text())
    results = []
    with connect() as conn:
        init_schema(conn)
        conn.execute("delete from memories where tenant_id = %s", (TENANT,))

        for case in cases:
            print(f"\n=== {case['id']}  (expect {case['expect']})")
            with login(make_token(TENANT, case["id"])):
                for side in ("old", "new"):
                    turn = case[side]
                    print(f"[{turn['date']}] {turn['says']}")
                    remember(conn, turn["says"], "Ali", turn["date"], f"{case['id']}-{side}")

                old = old_rows(conn, case["id"])
                closed_by_compare = any(valid_to for _, _, valid_to in old)
                proposals = check(conn, f"{case['id']}-new")
                old_texts = {text for _, text, _ in old}
                proposed = [p for p in proposals if p[0] in old_texts]

            for old_text, new_text, reason in proposals:
                print(f"  PROPOSE close '{old_text}'  because '{new_text}': {reason}")

            if not old:
                outcome = "old saved nothing"
            elif closed_by_compare:
                outcome = "compare closed it"
            elif proposed:
                outcome = "checker proposed"
            else:
                outcome = "kept"
            closed = outcome in ("compare closed it", "checker proposed")
            passed = bool(old) and closed == (case["expect"] == "close")
            print(f"{'PASS' if passed else 'FAIL'}  {outcome}")
            results.append({"id": case["id"], "expect": case["expect"], "outcome": outcome, "passed": passed,
                            "proposals": [{"old": o, "new": n, "reason": r} for o, n, r in proposals]})

    print(f"\n{'id':<8}{'expect':<8}{'outcome':<20}result")
    for r in results:
        print(f"{r['id']:<8}{r['expect']:<8}{r['outcome']:<20}{'PASS' if r['passed'] else 'FAIL'}")
    for kind in ("close", "keep"):
        group = [r for r in results if r["expect"] == kind]
        print(f"{kind}: {sum(r['passed'] for r in group)}/{len(group)}")

    out = ROOT / "evals" / "results" / "hidden.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
