"""V0 smoke test: exercise the 4 baseline Demo queries over HTTP.

Idempotently hits ``POST /api/query`` with the four canonical queries from
``docs/visuals/demo/screenshots/v0/README.md`` and writes the raw JSON
responses + a summary under ``docs/reports/v0_smoke/``.

Usage
-----
    # Start the demo server in another shell first:
    python scripts/run_demo_server.py

    # Then, from any shell:
    python scripts/v0_smoke_test.py
    python scripts/v0_smoke_test.py --server http://127.0.0.1:8000

Design notes
------------
* We use :mod:`urllib.request` (stdlib) to avoid adding a ``requests``
  dependency. The request body is explicitly UTF-8 encoded to dodge the
  Windows git-bash GBK quirk that corrupts non-ASCII literals piped through
  ``curl``.
* Output files are overwritten on each run; the script is safe to re-execute.
* The script does **not** start the server itself — it would outlive the
  caller's shell and is better handled by ``run_demo_server.py`` or
  ``start.bat``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "reports" / "v0_smoke"


@dataclass(frozen=True)
class SmokeCase:
    """One smoke-test case: the query, expected intent, and output file name."""

    idx: int
    query: str
    expected_intent: str
    filename: str


SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase(1, "你好", "greeting", "q1_greeting.json"),
    SmokeCase(2, "今天天气怎么样", "out_of_scope", "q2_out_of_scope.json"),
    SmokeCase(3, "帮我找内幕交易处罚案例", "case_retrieval", "q3_case_retrieval.json"),
    SmokeCase(4, "这种行为违反哪些法条", "law_grounding", "q4_law_grounding.json"),
)


def _post_query(server: str, query: str, timeout: float) -> dict:
    body = json.dumps({"query": query, "history": []}, ensure_ascii=False).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        f"{server.rstrip('/')}/api/query",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run(server: str, out_dir: Path, timeout: float = 180.0) -> dict:
    """Execute all smoke cases and persist results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for case in SMOKE_CASES:
        payload = _post_query(server, case.query, timeout)
        target = out_dir / case.filename
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        head = (
            (payload.get("answer", "") or "")
            .replace("\r", " ")
            .replace("\n", " | ")[:280]
        )
        results.append(
            {
                "idx": case.idx,
                "expected_intent": case.expected_intent,
                "query": case.query,
                "intent": payload.get("intent"),
                "intent_confidence": payload.get("intent_confidence"),
                "intent_method": payload.get("intent_method"),
                "response_backend": payload.get("response_backend"),
                "events_count": len(payload.get("events") or []),
                "answer_head": head,
                "pass": payload.get("intent") == case.expected_intent,
            }
        )
    summary = {"results": results, "ts": time.time()}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="Demo server base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write smoke-test outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = run(args.server, args.out_dir, args.timeout)
    except urllib.error.URLError as err:
        print(
            f"[v0_smoke_test] ERROR: unable to reach {args.server} — is the"
            " demo server running?\n"
            f"  detail: {err}",
            file=sys.stderr,
        )
        return 2

    failed = [row for row in summary["results"] if not row["pass"]]
    for row in summary["results"]:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"Q{row['idx']} [{status}] "
            f"expected={row['expected_intent']:<22} "
            f"got={row['intent']:<22} "
            f"conf={row['intent_confidence']} "
            f"backend={row['response_backend']} "
            f"events={row['events_count']}"
        )
    print(f"\nArtifacts written to: {args.out_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
