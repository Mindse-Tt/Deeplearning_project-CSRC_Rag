from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.retrieval.engine import RetrievalEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval for a query.")
    parser.add_argument("query", nargs="+", help="User query in Chinese")
    args = parser.parse_args()

    engine = RetrievalEngine(retrieval_mode="hybrid")
    response = engine.search(" ".join(args.query))
    print(json.dumps(
        {
            "intent": response.intent,
            "query_plan": response.query_plan,
            "answer": response.answer,
            "events": response.events[:5],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
