from __future__ import annotations

import argparse
import json

from csrc_rag.retrieval.engine import RetrievalEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BM25 retrieval baseline for a query.")
    parser.add_argument("query", nargs="+", help="User query in Chinese")
    args = parser.parse_args()

    engine = RetrievalEngine()
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

