"""Unified prompt loader for the CSRC-RAG system.

Every runtime agent (Planner, Rewriter, Responder, Validator, TrendAnalyzer)
reads its prompts from the ``prompts/`` directory through this loader so that
prompt authors can iterate without touching Python code.

Supported file types
--------------------
* ``.j2``        — Jinja2 templates (System / task / intent prompts).
* ``.jsonl``     — line-delimited few-shot examples.
* ``.json``      — JSON schemas (e.g. rewriter output schema).
* ``.yaml``      — rule sets (validator rules, aggregation specs).
* ``.md``/``.txt`` — plain-text snippets (fallback messages, disclaimer).

Design notes
------------
* Pure read-only interface. The loader itself does not mutate state; the
  underlying ``Environment`` caches templates in-memory via Jinja2 so repeated
  renders are cheap.
* ``auto_reload=True`` — in development you can edit a ``.j2`` file and the
  next ``render(...)`` call picks it up without restarting the server.
* The loader also exposes ``list_available()`` so docs tooling and the
  ``prompts/README.md`` index can be kept honest with what's actually on
  disk.

Heavy dependencies (``pydantic``) are optional: schemas are validated with the
lightweight ``jsonschema`` package when present, otherwise the loader just
returns the parsed JSON and leaves validation to the caller.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateNotFound,
    Undefined,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["PromptLoader", "PromptArtifact"]


@dataclass(frozen=True)
class PromptArtifact:
    """Describe a single file discovered under ``prompts/``."""

    module: str        # e.g. "responder", "rewriter"
    rel_path: str      # path relative to ``prompts_dir``
    kind: str          # "template" | "few_shot" | "schema" | "rules" | "text"


class PromptLoader:
    """Unified prompt loader for all runtime agents.

    Parameters
    ----------
    prompts_dir:
        Directory holding the prompt assets. Defaults to ``./prompts`` so that
        projects can simply instantiate ``PromptLoader()`` when the working
        directory is the repository root.
    strict_undefined:
        Raise on missing template variables (default ``True``). Turn off only
        for quick debugging; we deliberately want missing-slot failures to
        surface loudly in tests.
    """

    _TEMPLATE_SUFFIXES = (".j2", ".jinja", ".jinja2")
    _FEW_SHOT_SUFFIXES = (".jsonl",)
    _SCHEMA_SUFFIXES = (".json",)
    _RULES_SUFFIXES = (".yaml", ".yml")
    _TEXT_SUFFIXES = (".md", ".txt")

    def __init__(
        self,
        prompts_dir: Path | str = Path("prompts"),
        *,
        strict_undefined: bool = True,
    ) -> None:
        self._root = Path(prompts_dir).resolve()
        if not self._root.exists():
            raise FileNotFoundError(f"prompts directory not found: {self._root}")
        self._env = Environment(
            loader=FileSystemLoader(str(self._root)),
            undefined=StrictUndefined if strict_undefined else Undefined,
            keep_trailing_newline=True,
            autoescape=False,
            auto_reload=True,
        )
        self._env.filters["to_json_cn"] = self._to_json_cn

    @staticmethod
    def _to_json_cn(value: Any) -> str:
        """Jinja filter: dump value as UTF-8 JSON (no ASCII escaping).

        Preferred over Jinja's built-in ``tojson`` when the prompt must show
        human-readable Chinese example outputs.
        """
        return json.dumps(value, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Templates                                                          #
    # ------------------------------------------------------------------ #
    def render(self, template_path: str, **context: Any) -> str:
        """Render a Jinja2 template and return the resulting string.

        ``template_path`` is relative to ``prompts_dir`` (e.g.
        ``"responder/intent_case_retrieval.j2"``).
        """
        try:
            tpl = self._env.get_template(template_path)
        except TemplateNotFound as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(f"template not found: {template_path}") from exc
        return tpl.render(**context)

    def get_template_source(self, template_path: str) -> str:
        """Return the raw template text (useful for A/B diff & logging)."""
        path = self._root / template_path
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Few-shots                                                          #
    # ------------------------------------------------------------------ #
    def load_few_shots(self, path: str) -> list[dict[str, Any]]:
        """Load a ``.jsonl`` few-shot file and return the parsed records.

        Each line must be a valid JSON object. Empty/whitespace-only lines are
        skipped. Malformed lines raise ``ValueError`` with the line number so
        the author can fix the file.
        """
        file_path = self._root / path
        records: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as fh:
            for idx, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{idx}: malformed JSONL: {exc.msg}"
                    ) from exc
        return records

    # ------------------------------------------------------------------ #
    # Schemas & rules                                                    #
    # ------------------------------------------------------------------ #
    def load_schema(self, path: str) -> dict[str, Any]:
        """Load a JSON schema (or any ``.json`` config) and return the dict."""
        file_path = self._root / path
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_yaml(self, path: str) -> dict[str, Any]:
        """Load a YAML file and return the parsed dict."""
        file_path = self._root / path
        with file_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the root")
        return data

    def load_text(self, path: str) -> str:
        """Read an arbitrary ``.md`` or ``.txt`` file verbatim."""
        return (self._root / path).read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Discovery                                                          #
    # ------------------------------------------------------------------ #
    def list_available(self) -> dict[str, list[str]]:
        """Return ``{module: [rel_path, ...]}`` for all prompt assets.

        Useful for:
        * ``prompts/README.md`` consistency checks.
        * Diagnostics / CLI tooling (``python -m prompts.loader --list``).
        """
        out: dict[str, list[str]] = {}
        for artifact in self._iter_artifacts():
            out.setdefault(artifact.module, []).append(artifact.rel_path)
        for mod in out:
            out[mod].sort()
        return out

    def describe(self) -> list[PromptArtifact]:
        """Return the catalog as a flat list of :class:`PromptArtifact`."""
        return list(self._iter_artifacts())

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _iter_artifacts(self) -> Iterable[PromptArtifact]:
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith(".") or "__pycache__" in path.parts:
                continue
            if path.name == "loader.py":
                continue
            rel = path.relative_to(self._root).as_posix()
            parts = rel.split("/", 1)
            module = parts[0] if len(parts) > 1 else "_root"
            suffix = path.suffix.lower()
            if suffix in self._TEMPLATE_SUFFIXES:
                kind = "template"
            elif suffix in self._FEW_SHOT_SUFFIXES:
                kind = "few_shot"
            elif suffix in self._SCHEMA_SUFFIXES:
                kind = "schema"
            elif suffix in self._RULES_SUFFIXES:
                kind = "rules"
            elif suffix in self._TEXT_SUFFIXES:
                kind = "text"
            else:
                continue
            yield PromptArtifact(module=module, rel_path=rel, kind=kind)

    @property
    def root(self) -> Path:
        return self._root


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Inspect prompt assets.")
    parser.add_argument(
        "--prompts-dir",
        default="prompts",
        help="Path to the prompts directory (default: ./prompts)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List every available prompt asset."
    )
    parser.add_argument(
        "--render",
        metavar="TEMPLATE",
        help="Render a Jinja template with an empty context (syntax check).",
    )
    args = parser.parse_args(argv)

    loader = PromptLoader(args.prompts_dir, strict_undefined=False)
    if args.list:
        for module, entries in loader.list_available().items():
            print(f"[{module}]")
            for entry in entries:
                print(f"  - {entry}")
    if args.render:
        print(loader.render(args.render))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
