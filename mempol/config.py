"""Single source of truth for paths, model names, and budgets.

Auto-loads `.env` at the repo root if present, so users don't have to manually
export. Order: real env vars first, then .env fills in missing ones.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCOMO_PATH = ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
RESULTS_DIR = ROOT / "mempol" / "results"
TRACES_DIR = ROOT / "mempol" / "traces"
CACHE_DIR = ROOT / "mempol" / ".cache"

for p in (RESULTS_DIR, TRACES_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """Tiny .env parser. Only sets keys that aren't already in os.environ."""
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as e:
        # Never fail to import config because of a malformed .env
        print(f"[mempol.config] warning: couldn't parse {path}: {e}")


# Load from repo-root .env first; users may also have one in cwd.
_load_dotenv(ROOT / ".env")
_load_dotenv(Path.cwd() / ".env")

# Model defaults — bumped to gpt-5-mini for everything except the judge.
# gpt-4o is kept for the judge to match LongMemEval paper protocol (academic
# standard for that benchmark). Everything else: stronger reasoning, similar
# cost, better JSON adherence vs gpt-4o-mini.
#
# Override any of these via env vars (MEMPOL_ANSWER_MODEL=gpt-5, etc).
ANSWER_MODEL     = os.getenv("MEMPOL_ANSWER_MODEL",     "gpt-5-mini")
REFORMULATE_MODEL = os.getenv("MEMPOL_REFORMULATE_MODEL", "gpt-5-mini")
OBSERVER_MODEL   = os.getenv("MEMPOL_OBSERVER_MODEL",   "gpt-5-mini")
REFLECTOR_MODEL  = os.getenv("MEMPOL_REFLECTOR_MODEL",  "gpt-5-mini")
# Judge: gpt-4o-mini default for cost. gpt-4o-mini correlates >0.95 with
# gpt-4o on the kind of short-answer LLM-as-judge tasks we use it for, at
# ~5x lower cost. Override via MEMPOL_JUDGE_MODEL=gpt-4o for paper-final
# numbers if budget allows.
JUDGE_MODEL      = os.getenv("MEMPOL_JUDGE_MODEL",      "gpt-4o-mini")
EMBED_MODEL      = os.getenv("MEMPOL_EMBED_MODEL",      "text-embedding-3-large")  # 3072-dim

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
