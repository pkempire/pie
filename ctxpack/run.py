"""ctxpack v0 — does a studied, budgeted context pack beat RAG and hand-written docs?

The seed of the context compiler: (corpus, task distribution, token budget) -> optimized pack.
v0 tests the unoptimized floor at ONE budget point with THREE conditions, deterministically scored:

  handwritten : the corpus's own human-written docs (README/EXPLAINER), truncated to budget
                -- the "your CLAUDE.md" baseline everyone actually uses
  rag         : lexical top-chunks retrieved per-question, packed to budget
                -- the query-adaptive baseline; pack must justify being static
  pack        : a two-stage studied compression of the corpus to budget (map: per-group notes;
                reduce: merge to pack) -- the compiled artifact, UNOPTIMIZED (no GEPA yet)

Scoring: regex AND-of-ORs on short answers. No LLM judge.
Run:  python -m ctxpack.run [--budget-tokens 4000] [--tasks ctxpack/tasks/mempol.jsonl]
Cost: ~$0.30 (gpt-5-mini). Writes ctxpack/results/<name>.json
"""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI

client = OpenAI()
MODEL = os.environ.get("CTXPACK_MODEL", "gpt-5-mini")
CHARS_PER_TOKEN = 4  # coarse budget accounting; consistent across conditions


def chat(system: str, user: str, max_tokens: int = 700, effort: str = "low") -> str:
    # NB: reasoning models spend completion tokens on reasoning; max_tokens must include
    # generous headroom or the visible output silently starves (observed: empty map notes).
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        reasoning_effort=effort,
        max_completion_tokens=max_tokens,
    )
    return r.choices[0].message.content or ""


# ---------------- corpus ----------------

def load_corpus(globs: list[str]) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for g in globs:
        for p in sorted(REPO.glob(g)):
            if p.is_file():
                try:
                    files.append((str(p.relative_to(REPO)), p.read_text(errors="ignore")))
                except OSError:
                    pass
    return files


def chunk_corpus(files: list[tuple[str, str]], size: int = 2400) -> list[tuple[str, str]]:
    chunks = []
    for name, text in files:
        for i in range(0, len(text), size):
            chunks.append((f"{name}:{i // size}", text[i : i + size]))
    return chunks


# ---------------- conditions ----------------

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())


def rag_context(question: str, chunks: list[tuple[str, str]], budget_chars: int) -> str:
    q = set(_tok(question))
    scored = sorted(chunks, key=lambda c: -len(q & set(_tok(c[1]))))
    out, used = [], 0
    for name, text in scored:
        if used + len(text) > budget_chars:
            continue
        out.append(f"### {name}\n{text}")
        used += len(text)
        if used > budget_chars * 0.95:
            break
    return "\n\n".join(out)


def handwritten_context(doc_paths: list[str], budget_chars: int) -> str:
    buf = []
    used = 0
    for rel in doc_paths:
        p = REPO / rel
        if not p.exists():
            continue
        t = p.read_text(errors="ignore")
        take = t[: max(0, budget_chars - used)]
        buf.append(f"### {rel}\n{take}")
        used += len(take)
        if used >= budget_chars:
            break
    return "\n\n".join(buf)


MAP_SYS = (
    "You are compiling a knowledge pack about a codebase. From the files below, extract the facts "
    "most useful for later answering precise questions: default constant values, class and function "
    "names and their roles, environment variables, reward/weight/config values, tool/op names, and "
    "the mechanism of each component in one line. Dense bullet notes. No prose, no fluff."
)
REDUCE_SYS = (
    "Merge these notes into ONE context pack of AT MOST {budget} tokens (~{chars} characters). "
    "Keep: exact constants and defaults, exact identifier names, env vars, op/tool lists, one-line "
    "mechanisms. Drop anything generic. Organize by component. The pack will be the ONLY context "
    "available to answer precise questions about this code."
)


def compile_pack(files: list[tuple[str, str]], budget_tokens: int, group_chars: int = 90_000) -> str:
    groups: list[str] = []
    cur, used = [], 0
    for name, text in files:
        t = text[:30_000]
        if used + len(t) > group_chars and cur:
            groups.append("\n\n".join(cur))
            cur, used = [], 0
        cur.append(f"### FILE {name}\n{t}")
        used += len(t)
    if cur:
        groups.append("\n\n".join(cur))
    notes = [chat(MAP_SYS, g, max_tokens=6000, effort="minimal") for g in groups]
    empty = sum(1 for x in notes if len(x.strip()) < 50)
    if empty:
        print(f"warning: {empty}/{len(notes)} map notes near-empty")
    budget_chars = budget_tokens * CHARS_PER_TOKEN
    pack = chat(
        REDUCE_SYS.format(budget=budget_tokens, chars=budget_chars)
        + " Use the budget fully — aim for close to the character limit; do not stop early.",
        "\n\n---\n\n".join(notes),
        max_tokens=budget_tokens * 2 + 4000,
    )
    return pack[:budget_chars]


ANSWER_SYS = (
    "Answer using ONLY the provided context. Be concise and specific: give the exact value, name, "
    "or list requested. If the context does not contain the answer, say 'not in context'."
)


def answer(context: str, question: str) -> str:
    # minimal effort + generous cap: reasoning tokens otherwise starve the visible answer
    return chat(ANSWER_SYS, f"Context:\n{context}\n\nQuestion: {question}",
                max_tokens=1200, effort="minimal")


def score(pred: str, accept: list[list[str]]) -> bool:
    p = pred.lower()
    return all(any(re.search(alt, p) for alt in alts) for alts in accept)


# ---------------- run ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="ctxpack/tasks/mempol.jsonl")
    ap.add_argument("--budget-tokens", type=int, default=4000)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    spec = [json.loads(l) for l in (REPO / args.tasks).read_text().splitlines() if l.strip()]
    meta = spec[0]
    tasks = spec[1:]
    budget_chars = args.budget_tokens * CHARS_PER_TOKEN

    files = load_corpus(meta["corpus_globs"])
    chunks = chunk_corpus(files)
    corpus_chars = sum(len(t) for _, t in files)
    print(f"corpus: {len(files)} files, {corpus_chars:,} chars; budget {args.budget_tokens} tok")

    t0 = time.time()
    pack = compile_pack(files, args.budget_tokens)
    hand = handwritten_context(meta["handwritten_docs"], budget_chars)
    print(f"pack compiled: {len(pack):,} chars in {time.time()-t0:.0f}s; handwritten: {len(hand):,} chars")

    conds = ["handwritten", "rag", "pack"]
    results = {c: [] for c in conds}
    for t in tasks:
        row = f"{t['id']:<28}"
        for c in conds:
            ctx = pack if c == "pack" else hand if c == "handwritten" else rag_context(t["q"], chunks, budget_chars)
            try:
                a = answer(ctx, t["q"])
            except Exception as e:  # keep the run alive; count as wrong
                a = f"[error: {e}]"
            ok = score(a, t["accept"])
            results[c].append({"id": t["id"], "q": t["q"], "a": a, "ok": ok})
            row += f"  {c}:{'OK' if ok else '..'}"
        print(row)

    n = len(tasks)
    summary = {c: round(sum(r["ok"] for r in results[c]) / n, 4) for c in conds}
    out = {
        "model": MODEL, "budget_tokens": args.budget_tokens, "n": n,
        "corpus_files": len(files), "corpus_chars": corpus_chars,
        "compression_ratio": round(corpus_chars / max(1, len(pack)), 1),
        "accuracy": summary, "pack_chars": len(pack), "results": results,
        "note": "v0 floor: pack is UNOPTIMIZED (no GEPA); deterministic regex scoring; single seed",
    }
    name = args.name or f"{Path(args.tasks).stem}_b{args.budget_tokens}"
    outdir = REPO / "ctxpack" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(out, indent=2))
    (outdir / f"{name}_pack.md").write_text(pack)
    print(f"\n==== accuracy @ {args.budget_tokens} tok budget (n={n}, deterministic) ====")
    for c in conds:
        print(f"  {c:<12} {summary[c]*100:5.1f}%")
    print(f"compression: {out['compression_ratio']}x  -> ctxpack/results/{name}.json")


if __name__ == "__main__":
    main()
