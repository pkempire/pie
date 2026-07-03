"""Visualize official H-Net dynamic chunking against fixed chunking.

This script does not invent boundaries. If you pass `--model-path` and
`--config-path`, it imports the official H-Net code from `external/hnet`,
runs the model, and reads the returned `bpred_output[stage].boundary_mask`.

Without a checkpoint, it still writes the fixed-window panel and explains what
is missing, but it does not call that H-Net.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mempol.data.longmemeval import load as load_lme


def _compact(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n]


def _load_lme_text(variant: str, question_id: str, session: str | None, max_chars: int) -> str:
    for conv, qas in load_lme(variant=variant, n_convs=None, download=False):
        if conv.sample_id != question_id and qas[0].qid != question_id:
            continue
        turns = conv.turns
        if session is not None:
            turns = [t for t in turns if str(t.session) == str(session)]
        text = "\n".join(f"[{t.dia_id} | {t.session_date} | {t.speaker}] {t.text}" for t in turns)
        return _compact(text, max_chars)
    raise SystemExit(f"question_id not found: {question_id}")


def _fixed_chunks(text: str, chars: int) -> list[str]:
    return [text[i : i + chars] for i in range(0, len(text), chars)]


def _hnet_chunks(text: str, hnet_repo: Path, model_path: Path, config_path: Path, stage: int) -> list[dict[str, Any]]:
    import torch

    if str(hnet_repo) not in sys.path:
        sys.path.insert(0, str(hnet_repo))
    from generate import load_from_pretrained
    from hnet.utils.tokenizers import ByteTokenizer

    model = load_from_pretrained(str(model_path), str(config_path))
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode([text], add_bos=False)[0]["input_ids"]
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
    with torch.inference_mode():
        output = model.forward(input_ids, mask=mask)

    if stage >= len(output.bpred_output):
        raise SystemExit(f"stage {stage} unavailable; model returned {len(output.bpred_output)} boundary stages")
    boundary = output.bpred_output[stage].boundary_mask[0].detach().cpu().numpy().astype(bool).tolist()
    probs = output.bpred_output[stage].boundary_prob[0, :, 1].detach().float().cpu().numpy().tolist()

    text_bytes = text.encode("utf-8")
    chunks: list[dict[str, Any]] = []
    start = 0
    # First byte is usually a boundary. Close the previous chunk on later boundaries.
    for i, is_boundary in enumerate(boundary):
        if i == 0:
            continue
        if is_boundary:
            raw = text_bytes[start:i]
            chunks.append({
                "start_byte": start,
                "end_byte": i,
                "boundary_prob": probs[i],
                "text": raw.decode("utf-8", errors="replace"),
            })
            start = i
    if start < len(text_bytes):
        chunks.append({
            "start_byte": start,
            "end_byte": len(text_bytes),
            "boundary_prob": probs[-1] if probs else None,
            "text": text_bytes[start:].decode("utf-8", errors="replace"),
        })
    return chunks


def _panel(title: str, chunks: list[dict[str, Any]] | list[str]) -> str:
    colors = ["#eaf4ff", "#fff4db", "#eaffe9", "#ffecef", "#f2edff"]
    parts = [f"<section><h2>{html.escape(title)}</h2><div class='chunks'>"]
    for i, ch in enumerate(chunks):
        if isinstance(ch, str):
            text = ch
            meta = f"chunk {i + 1} · {len(ch)} chars"
        else:
            text = str(ch.get("text", ""))
            prob = ch.get("boundary_prob")
            prob_s = f" · p={prob:.3f}" if isinstance(prob, (float, int)) else ""
            meta = f"chunk {i + 1} · bytes {ch.get('start_byte')}-{ch.get('end_byte')}{prob_s}"
        parts.append(
            f"<div class='chunk' style='background:{colors[i % len(colors)]}'>"
            f"<div class='meta'>{html.escape(meta)}</div>"
            f"<pre>{html.escape(text)}</pre></div>"
        )
    parts.append("</div></section>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="longmemeval_s")
    ap.add_argument("--question-id", required=True)
    ap.add_argument("--session", default=None, help="Optional LongMemEval session id to visualize.")
    ap.add_argument("--max-chars", type=int, default=12_000)
    ap.add_argument("--fixed-chars", type=int, default=1_000)
    ap.add_argument("--hnet-repo", default=str(REPO / "external" / "hnet"))
    ap.add_argument("--model-path", default=None, help="Official H-Net .pt checkpoint.")
    ap.add_argument("--config-path", default=None, help="Official H-Net config JSON.")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", default="mempol/results/hnet_chunk_viz.html")
    args = ap.parse_args()

    text = _load_lme_text(args.variant, args.question_id, args.session, args.max_chars)
    fixed = _fixed_chunks(text, args.fixed_chars)
    hnet_chunks: list[dict[str, Any]] = []
    hnet_error = None

    if args.model_path and args.config_path:
        hnet_chunks = _hnet_chunks(
            text,
            Path(args.hnet_repo),
            Path(args.model_path),
            Path(args.config_path),
            args.stage,
        )
    else:
        hnet_error = "No --model-path/--config-path provided, so learned H-Net boundaries were not run."

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>H-Net Chunk Visualization</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #172033; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
.chunk {{ border: 1px solid #d6dde8; border-radius: 12px; margin: 12px 0; padding: 12px; }}
.meta {{ font-size: 12px; font-weight: 700; color: #667085; letter-spacing: .02em; margin-bottom: 8px; }}
pre {{ white-space: pre-wrap; margin: 0; font-size: 13px; line-height: 1.45; }}
.warn {{ background: #fff4db; border: 1px solid #e7b85c; border-radius: 12px; padding: 12px; }}
</style>
<h1>H-Net Dynamic Chunking vs Fixed Windows</h1>
<p><b>question_id:</b> {html.escape(args.question_id)} · <b>session:</b> {html.escape(str(args.session or 'all'))} · <b>chars visualized:</b> {len(text):,}</p>
{f"<div class='warn'>{html.escape(hnet_error)}</div>" if hnet_error else ""}
<div class="grid">
{_panel(f"Fixed {args.fixed_chars}-char windows", fixed)}
{_panel(f"Official H-Net boundary_mask stage {args.stage}", hnet_chunks) if hnet_chunks else "<section><h2>Official H-Net boundaries</h2><p>Provide checkpoint/config to render this panel.</p></section>"}
</div>
"""
    out.write_text(html_doc)
    meta = {
        "out": str(out),
        "question_id": args.question_id,
        "session": args.session,
        "text_chars": len(text),
        "fixed_chunks": len(fixed),
        "hnet_chunks": len(hnet_chunks),
        "hnet_error": hnet_error,
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
