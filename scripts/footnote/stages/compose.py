"""Stage 9: compose final video via Remotion.

Emits a props.json that the Remotion project consumes. The Remotion project
itself is at remotion/footnote/ — a sibling directory of scripts/footnote/.

We invoke `npx remotion render` if the project is installed. If not, we
emit the props.json and instructions so the user can render manually.

Why Remotion solves the temporal-consistency problem:
  - The LLM produces a SPEC (timing, text, position, image URL)
  - Remotion renders the spec deterministically with React + frame-by-frame
    interpolation
  - Same spec → identical pixels every time
  - No LLM in the render loop → no drift, no flicker, no inconsistency
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def build_props(researched_overlays: list[dict], layouts: list[dict],
                 video_path: Path) -> dict:
    """Combine researched-overlay specs with layout decisions into one
    Remotion props object."""
    overlays_props: list[dict] = []
    for ro, layout in zip(researched_overlays, layouts):
        prop = ro["proposal"]
        overlays_props.append({
            "id": f"overlay-{layout['overlay_idx']}",
            "timestamp_sec": prop.get("timestamp_sec", 0),
            "duration_sec": prop.get("duration_sec", 8),
            "type": prop.get("type", "fact-card"),
            "text": ro["text"],
            "citation_url": ro["citation_url"],
            "citation_label": ro["citation_label"],
            "image_url": ro.get("image_url"),
            "bbox": layout["bbox"],   # (x, y, w, h) normalized
            "entry_animation": layout["entry_animation"],
            "exit_animation": layout["exit_animation"],
        })

    return {
        "videoSrc": str(video_path.resolve()),
        "overlays": overlays_props,
    }


def render_with_remotion(props: dict, output_path: Path,
                          remotion_project: Path) -> Path | None:
    """Invoke Remotion render. Returns output path on success, None on failure."""
    props_path = remotion_project / "footnote_props.json"
    props_path.write_text(json.dumps(props, indent=2))
    logger.info("wrote props to %s", props_path)

    if not (remotion_project / "package.json").exists():
        logger.warning(
            "Remotion project not found at %s. To render, run:\n"
            "  cd %s\n"
            "  bun create video\n"
            "  npx remotion render src/index.tsx FootnoteComposition %s "
            "--props=footnote_props.json",
            remotion_project, remotion_project, output_path,
        )
        return None

    if shutil.which("npx") is None:
        logger.warning("npx not on PATH; install Node.js to render")
        return None

    cmd = [
        "npx", "remotion", "render",
        "src/index.tsx", "FootnoteComposition",
        str(output_path),
        f"--props={props_path}",
    ]
    logger.info("rendering: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=remotion_project)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error("remotion render failed: %s", e)
        return None


def compose(researched_overlays: list[dict], layouts: list[dict],
             video_path: Path, output_path: Path) -> Path | None:
    """Top-level compose entrypoint."""
    repo = Path(__file__).resolve().parents[3]
    remotion_project = repo / "remotion" / "footnote"
    props = build_props(researched_overlays, layouts, video_path)
    return render_with_remotion(props, output_path, remotion_project)
