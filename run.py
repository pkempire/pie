#!/usr/bin/env python3
"""
PIE Runner — CLI entry point for the ingestion pipeline and UI.

Usage:
    python run.py                          # Full ingestion (all data)
    python run.py wiki                     # Launch personal wiki UI
    python run.py query                    # Launch interactive query CLI
    python run.py add notes.txt            # Ingest a single file into PIE
    python run.py add --title "Meeting"    # Ingest from stdin
    python run.py watch                    # Watch inbox/ folder for new files
    python run.py watch my_inbox/          # Watch a custom folder
    python run.py bench                    # Run all benchmarks (5 questions each)
    python run.py bench -b locomo -n 10   # Run LoCoMo, 10 questions
    python run.py --test                   # Test run (5 batches)
    python run.py --year 2025              # Only 2025+ conversations
"""

import argparse
import logging
import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pie.config import PIEConfig
from pie.ingestion.pipeline import IngestionPipeline


def setup_logging(verbose: bool = True):
    """Configure logging."""
    level = logging.INFO if verbose else logging.WARNING
    
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger("pie")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def make_web_search_fn():
    """
    Create a web search function.
    Uses Brave Search API if available, otherwise returns None.
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        logging.getLogger("pie").warning("BRAVE_API_KEY not set — web grounding disabled")
        return None
    
    import requests
    import time as _time

    _last_request_time = [0.0]  # mutable for closure
    _MIN_REQUEST_INTERVAL = 1.0  # seconds between requests (Brave free tier: 1 req/sec)

    def search(query: str, count: int = 3) -> list[dict]:
        """Search via Brave API with rate limiting and exponential backoff."""
        # Rate limiting: ensure minimum interval between requests
        elapsed = _time.time() - _last_request_time[0]
        if elapsed < _MIN_REQUEST_INTERVAL:
            _time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        max_retries = 4
        for attempt in range(max_retries):
            try:
                _last_request_time[0] = _time.time()
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": api_key},
                    params={"q": query, "count": count},
                    timeout=10,
                )

                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 1), 30)  # 2, 4, 8, 16 sec
                    logging.getLogger("pie").warning(
                        f"Brave API 429 (rate limited), waiting {wait}s (attempt {attempt+1}/{max_retries})"
                    )
                    _time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                results = []
                for r in data.get("web", {}).get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "description": r.get("description", ""),
                        "url": r.get("url", ""),
                    })
                return results

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logging.getLogger("pie").warning(f"Brave API error: {e}, retrying in {wait}s")
                    _time.sleep(wait)
                else:
                    logging.getLogger("pie").warning(f"Brave API failed after {max_retries} attempts: {e}")
                    return []

        return []  # all retries exhausted
    
    return search


def _quick_ingest(text: str, title: str, source: str, wm_path: Path):
    """Ingest a single piece of text into the world model."""
    import uuid, datetime, json as _json
    from pie.core.models import Conversation, Turn, DailyBatch, Entity, EntityType
    from pie.core.world_model import WorldModel
    from pie.core.llm import LLMClient, parse_extraction_result
    from pie.ingestion.prompts import (
        EXTRACTION_SYSTEM_PROMPT,
        build_extraction_user_message,
        format_conversations_for_extraction,
    )
    from pie.resolution.resolver import EntityResolver
    from pie.config import PIEConfig

    setup_logging()
    logger = logging.getLogger("pie")
    logger.info(f"Ingesting: '{title}' ({len(text)} chars)")

    config = PIEConfig()
    wm = WorldModel(persist_path=wm_path)
    llm = LLMClient()

    import time as _time
    now = _time.time()
    conv_id = str(uuid.uuid4())

    conv = Conversation(
        id=conv_id, title=title, created_at=now, updated_at=now,
        model=source, turns=[Turn(role="user", text=text, timestamp=now)],
    )
    batch = DailyBatch(
        date=datetime.datetime.now().strftime("%Y-%m-%d"),
        conversations=[conv],
    )

    context_preamble = wm.build_context_preamble(now) if wm.entities else ""
    conversations_text = format_conversations_for_extraction(
        batch.conversations, max_chars_per_turn=8000, max_turns_per_conversation=50
    )
    user_message = build_extraction_user_message(
        batch_date=batch.date, conversations_text=conversations_text,
        context_preamble=context_preamble, num_conversations=1,
    )

    result = llm.chat(
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model="gpt-5.4", json_mode=True, max_tokens=4000,
    )
    extraction = parse_extraction_result(
        raw=result["content"], conversation_ids=[conv_id], tokens=result["tokens"]
    )

    resolver = EntityResolver(world_model=wm, llm=llm, config=config.resolution)
    resolved = resolver.resolve(extraction.entities)

    creates, updates = 0, 0
    for r in resolved:
        if r.action == "create":
            entity = Entity(
                id=str(uuid.uuid4()),
                type=EntityType(r.extracted.type),
                name=r.extracted.name,
                current_state=r.extracted.state or {},
                first_seen=now, last_seen=now,
                importance=r.extracted.confidence or 0.5,
            )
            wm.entities[entity.id] = entity
            creates += 1
        elif r.action == "update" and r.matched_id:
            wm.update_entity_state(
                entity_id=r.matched_id,
                new_state=r.extracted.state or {},
                source_conversation_id=conv_id,
                timestamp=now,
                trigger_summary=f"[{source}] {title}",
            )
            updates += 1

    for sc in extraction.state_changes:
        match = wm.find_by_name(sc.entity_name)
        if match:
            wm.update_entity_state(
                entity_id=match.id,
                new_state={"description": sc.new_state} if isinstance(sc.new_state, str) else (sc.new_state or {}),
                source_conversation_id=conv_id,
                timestamp=now,
                trigger_summary=sc.what_changed,
                is_contradiction=sc.is_contradiction,
            )

    wm.rebuild_embedding_matrix()
    wm.save()
    logger.info(f"Done: {creates} entities created, {updates} updated, {len(extraction.state_changes)} state changes.")
    logger.info(f"World model now has {len(wm.entities)} entities.")


def _watch_inbox(inbox_dir: Path, wm_path: Path = Path("output/world_model.json")):
    """Watch a folder for new .txt/.md files and ingest them automatically."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    done_dir = inbox_dir / ".processed"
    done_dir.mkdir(exist_ok=True)

    setup_logging()
    logger = logging.getLogger("pie")
    logger.info(f"Watching {inbox_dir} for new .txt/.md files. Press Ctrl+C to stop.")
    logger.info(f"Drop files into {inbox_dir}/ to ingest them automatically.")

    import time as _time
    seen = set(p.name for p in done_dir.iterdir())

    try:
        while True:
            for fpath in sorted(inbox_dir.glob("*.txt")) + sorted(inbox_dir.glob("*.md")):
                if fpath.name not in seen:
                    seen.add(fpath.name)
                    logger.info(f"New file detected: {fpath.name}")
                    try:
                        text = fpath.read_text(encoding="utf-8")
                        _quick_ingest(text, fpath.stem, "inbox_watcher", wm_path)
                        # Move to processed
                        fpath.rename(done_dir / fpath.name)
                    except Exception as e:
                        logger.error(f"Failed to ingest {fpath.name}: {e}")
            _time.sleep(3)
    except KeyboardInterrupt:
        logger.info("Watcher stopped.")


def main():
    # Intercept shortcut subcommands before argparse
    if len(sys.argv) > 1 and sys.argv[1] == "wiki":
        wm_path = sys.argv[2] if len(sys.argv) > 2 else "output/world_model.json"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "pie/ui/app.py", "--",
            "--world-model", wm_path,
        ])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        wm_path = sys.argv[2] if len(sys.argv) > 2 else "output/world_model.json"
        subprocess.run([
            sys.executable, "-m", "pie.eval.query_interface",
            "--world-model", wm_path,
        ])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "add":
        # Quick ingest: python run.py add <file_or_text> [--title "..."] [--source "..."]
        _add_parser = argparse.ArgumentParser(prog="run.py add")
        _add_parser.add_argument("input", nargs="?", help="Path to .txt/.md file, or omit to read from stdin")
        _add_parser.add_argument("--title", default=None)
        _add_parser.add_argument("--source", default="manual_add")
        _add_parser.add_argument("--wm", default="output/world_model.json")
        add_args = _add_parser.parse_args(sys.argv[2:])

        if add_args.input:
            p = Path(add_args.input)
            text = p.read_text() if p.exists() else add_args.input
            title = add_args.title or p.stem
        else:
            print("Paste your conversation/notes, then press Ctrl+D:")
            text = sys.stdin.read()
            title = add_args.title or "stdin note"

        _quick_ingest(text, title, add_args.source, Path(add_args.wm))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        # Watch an inbox folder: python run.py watch [inbox_dir]
        inbox = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("inbox")
        _watch_inbox(inbox)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        # Run benchmarks: python run.py bench [--benchmark locomo] [--n 5] [--baseline pie_temporal]
        _bench_parser = argparse.ArgumentParser(prog="run.py bench")
        _bench_parser.add_argument(
            "--benchmark", "-b", default=None,
            choices=["locomo", "longmemeval", "msc", "all"],
            help="Which benchmark to run (default: all)",
        )
        _bench_parser.add_argument("--n", "-n", type=int, default=5, help="Questions per benchmark (default: 5)")
        _bench_parser.add_argument(
            "--baseline", default="pie_temporal",
            choices=["full_context", "naive_rag", "pie_temporal", "all"],
        )
        _bench_parser.add_argument("--model", default="gpt-4o")
        _bench_parser.add_argument("--output", "-o", default=None)
        _bench_parser.add_argument("--debug", "-d", action="store_true")
        bench_args = _bench_parser.parse_args(sys.argv[2:])

        bench_cmd = [
            sys.executable, "-m", "benchmarks.eval_harness",
            "--subset", str(bench_args.n),
            "--baseline", bench_args.baseline,
            "--model", bench_args.model,
        ]
        if bench_args.benchmark and bench_args.benchmark != "all":
            bench_cmd += ["--benchmarks", bench_args.benchmark]
        if bench_args.output:
            bench_cmd += ["--output", bench_args.output]
        if bench_args.debug:
            bench_cmd += ["--debug"]

        subprocess.run(bench_cmd)
        return

    parser = argparse.ArgumentParser(description="PIE Ingestion Pipeline")
    parser.add_argument("--test", action="store_true", help="Test mode (5 batches)")
    parser.add_argument("--batches", type=int, default=None, help="Limit number of batches")
    parser.add_argument("--conversations", type=int, default=None, help="Limit conversations parsed")
    parser.add_argument("--year", type=int, default=2025, help="Minimum year (default: 2025)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Extraction model")
    parser.add_argument("--no-web", action="store_true", help="Disable web grounding")
    parser.add_argument("--no-context", action="store_true", help="Disable sliding window context")
    parser.add_argument("--input", type=str, default=None, help="Path to conversations.json")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N batches")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N batches (for resuming)")
    parser.add_argument("--start-date", type=str, default=None, help="Resume from date (YYYY-MM-DD), skip earlier batches")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--stats", action="store_true", help="Show world model stats and exit")

    args = parser.parse_args()
    setup_logging(verbose=not args.quiet)

    logger = logging.getLogger("pie")

    # Stats mode — just load and display
    if args.stats:
        import json
        wm_path = Path(args.output) / "world_model.json"
        if not wm_path.exists():
            print(f"No world model found at {wm_path}")
            sys.exit(1)
        with open(wm_path) as f:
            wm = json.load(f)
        entities = wm.get("entities", {})
        transitions = wm.get("transitions", {})
        relationships = wm.get("relationships", {})
        print(f"\n{'='*50}")
        print(f"  PIE World Model — {wm_path}")
        print(f"{'='*50}")
        print(f"  Entities:      {len(entities)}")
        print(f"  Transitions:   {len(transitions)}")
        print(f"  Relationships: {len(relationships)}")
        # Type breakdown
        from collections import Counter
        types = Counter(e.get("type", "unknown") for e in entities.values())
        print(f"\n  Entity types:")
        for t, c in types.most_common():
            print(f"    {t:20s} {c:>4d}")
        print(f"{'='*50}\n")
        sys.exit(0)

    # Build config
    config = PIEConfig(
        output_dir=Path(args.output),
        use_web_grounding=not args.no_web,
        use_sliding_window=not args.no_context,
    )
    config.llm.extraction_model = args.model
    config.ingestion.year_min = args.year
    
    if args.input:
        config.conversations_path = Path(args.input)
    
    # Determine limits
    limit_batches = args.batches
    if args.test and limit_batches is None:
        limit_batches = 5
    
    logger.info("=" * 60)
    logger.info("PIE: Personal Intelligence Engine")
    logger.info("=" * 60)
    logger.info(f"Model: {config.llm.extraction_model}")
    logger.info(f"Input: {config.conversations_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Year filter: >= {config.ingestion.year_min}")
    logger.info(f"Sliding window: {config.use_sliding_window}")
    logger.info(f"Web grounding: {config.use_web_grounding}")
    if limit_batches:
        logger.info(f"Batch limit: {limit_batches}")
    logger.info("=" * 60)
    
    # Build web search function
    web_search_fn = None if args.no_web else make_web_search_fn()
    
    # Create and run pipeline
    pipeline = IngestionPipeline(config=config, web_search_fn=web_search_fn)
    
    pipeline.run(
        year_min=args.year,
        limit_batches=limit_batches,
        limit_conversations=args.conversations,
        save_every=args.save_every,
        skip_batches=args.skip,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
