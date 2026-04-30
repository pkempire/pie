"""Initialise the architect DB and load the hand-curated seed components.

Idempotent: running it multiple times is safe (uses upserts on slug).
After the seed lands you can run enrich_one to web-ground each component
with its actual homepage + README, and mine_architectures to populate
the architecture co-occurrence graph.

Usage:
    python -m architect.scripts.seed
    python -m architect.scripts.seed --skip-embed   # for offline tests
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from .. import db
from .. import ingestion  # noqa: F401  ensure subpackage importable

logger = logging.getLogger(__name__)


SEED_PATH = Path(__file__).resolve().parent.parent / "db" / "seed_components.json"


def _slugify(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
    return re.sub(r"^-+|-+$", "", s)


def _embed_seed(card: dict) -> list[float]:
    """Embed the seed card's high-signal fields. Optional; safe to skip
    if the OpenAI key isn't available (the entry just won't show up in
    semantic search until enriched)."""
    from mempol import llm
    text = "\n".join(p for p in [
        card.get("name", ""),
        card.get("one_liner", ""),
        card.get("summary", ""),
        " ".join(card.get("tags") or []),
    ] if p)
    return llm.embed([text])[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_path", default=str(SEED_PATH), type=Path)
    parser.add_argument("--skip-embed", action="store_true",
                        help="Do not generate embeddings (useful in tests).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    db.init_db()
    seeds = json.loads(args.seed_path.read_text())
    logger.info("loading %d seed components", len(seeds))

    n = 0
    with db.connect() as conn:
        for card in seeds:
            slug = card.get("slug") or _slugify(card.get("name", ""))
            embedding = []
            if not args.skip_embed:
                try:
                    embedding = _embed_seed(card)
                except Exception as e:
                    logger.warning("embedding failed for %s: %s — continuing",
                                   card.get("name"), e)

            cid = db.upsert_component(
                conn,
                slug=slug,
                name=card["name"],
                aliases_json=card.get("aliases") or [],
                type=card.get("type", "tool"),
                one_liner=card.get("one_liner", ""),
                summary=card.get("summary", ""),
                capability_long=card.get("capability_long", ""),
                homepage_url=card.get("homepage_url"),
                github_url=card.get("github_url"),
                docs_url=card.get("docs_url"),
                mcp_url=card.get("mcp_url"),
                pricing_model=card.get("pricing_model"),
                hosted_or_self=card.get("hosted_or_self"),
                license=card.get("license"),
                embedding_json=embedding,
                last_verified_at=db._now(),
                importance=1.0,                     # hand-curated → trusted
            )
            for tag in (card.get("tags") or []):
                tag_id = db.upsert_tag(conn, slug=_slugify(tag), name=tag)
                db.tag_component(conn, cid, tag_id, weight=1.0)
            n += 1

    logger.info("seeded %d components into %s", n, db.DB_PATH)
    print(f"OK — DB at {db.DB_PATH}")


if __name__ == "__main__":
    main()
