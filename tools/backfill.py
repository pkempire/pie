#!/usr/bin/env python3
"""
Backfill — clean and enrich the existing world model without re-ingesting.

1. Remove redundant transitions (same state re-extracted)
2. Compute importance scores via dynamics engine
3. Save cleaned model

Usage:
    python tools/backfill.py                    # Dry run (shows what would change)
    python tools/backfill.py --apply            # Apply changes and save
    python tools/backfill.py --apply --backup   # Backup first, then apply
"""

import argparse
import json
import math
import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.core.world_model import WorldModel, _normalize_value, _normalize
from pie.core.dynamics import TransitionDynamics
from pie.core.models import TransitionType


def dedup_transitions(wm: WorldModel, dry_run: bool = True) -> dict:
    """Remove transitions where the state didn't actually change."""
    removed = 0
    kept = 0
    by_entity = defaultdict(lambda: {"removed": 0, "kept": 0})

    transition_ids_to_remove = []

    for eid, entity in wm.entities.items():
        transitions = wm.get_transitions(eid, ordered=True)
        if len(transitions) < 2:
            by_entity[eid]["kept"] += len(transitions)
            kept += len(transitions)
            continue

        for i, t in enumerate(transitions):
            # Always keep creation transitions
            if t.transition_type == TransitionType.CREATION:
                by_entity[eid]["kept"] += 1
                kept += 1
                continue

            # Always keep contradiction transitions
            if t.transition_type == TransitionType.CONTRADICTION:
                by_entity[eid]["kept"] += 1
                kept += 1
                continue

            # Check the trigger summary
            trigger = t.trigger_summary or ""
            is_batch_update = trigger.startswith("Updated from ") and trigger.endswith(" batch")
            is_merge_dedup = trigger.startswith("Merged from intra-batch dedup")
            has_semantic_trigger = trigger and not is_batch_update and not is_merge_dedup

            # Transitions with semantic triggers are always real changes
            # (e.g. "Strategy refined to integrate Builder Fund...")
            if has_semantic_trigger:
                by_entity[eid]["kept"] += 1
                kept += 1
                continue

            # Generic "Updated from X batch" — LLM re-extracted same entity
            # with different wording. Check if NEW KEYS were added (structural
            # change) vs just description rewording (noise).
            from_s = t.from_state or {}
            to_s = t.to_state or {}

            structural_change = False

            # New keys that weren't in old state → real structural change
            new_keys = set(to_s.keys()) - set(from_s.keys())
            for k in new_keys:
                if _normalize_value(to_s.get(k)):
                    structural_change = True
                    break

            if structural_change:
                by_entity[eid]["kept"] += 1
                kept += 1
            else:
                # Just rewording of existing fields — noise
                by_entity[eid]["removed"] += 1
                removed += 1
                transition_ids_to_remove.append(t.id)

    # Apply removals
    if not dry_run:
        for tid in transition_ids_to_remove:
            t = wm.transitions.get(tid)
            if t:
                # Remove from entity index
                eid = t.entity_id
                if eid in wm._entity_transitions:
                    try:
                        wm._entity_transitions[eid].remove(tid)
                    except ValueError:
                        pass
                # Remove from transitions dict
                del wm.transitions[tid]

    return {
        "total_transitions": kept + removed,
        "kept": kept,
        "removed": removed,
        "removal_rate": f"{100*removed/(kept+removed):.1f}%" if (kept+removed) > 0 else "0%",
    }


def clean_aliases(wm: WorldModel, dry_run: bool = True) -> dict:
    """Remove aliases that are semantically unrelated to their parent entity.

    Uses word-overlap heuristic: an alias is suspicious if it shares zero
    meaningful words with the entity name AND all other aliases.  We also
    flag entities that exceed a sane alias cap (>10).

    Returns stats dict and prints per-entity detail.
    """
    stop_words = {
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
        "is", "it", "with", "by", "at", "from", "as", "be", "was", "are",
    }

    def _clean_words(text: str) -> set[str]:
        """Extract meaningful words: alphanumeric, ≥3 chars, not stop words."""
        raw = _normalize(text).split()
        return {
            w.strip("().,;:!?/&—–-")
            for w in raw
            if len(w.strip("().,;:!?/&—–-")) >= 3
        } - stop_words

    total_aliases = 0
    removed_aliases = 0
    entities_cleaned = 0
    details: list[dict] = []

    for eid, entity in wm.entities.items():
        if not entity.aliases:
            continue

        total_aliases += len(entity.aliases)
        name_words = _clean_words(entity.name)

        # Build "core words" from the entity name + first 3 aliases
        # (first 3 are most likely legitimate)
        core_words = name_words.copy()
        for a in entity.aliases[:3]:
            core_words.update(_clean_words(a))

        to_remove = []
        for alias in entity.aliases:
            alias_words = _clean_words(alias)

            # Check 1: alias shares at least one meaningful word with core words
            overlap = alias_words & core_words
            if overlap:
                continue  # looks legitimate

            # Check 2: short aliases (1-2 meaningful words) — require stronger
            # evidence: word must appear as substring in entity name, OR
            # alias itself is a substring of the entity name / vice versa.
            # Character overlap alone is NOT enough (too many false positives).
            if len(alias_words) <= 2 and len(alias) <= 25:
                # Very short aliases (≤5 chars) get an extra fuzzy check:
                # must have reasonable similarity to entity name
                from difflib import SequenceMatcher
                if len(alias) <= 5:
                    ratio = SequenceMatcher(
                        None, _normalize(alias), _normalize(entity.name)
                    ).ratio()
                    if ratio < 0.4:
                        to_remove.append(alias)
                        continue
                norm_alias = _normalize(alias)
                norm_entity = _normalize(entity.name)

                # Is the alias a substring of the entity name?
                if norm_alias in norm_entity or norm_entity in norm_alias:
                    continue

                # Does any alias word appear as a whole word in the entity name?
                entity_name_words = _clean_words(entity.name)
                if alias_words and alias_words & entity_name_words:
                    continue

                # Check against first 3 aliases (most likely legitimate)
                found_in_alias = False
                for existing in entity.aliases[:3]:
                    norm_existing = _normalize(existing)
                    if norm_alias in norm_existing or norm_existing in norm_alias:
                        found_in_alias = True
                        break
                    existing_words = _clean_words(existing)
                    if alias_words and alias_words & existing_words:
                        found_in_alias = True
                        break
                if found_in_alias:
                    continue

            # No word overlap, not a plausible abbreviation → suspicious
            to_remove.append(alias)

        if to_remove:
            entities_cleaned += 1
            removed_aliases += len(to_remove)
            details.append({
                "entity": entity.name,
                "type": entity.type.value,
                "total_aliases": len(entity.aliases),
                "removed": to_remove,
                "kept": [a for a in entity.aliases if a not in to_remove],
            })

            if not dry_run:
                for alias in to_remove:
                    wm.remove_alias(eid, alias)

    return {
        "total_aliases": total_aliases,
        "removed": removed_aliases,
        "entities_cleaned": entities_cleaned,
        "details": details,
    }


def compute_importance(wm: WorldModel) -> dict:
    """Run dynamics and write importance scores."""
    dynamics = TransitionDynamics(wm)
    report = dynamics.analyze()

    updated = 0
    for eid, profile in report.entity_profiles.items():
        entity = wm.entities.get(eid)
        if not entity:
            continue
        base = math.log2(1 + profile.total_transitions) / 8.0
        base = min(base, 1.0)
        recency = 1.0 - 0.8 * profile.staleness_score
        entity.importance = round(base * recency, 4)
        updated += 1

    # Stats
    importances = [e.importance for e in wm.entities.values()]
    nonzero = [i for i in importances if i > 0]

    return {
        "entities_scored": updated,
        "stale_count": len(report.stale_entities),
        "volatile_count": len(report.volatile_entities),
        "cooccurrences": len(report.cooccurrences),
        "avg_importance": round(sum(importances) / len(importances), 4) if importances else 0,
        "max_importance": round(max(importances), 4) if importances else 0,
        "nonzero_importance": len(nonzero),
    }


def recompute_embeddings(wm: WorldModel) -> dict:
    """Recompute embeddings for all entities that lack them.

    Embeddings are NOT saved in the JSON (too large), so after loading
    from disk every entity has embedding=None.  This function batch-
    computes them so the world model is ready for Tier 2 resolution.
    """
    from pie.core.llm import LLMClient

    missing = wm.get_entities_without_embeddings()
    if not missing:
        return {"total": len(wm.entities), "computed": 0, "already_had": len(wm.entities)}

    llm = LLMClient()
    BATCH_SIZE = 100
    computed = 0
    errors = 0

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        texts = []
        for e in batch:
            desc = e.current_state.get("description", str(e.current_state))
            texts.append(f"{e.name} ({e.type.value}): {desc}")

        try:
            embeddings = llm.embed(texts)
            for entity, emb in zip(batch, embeddings):
                wm.set_entity_embedding(entity.id, emb)
            computed += len(batch)
            print(f"    ... {computed}/{len(missing)} embeddings computed")
        except Exception as e:
            errors += 1
            print(f"    ERROR batch {i}-{i+BATCH_SIZE}: {e}")
            continue

    return {
        "total": len(wm.entities),
        "computed": computed,
        "already_had": len(wm.entities) - len(missing),
        "errors": errors,
        "embedding_tokens": llm.stats["total_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description="PIE Backfill — clean and enrich existing world model")
    parser.add_argument("--output", type=str, default="./output", help="World model directory")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--backup", action="store_true", help="Create backup before applying")
    parser.add_argument("--embeddings", action="store_true", help="Recompute all missing embeddings (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    wm_path = Path(args.output) / "world_model.json"
    if not wm_path.exists():
        print(f"No world model at {wm_path}")
        sys.exit(1)

    mode = "APPLYING" if args.apply else "DRY RUN"
    print(f"\n{'='*60}")
    print(f"  PIE BACKFILL — {mode}")
    print(f"{'='*60}")

    # Backup
    if args.apply and args.backup:
        backup_path = wm_path.parent / f"world_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(wm_path, backup_path)
        print(f"  Backup: {backup_path}")

    # Load
    wm = WorldModel(persist_path=wm_path)
    print(f"  Loaded: {len(wm.entities)} entities, {len(wm.transitions)} transitions, {len(wm.relationships)} relationships")

    # Step 1: Clean bad aliases
    print(f"\n  Step 1: Cleaning invalid aliases...")
    alias_result = clean_aliases(wm, dry_run=not args.apply)
    print(f"    Total aliases: {alias_result['total_aliases']}")
    print(f"    Removed: {alias_result['removed']} bad aliases from {alias_result['entities_cleaned']} entities")
    for d in alias_result["details"]:
        print(f"    {d['entity']} ({d['type']}): removed {len(d['removed'])} aliases")
        for a in d["removed"][:10]:
            print(f"      - {a}")
        if len(d["removed"]) > 10:
            print(f"      ... and {len(d['removed']) - 10} more")
        print(f"      Kept: {d['kept'][:5]}")

    # Step 2: Dedup transitions
    print(f"\n  Step 2: Deduplicating transitions...")
    dedup_result = dedup_transitions(wm, dry_run=not args.apply)
    print(f"    Before: {dedup_result['total_transitions']} transitions")
    print(f"    Removed: {dedup_result['removed']} redundant ({dedup_result['removal_rate']})")
    print(f"    After: {dedup_result['kept']} meaningful transitions")

    # Step 3: Compute importance
    print(f"\n  Step 3: Computing dynamics & importance scores...")
    importance_result = compute_importance(wm)
    print(f"    Scored: {importance_result['entities_scored']} entities")
    print(f"    Stale: {importance_result['stale_count']} | Volatile: {importance_result['volatile_count']}")
    print(f"    Co-occurrences: {importance_result['cooccurrences']}")
    print(f"    Avg importance: {importance_result['avg_importance']} | Max: {importance_result['max_importance']}")
    print(f"    Nonzero importance: {importance_result['nonzero_importance']} / {importance_result['entities_scored']}")

    # Step 4: Recompute embeddings (optional, requires API key)
    if args.embeddings:
        print(f"\n  Step 4: Recomputing embeddings...")
        embed_result = recompute_embeddings(wm)
        print(f"    Total entities: {embed_result['total']}")
        print(f"    Computed: {embed_result['computed']} embeddings")
        print(f"    Already had: {embed_result['already_had']}")
        if embed_result.get('errors'):
            print(f"    Errors: {embed_result['errors']} batches failed")
        print(f"    Embedding tokens used: {embed_result.get('embedding_tokens', 0):,}")
    else:
        print(f"\n  Step 4: Skipped embeddings (use --embeddings to enable)")

    # Step 5: Show top entities
    print(f"\n  Top 15 entities by importance:")
    top = sorted(wm.entities.values(), key=lambda e: e.importance, reverse=True)[:15]
    for i, e in enumerate(top, 1):
        print(f"    {i:2d}. {e.name} ({e.type.value}) — importance={e.importance:.4f}")

    # Save
    if args.apply:
        wm.save()
        print(f"\n  Saved to {wm_path}")
    else:
        print(f"\n  DRY RUN — no changes saved. Use --apply to save.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
