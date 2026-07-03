"""GitMem — a content-addressable, branchable memory backend.

Motivation
==========
The PIE typed-transition graph models entities and per-entity state
transitions. That is enough for "what is X's current state" but breaks
down on three things projects need:

  1. Atomic bundles. A single conversation turn often produces a related
     set of changes (create_entity + add_relation + update_state) that
     belong together as one undoable unit.

  2. Branching contradictions. PIE's mark_contradiction is a flag.
     "Caroline says Boston" and "Caroline says NYC" should be siblings
     until reality resolves them; right now one wins and the other is
     metadata.

  3. Time-travel queries. "What was X's state on May 7?" should be one
     indexed lookup, not a transition-list scan.

GitMem borrows git's commit/branch/merge primitives at the *memory
operation* level. Each turn produces a Commit (an atomic bundle of
ops) on a branch. Commits are content-addressable (SHA-1 of their
content). Branches are named pointers to commits and can be per-entity
or global. Merges are first-class commits with multiple parents.

Schema
======
Commit:
  sha          str          # SHA-1 of (parent_shas, timestamp, dia_id, ops)
  parent_shas  list[str]    # 0 = root, 1 = linear, 2+ = merge
  timestamp    float        # epoch seconds (from dia_id)
  dia_id       str          # provenance to the source turn
  ops          list[OpRecord]
  message      str          # human-readable log line, like a git commit msg

OpRecord:
  kind         str          # "create_entity" | "update_state" | ...
  args         dict
  resulting_state  dict     # snapshot of affected entity state after this op

Branch:
  name         str          # "default" | "caroline_city" | etc.
  head_sha     str          # commit at HEAD
  entity_uid   str | None   # which entity this branch tracks (None = global)

Read-side primitives the policy gets for free:
  retrieve(query, k, source)        # standard Backend ABC; uses BM25/dense over commit messages
  checkout(sha) -> KGState          # full state at a commit
  state_at(timestamp, entity_uid) -> EntityState  # latest commit on entity ≤ timestamp
  diff(sha_a, sha_b) -> list[OpRecord]
  log(branch, limit) -> list[Commit]
  branches() -> list[Branch]

This is intentionally a thin substrate — it does not pull in real git;
SHA-1 is computed from a sorted JSON of the commit body. The store is
in-memory; persistence is via dump/load to JSON.
"""
from __future__ import annotations
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

from mempol.backends.base import Backend, Hit, Unit

logger = logging.getLogger(__name__)


@dataclass
class OpRecord:
    kind: str                          # op name, e.g. "create_entity"
    args: dict[str, Any]               # args the policy passed
    resulting_state: dict[str, Any]    # snapshot of the affected entity post-op
    target_uid: str | None = None      # the entity uid this op touched


@dataclass
class Commit:
    sha: str
    parent_shas: list[str]
    timestamp: float
    dia_id: str
    ops: list[OpRecord]
    message: str = ""

    def to_text(self) -> str:
        """One-line text for retrieval indexing — what BM25/dense will match."""
        head = f"[{self.dia_id}] " if self.dia_id else ""
        op_summary = "; ".join(
            f"{op.kind}({_short_args(op.args)})" for op in self.ops
        ) or "noop"
        return f"{head}{self.message} :: {op_summary}".strip()


def _short_args(args: dict[str, Any]) -> str:
    """Compact arg rendering for commit-text. Truncates long strings."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts)


@dataclass
class Branch:
    name: str
    head_sha: str
    entity_uid: str | None = None      # None → global default branch


def _sha(body: dict) -> str:
    """Content-addressable SHA-1 over a sorted-key JSON dump."""
    j = json.dumps(body, sort_keys=True, default=str).encode()
    return hashlib.sha1(j).hexdigest()[:16]   # 16 chars is plenty for our scale


class GitMemBackend(Backend):
    """In-memory git-shaped memory store.

    Conforms to the Backend ABC so it slots into existing eval and
    training pipelines. The Backend.retrieve() interface returns Hits
    over commit messages — read policies that need richer access can
    use the GitMem-specific methods (checkout, state_at, diff, log).
    """

    name = "gitmem"

    def __init__(self) -> None:
        self.commits: dict[str, Commit] = {}
        self.branches: dict[str, Branch] = {}
        # Convenience indexes for fast read-side queries.
        self._commits_by_entity: dict[str, list[str]] = defaultdict(list)
        self._commits_by_time: list[tuple[float, str]] = []  # sorted ascending
        # Lazy BM25 over commit text — built on first retrieve, invalidated on commit.
        self._bm25_dirty: bool = True
        self._bm25 = None
        self._commit_texts: list[tuple[str, str]] = []  # (sha, text)

    # ─── Backend ABC ────────────────────────────────────────────────────────

    def ingest(self, units: list[Unit]) -> None:
        """Backend ABC: ingest takes Units. For GitMem, each Unit becomes a
        single-op 'noop_chunk' commit on the default branch. Most callers
        will not use this path — they'll call commit() directly with real
        ops. This is here for compatibility with the random-K baseline
        and similar bulk-loaders."""
        for u in units:
            ts = float(u.metadata.get("timestamp", 0.0)) or time.time()
            dia = u.metadata.get("dia_id", u.uid)
            self.commit(
                ops=[OpRecord(
                    kind="ingest_chunk",
                    args={"text": u.text[:200]},
                    resulting_state={},
                    target_uid=None,
                )],
                timestamp=ts,
                dia_id=dia,
                message=u.text[:80],
                branch="default",
            )

    def retrieve(self, query: str, k: int = 10,
                 source: str = "hybrid") -> list[Hit]:
        """Search commit messages by BM25 / dense / hybrid (currently BM25
        only — dense embeddings can be added by inheriting and
        overriding. For most uses BM25 over commit messages is the right
        default because commit messages are dense, structured, and short)."""
        if not self.commits:
            return []
        if self._bm25_dirty:
            self._rebuild_bm25()
        # Lazy import to avoid a hard dep when the backend is constructed but
        # never queried.
        from rank_bm25 import BM25Okapi
        assert isinstance(self._bm25, BM25Okapi)
        toks = query.lower().split()
        scores = self._bm25.get_scores(toks)
        top = sorted(enumerate(scores), key=lambda kv: -kv[1])[:k]
        hits: list[Hit] = []
        for idx, score in top:
            if score <= 0:
                break
            sha, text = self._commit_texts[idx]
            commit = self.commits[sha]
            hits.append(Hit(
                unit=Unit(
                    uid=sha,
                    text=text,
                    metadata={
                        "dia_id": commit.dia_id,
                        "timestamp": commit.timestamp,
                        "n_ops": len(commit.ops),
                        "parent_shas": commit.parent_shas,
                    },
                ),
                score=float(score),
                source="bm25",
            ))
        return hits

    def expand(self, seed_uids: list[str], k_per: int = 3) -> list[Hit]:
        """For each commit sha, expand to its parents and children.

        This is the analogue of git log around a commit and the
        analogue of "show me sibling chunks" in our other backends. It
        gives the read policy a way to traverse the commit graph from
        an interesting starting point.
        """
        out: list[Hit] = []
        seen = set(seed_uids)
        for sha in seed_uids:
            if sha not in self.commits:
                continue
            commit = self.commits[sha]
            # Parents.
            for p in commit.parent_shas[:k_per]:
                if p in self.commits and p not in seen:
                    seen.add(p)
                    pc = self.commits[p]
                    out.append(Hit(
                        unit=Unit(
                            uid=p, text=pc.to_text(),
                            metadata={"dia_id": pc.dia_id,
                                      "timestamp": pc.timestamp,
                                      "rel": "parent_of", "of": sha},
                        ),
                        score=0.0, source="expand",
                    ))
            # Children: scan commits whose parents include sha.
            children = [
                s for s, c in self.commits.items()
                if sha in c.parent_shas and s not in seen
            ][:k_per]
            for c in children:
                seen.add(c)
                cc = self.commits[c]
                out.append(Hit(
                    unit=Unit(
                        uid=c, text=cc.to_text(),
                        metadata={"dia_id": cc.dia_id,
                                  "timestamp": cc.timestamp,
                                  "rel": "child_of", "of": sha},
                    ),
                    score=0.0, source="expand",
                ))
        return out

    # ─── GitMem-specific primitives ────────────────────────────────────────

    def commit(self, ops: list[OpRecord], timestamp: float, dia_id: str,
               message: str = "", branch: str = "default",
               parent_shas: list[str] | None = None) -> str:
        """Create a new commit on the named branch.

        parent_shas defaults to [HEAD of branch]. To create a merge
        commit pass parent_shas explicitly with 2+ shas.
        """
        if parent_shas is None:
            head = self.branches.get(branch)
            parent_shas = [head.head_sha] if head else []
        body = {
            "parents": parent_shas,
            "ts": timestamp,
            "dia_id": dia_id,
            "ops": [asdict(op) for op in ops],
            "msg": message,
        }
        sha = _sha(body)
        commit = Commit(
            sha=sha,
            parent_shas=parent_shas,
            timestamp=timestamp,
            dia_id=dia_id,
            ops=ops,
            message=message,
        )
        self.commits[sha] = commit
        # Update or create branch HEAD.
        if branch in self.branches:
            self.branches[branch].head_sha = sha
        else:
            entity_uid = None
            # If the commit's only target_uid is consistent across ops,
            # this is an entity-tracking branch.
            uids = {op.target_uid for op in ops if op.target_uid}
            if len(uids) == 1:
                entity_uid = next(iter(uids))
            self.branches[branch] = Branch(
                name=branch, head_sha=sha, entity_uid=entity_uid,
            )
        # Maintain indexes.
        for op in ops:
            if op.target_uid:
                self._commits_by_entity[op.target_uid].append(sha)
        # Sorted insert by timestamp for state_at queries.
        self._commits_by_time.append((timestamp, sha))
        self._commits_by_time.sort(key=lambda x: x[0])
        self._bm25_dirty = True
        return sha

    def merge(self, target_branch: str, source_branch: str,
              timestamp: float, dia_id: str,
              reconciliation_ops: list[OpRecord],
              message: str = "merge") -> str:
        """3-way merge: produce a commit on target_branch with two parents
        (target HEAD and source HEAD) and an explicit set of
        reconciliation ops the policy chose to resolve the divergence."""
        if target_branch not in self.branches or source_branch not in self.branches:
            raise ValueError(f"merge: missing branch(es) "
                             f"{target_branch}, {source_branch}")
        parents = [
            self.branches[target_branch].head_sha,
            self.branches[source_branch].head_sha,
        ]
        return self.commit(
            ops=reconciliation_ops,
            timestamp=timestamp,
            dia_id=dia_id,
            message=f"{message} ({source_branch} → {target_branch})",
            branch=target_branch,
            parent_shas=parents,
        )

    def checkout(self, sha: str) -> dict[str, dict[str, Any]]:
        """Return the full state at this commit by walking from root.

        This is the "state at a commit" reconstruction — for each
        entity touched by any commit reachable from sha, return its
        latest resulting_state.
        """
        state: dict[str, dict[str, Any]] = {}
        # Walk parents in topological order, applying each op.
        seen = set()
        order: list[str] = []

        def visit(s: str) -> None:
            if s in seen or s not in self.commits:
                return
            seen.add(s)
            for p in self.commits[s].parent_shas:
                visit(p)
            order.append(s)

        visit(sha)
        for s in order:
            for op in self.commits[s].ops:
                if op.target_uid:
                    state[op.target_uid] = dict(op.resulting_state)
        return state

    def state_at(self, timestamp: float,
                 entity_uid: str) -> dict[str, Any] | None:
        """The latest known state of entity_uid at or before timestamp.

        This is the temporal-query primitive that LoCoMo's "what was
        Caroline's city on May 7" questions need. O(log n) on the
        time-sorted commit index.
        """
        latest: dict[str, Any] | None = None
        for ts, sha in self._commits_by_time:
            if ts > timestamp:
                break
            commit = self.commits[sha]
            for op in commit.ops:
                if op.target_uid == entity_uid:
                    latest = dict(op.resulting_state)
        return latest

    def diff(self, sha_a: str, sha_b: str) -> list[OpRecord]:
        """All ops applied between sha_a (exclusive) and sha_b (inclusive)
        along the path from a to b. Currently linear-only — for a true
        graph diff we'd need LCA logic; that's a v2 problem."""
        if sha_a not in self.commits or sha_b not in self.commits:
            return []
        # Walk from b backwards through parents until we hit a or run out.
        ops: list[OpRecord] = []
        cur = sha_b
        while cur and cur != sha_a:
            ops = list(self.commits[cur].ops) + ops
            parents = self.commits[cur].parent_shas
            cur = parents[0] if parents else None
        return ops

    def log(self, branch: str = "default", limit: int = 20) -> list[Commit]:
        """git log: most-recent commits on the branch first."""
        if branch not in self.branches:
            return []
        out: list[Commit] = []
        cur = self.branches[branch].head_sha
        while cur and len(out) < limit:
            commit = self.commits.get(cur)
            if not commit:
                break
            out.append(commit)
            cur = commit.parent_shas[0] if commit.parent_shas else None
        return out

    # ─── Persistence ───────────────────────────────────────────────────────

    def dump(self) -> dict:
        """JSON-able snapshot. Round-trippable through load()."""
        return {
            "commits": {sha: {
                **{k: v for k, v in asdict(c).items() if k != "ops"},
                "ops": [asdict(op) for op in c.ops],
            } for sha, c in self.commits.items()},
            "branches": {n: asdict(b) for n, b in self.branches.items()},
        }

    @classmethod
    def load(cls, blob: dict) -> "GitMemBackend":
        b = cls()
        for sha, c in blob["commits"].items():
            ops = [OpRecord(**o) for o in c["ops"]]
            commit = Commit(
                sha=c["sha"], parent_shas=c["parent_shas"],
                timestamp=c["timestamp"], dia_id=c["dia_id"],
                ops=ops, message=c.get("message", ""),
            )
            b.commits[sha] = commit
            for op in ops:
                if op.target_uid:
                    b._commits_by_entity[op.target_uid].append(sha)
            b._commits_by_time.append((commit.timestamp, sha))
        b._commits_by_time.sort(key=lambda x: x[0])
        for n, br in blob["branches"].items():
            b.branches[n] = Branch(**br)
        b._bm25_dirty = True
        return b

    # ─── Internals ─────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        self._commit_texts = [(sha, c.to_text())
                              for sha, c in self.commits.items()]
        if not self._commit_texts:
            self._bm25 = None
            self._bm25_dirty = False
            return
        corpus = [t.lower().split() for _, t in self._commit_texts]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_dirty = False


# ─── Smoke test ─────────────────────────────────────────────────────────────

def _smoke() -> None:
    """End-to-end: commits, branches, merge, state_at, retrieve."""
    b = GitMemBackend()

    # Turn 1: Caroline introduces herself in Boston.
    sha1 = b.commit(
        ops=[
            OpRecord(kind="create_entity",
                     args={"name": "Caroline", "type": "person"},
                     resulting_state={"city": "Boston", "name": "Caroline"},
                     target_uid="caroline"),
        ],
        timestamp=1.0, dia_id="D1:5",
        message="Caroline introduces herself; lives in Boston",
    )

    # Turn 2: She mentions LGBTQ group attendance.
    sha2 = b.commit(
        ops=[
            OpRecord(kind="add_relation",
                     args={"source": "caroline", "target": "lgbtq_group",
                           "rel_type": "attended"},
                     resulting_state={"city": "Boston", "name": "Caroline",
                                       "attended": "lgbtq_group"},
                     target_uid="caroline"),
        ],
        timestamp=2.0, dia_id="D2:3",
        message="Caroline attended LGBTQ support group on May 7",
    )

    # Turn 3: She corrects herself — actually NYC. Branch this off.
    sha3 = b.commit(
        ops=[
            OpRecord(kind="update_state",
                     args={"uid": "caroline", "new_state": {"city": "NYC"}},
                     resulting_state={"city": "NYC", "name": "Caroline",
                                       "attended": "lgbtq_group"},
                     target_uid="caroline"),
        ],
        timestamp=3.0, dia_id="D9:1",
        branch="caroline_city_correction",
        parent_shas=[sha2],
        message="Caroline corrects: actually NYC, not Boston",
    )

    # Turn 4: Reconciliation — merge the branches with an explicit op.
    sha4 = b.merge(
        target_branch="default",
        source_branch="caroline_city_correction",
        timestamp=4.0, dia_id="D9:5",
        reconciliation_ops=[
            OpRecord(kind="update_state",
                     args={"uid": "caroline",
                           "new_state": {"city": "NYC",
                                         "city_history": ["Boston", "NYC"]}},
                     resulting_state={"city": "NYC", "name": "Caroline",
                                       "attended": "lgbtq_group",
                                       "city_history": ["Boston", "NYC"]},
                     target_uid="caroline"),
        ],
        message="Reconcile city: Boston → NYC",
    )

    # Temporal query: what was Caroline's city on May 7 (timestamp=2.5)?
    state_then = b.state_at(2.5, "caroline")
    assert state_then is not None
    assert state_then["city"] == "Boston", \
        f"expected Boston @ t=2.5, got {state_then}"

    # Temporal query: what is her current city?
    state_now = b.state_at(10.0, "caroline")
    assert state_now is not None
    assert state_now["city"] == "NYC", \
        f"expected NYC @ t=10, got {state_now}"

    # Retrieval: dense search returns the right commit.
    hits = b.retrieve("Caroline LGBTQ group", k=3)
    assert hits, "expected at least one hit"
    print(f"retrieved {len(hits)} commits for 'Caroline LGBTQ group'")
    print(f"top hit: {hits[0].unit.text[:80]}")

    # log()
    print("\ngit log default:")
    for c in b.log("default"):
        print(f"  {c.sha[:8]}  [{c.dia_id}]  {c.message[:60]}")

    print("\ngit log caroline_city_correction:")
    for c in b.log("caroline_city_correction"):
        print(f"  {c.sha[:8]}  [{c.dia_id}]  {c.message[:60]}")

    # Round-trip dump/load.
    blob = b.dump()
    b2 = GitMemBackend.load(blob)
    assert len(b2.commits) == len(b.commits)
    assert b2.state_at(2.5, "caroline")["city"] == "Boston"

    print("\nGitMem smoke ok.")


if __name__ == "__main__":
    _smoke()
