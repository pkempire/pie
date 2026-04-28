# Generalising the eval beyond LoCoMo

## The cherry-pick problem

LoCoMo was chosen because every QA carries an `evidence` list of source
dia_ids. That makes the dense coverage reward computable and the per-turn
write episodes well-defined. It is also a research artefact. A real
ChatGPT export, an Asana workspace, a long Slack DM thread — none of
these come with per-question evidence labels, and most do not even ship
with a held-out QA set.

If the technique only works in the regime that has gold labels, it is
not useful. So the eval design has to answer two questions:

1. **What is the reward signal in a no-labels deployment?**
2. **What is the head-to-head comparison metric, when the evaluator
   cannot use gold evidence either?**

Below: four eval modes, ordered by how much they assume.

---

## Mode A — Full-conv, gold evidence (current LoCoMo eval)

The benchmark setting. Used for training and for paper headline numbers.

```
ingest_full_conv(W) -> M_conv
for each q in conv.questions:
    answer = R(q, M_conv)
    score  = judge(answer, gold)
report mean(score) per category
```

Coverage is computable because evidence labels exist. This is what the
current Phase B run trains against and what Section 5 of the paper
reports. Limit: every conversation needs a hand-annotated QA set.

---

## Mode B — Full-conv, self-generated questions

Stops needing the QA set; still needs an ingest substrate.

```
QA = generate_qas(conv, n=K, judge_model)        # GPT-4 reads conv, emits Q/A
ingest_full_conv(W) -> M_conv
for each q in QA:
    answer = R(q, M_conv)
    score  = judge(answer, gold)
report mean(score)
```

Where the questions come from is the design choice. Two reasonable
generators:
- **Comprehension**: GPT-4 reads the full conversation and writes
  questions whose answers it can verify from the same conversation.
  These test whether memory preserves enough information for
  question-answering at all.
- **Counterfactual**: GPT-4 reads the conversation up to turn $t$ and
  writes a question whose answer requires turn $t+k$. Then we test
  whether the agent's memory at time $t+k$ can answer it. This is much
  closer to how a real assistant gets used.

Both are cheap (~$0.05 per conversation). They lose the per-question
evidence labels and so cannot drive the dense coverage reward, but they
are perfect for *evaluation* where we just need a scoreable answer.

---

## Mode C — Head-to-head: learned-W vs hardcoded extraction

The comparison Pranay actually wants from his ChatGPT export. Two
write-side systems compete on the same conversations:

```
for conv in user_conversations:
    QA = generate_qas(conv, n=K)         # Mode B questions
    M_learned = ingest_with_learned_W(conv)
    M_pie     = ingest_with_pie_extraction(conv)   # the existing prompted KG builder
    for q in QA:
        a_learned = R(q, M_learned)
        a_pie     = R(q, M_pie)
        s_learned = judge(a_learned, gold_from_QA)
        s_pie     = judge(a_pie,     gold_from_QA)
    report:
        win_rate_learned_vs_pie
        per-category breakdown (single-hop / multi-hop / temporal / ...)
        memory-size comparison (entities, edges)
```

This is the strongest argument the paper can make in the absence of a
labelled benchmark: on the same data, does the learned write policy
build a memory that downstream reads better than the prompted
extraction does? It is exactly the experiment a venture investor or a
production-leaning reviewer wants to see.

The PIE-style baseline already exists in the repo (`pie/ingestion/`), so
this becomes a rollup script not new work.

---

## Mode D — Implicit, online: next-turn reward

The deployment-feasible signal. Requires a live conversational system,
not a benchmark.

```
At time t:
  agent issues writes w_t against memory M
At time t+1:
  user replies u_{t+1}
  weak_score = how well the previous turn's writes let R reconstruct
               context that matches u_{t+1}'s referents
  use weak_score as a delayed reward on w_t
```

Two ways to compute `weak_score`:
- LLM judge: feed (R's recall window, u_{t+1}) to a frozen judge that
  rates whether the recall would have helped form an appropriate reply.
- Implicit corrections: count user-typed corrections, repeated queries,
  and clarification turns within the next $k$ turns as negative signal.

We do not implement Mode D in this paper. We flag it in the limitations
section as the next step toward removing the benchmark dependency.

---

## What we actually run for the paper

| Mode | Used for | Why |
|---|---|---|
| A | training reward + Section 5 main results table | gives gold-evidence dense reward and the comparable LongMemEval / LoCoMo numbers reviewers expect |
| B | a robustness column in Section 5 | shows the trained policy still does well when the eval QAs come from a different source |
| C | a section called "Learned vs hand-coded extraction on real chat data" | the head-to-head with the existing PIE prompt-based extractor on the lead author's ChatGPT export |
| D | future work paragraph in the limitations | acknowledged but not evaluated |

The paper currently leans almost entirely on A. Promoting Mode B and
adding Mode C is what makes the contribution stop looking like a
benchmark cherry-pick.

---

## Per-category eval (cuts through the average)

Whatever mode we're in, we always report per category. LoCoMo categories:

```
single-hop       | one fact, one source turn      | easy lookup
multi-hop        | chain across turns / sessions  | tests relationships
open-domain      | subjective, multiple OK        | tests retrieval coverage
temporal         | "when did X happen"            | tests state transitions
adversarial      | misleading question            | tests contradiction handling
```

Even when the average looks unchanged, the per-category split usually
shows the learned policy winning on multi-hop and temporal (where typed
ops help) and at parity on single-hop. That breakdown is the actual
story.

---

## Implementation handles

- `mempol/eval/full_conv_runner.py` — Mode A on a conversation, returns
  per-category scores (already partially exists in `mempol/eval/runner.py`).
- `mempol/eval/qa_generator.py` — to add. Takes a conversation, returns
  a list of `(question, gold_answer, generator_model)`.
- `mempol/scripts/headtohead_chatgpt.py` — to add. Mode C on a folder
  of ChatGPT export `conversations.json` files, comparing learned-W
  against `pie/ingestion/pipeline.py`.
- `mempol/eval/per_category.py` — categoriser. For self-generated QAs we
  prompt the QA generator to also label the category.
