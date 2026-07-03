# Temporal reasoning & time-aware agents — lit review

## The seven papers in one table

| arXiv | Short name | Failure mode | Benchmark | Headline | Thread |
|---|---|---|---|---|---|
| 2510.23853 | Temporally Blind | Agents over- or under-call tools because they ignore wall-clock between turns | TicToc, 76 scenarios, human-graded preferred decision | Best model 65% alignment with human preferences | behaving-in-time |
| 2601.13206 | Real-Time Deadlines | LLMs handle turn-budget deadlines fine but collapse under wall-clock deadlines | Paired LLM negotiation, two conditions (turn-based vs continuous time) | GPT-5.1 deal closure 32% with periodic time updates vs 4% without | behaving-in-time |
| 2502.05227 | Robotouille | ReAct loops break when tasks have asynchronous side-effects (cooking, interruptions, parallel waiting) | Robotouille sync + async splits | gpt-4o ReAct: 47% sync → 11% async | reasoning-over-time (planning-over-async-time) |
| 2505.13508 | Time-R1 | Foundation models cannot extrapolate past their cutoff and fail on creative future-event generation | Time-Bench (10y of news, three task families) | 3B model trained with 3-stage RL curriculum beats 671B DeepSeek-R1 on future-event prediction | reasoning-over-time |
| 2406.09170 | Test of Time | Existing temporal-reasoning benchmarks leak training data; can't isolate reasoning from memorisation | Synthetic temporal-logic tasks with controlled structure | Used as a probe rather than a leaderboard | reasoning-over-time |
| 2508.02045 | TDBench | Wikipedia-rooted TSQA benchmarks don't scale to application-specific time-sensitive facts | TDBench: SQL-generated TSQA pairs from temporal DB joins | Adds a "time accuracy" metric distinct from answer accuracy | reasoning-over-time |
| 2401.14192 | STG-LLM | LLMs can't ingest spatial-temporal graph data without a translator layer | STG-Tokeniser + adapter on standard ST forecasting benchmarks | Matches dedicated ST-forecasting models | reasoning-over-time |

## Synthesis

Three research threads run through this literature. The first and oldest is **reasoning over time**: can a model do arithmetic on dates, infer the correct year of an event, parse "three months before X", and so on. Test of Time, TDBench, STG-LLM, and the bulk of Time-R1 sit here. These benchmarks are mostly probes — the LLM is a calculator that happens to be asked about timestamps. The fix is data: synthetic curricula (Time-R1), database-backed pair generation (TDBench), or domain adapters (STG-LLM). Improvements compound but the failure mode is well-understood and the technique is unsurprising.

The second thread is much younger and much more interesting: **behaving in time**. The model is asked to act, not to compute. Temporally Blind frames the agent as a deciding-when-to-act agent — given a session that's been idle for ninety minutes, should the assistant re-fetch the user's calendar before answering, or trust what it cached? Real-Time Deadlines puts a clock on the table during a strategic dialogue and shows that LLMs that ignore the clock close 4% of deals while LLMs that get periodic time updates close 32%. The failure modes are not arithmetic errors. They are *omissions of behaviour that depends on time*. The fixes — periodic clock-in-prompt, retrieval freshness signals, post-training on tool-use trajectories — are barely studied and work poorly. Best result on TicToc is 65% alignment, and the paper concludes that prompt engineering "has limited effectiveness."

The third thread is **planning under asynchrony** — Robotouille's contribution. A ReAct loop runs sequentially (think, act, observe, think, act, observe). But the world doesn't. Some actions complete asynchronously (food cooks for ten minutes; an external API takes three seconds; a sub-agent runs in the background). Robotouille shows that gpt-4o drops from 47% to 11% the moment the planning task requires interleaving synchronous and asynchronous actions. This is closer to behaving-in-time than reasoning-over-time, but it's specifically about *plan structure* rather than *deciding when to act*.

Where mempol's TemporalBench fits: cleanly in the second thread. The six axes (proactive surfacing, deadline tracking, gap-aware resumption, staleness detection, rhythm recognition, commitment follow-through) are all behaving-in-time tasks. The closest competitor is **Temporally Blind's TicToc** — same shape (multi-turn dialogue with implicit time), same scoring philosophy (human-graded behavioural choice), but TicToc only tests one axis (decide-when-to-fetch) and reports one number. TemporalBench is a strict generalisation. Real-Time Deadlines tests one axis too (deadline awareness in negotiation) and is closer to a single-axis stress test than a benchmark suite.

Three ideas worth stealing. First, **periodic time-in-prompt updates** (from Real-Time Deadlines). Their +28-point gap on deal closure when the model sees a remaining-time tag at every turn is a strong piece of evidence that the model *can* condition on time, it just doesn't on its own. We should add a "current_time" header to every memory-policy observation. Second, **TicToc's pairwise human preference scoring** is a more honest metric than 0/1 accuracy for behaviour questions where there is a "correct" choice but no canonical reference answer. We should adopt this for at least the proactive-surfacing axis. Third, **Robotouille's sync-vs-async split** is a benchmark-design pattern: take the same task family and add asynchrony as the only varied axis. We can do the same on TemporalBench's deadline-tracking axis (deadline given as turn-count vs deadline given as wall-clock).

Two intro-quotable insights. (a) "Best alignment with human time perception on tool-calling decisions: 65%" — an honest indictment of where the field is. (b) The 47%→11% gap on Robotouille shows that *plan structure*, not *clock arithmetic*, is the bottleneck. mempol's per-op write trajectories are a special case of the same problem: the decision of *when* to write is structurally asynchronous from the decision of when to read.

## BibTeX

```bibtex
@article{tictoc2025,
  title={Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception},
  author={Anonymous},
  journal={arXiv preprint arXiv:2510.23853},
  year={2025}
}
@article{timer1_2025,
  title={Time-R1: Towards Comprehensive Temporal Reasoning in LLMs},
  author={Anonymous},
  journal={arXiv preprint arXiv:2505.13508},
  year={2025}
}
@article{testoftime2024,
  title={Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning},
  author={Anonymous},
  journal={arXiv preprint arXiv:2406.09170},
  year={2024}
}
@article{tdbench2025,
  title={Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive QA in Large Language Models},
  author={Anonymous},
  journal={arXiv preprint arXiv:2508.02045},
  year={2025}
}
@article{realtimedeadlines2026,
  title={Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues},
  author={Sehgal and Guntuku and Ungar},
  journal={arXiv preprint arXiv:2601.13206},
  year={2026}
}
@article{robotouille2025,
  title={Robotouille: An Asynchronous Planning Benchmark for LLM Agents},
  author={Gonzalez-Pumariega, Gonzalo and Su Yean and Sunkara and Choudhury},
  journal={arXiv preprint arXiv:2502.05227},
  year={2025}
}
@article{stgllm2024,
  title={How Can Large Language Models Understand Spatial-Temporal Data?},
  author={Liu and Yu and Wang and Ma and Shen},
  journal={arXiv preprint arXiv:2401.14192},
  year={2024}
}
```
