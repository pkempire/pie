# The behavioral contract — what "learns from experience" means, testably

*General form + the research-agent instantiation. Iterated with Parth 2026-07-08. The goal
statement for everything in this repo; benchmarks are samplings of this contract.*

## General (any agent)
1. **Improvement** — performance on recurring task *families* rises with experience, on fresh
   instances, at fixed inference budget.
2. **Stability** — new experience never degrades established capability beyond noise.
3. **Currency** — beliefs track the world: bounded-time updates on contradiction; confidence in
   unrefreshed change-prone beliefs decays with elapsed time even without contradiction.
4. **Negative retention** — failed approaches are not re-attempted under equivalent conditions.
5. **Self-calibration** — predictions about itself (duration, success, confidence) converge to
   its own actuals, not population priors.
6. **Consolidation efficiency** — cost-per-use of knowledge falls with use frequency.
7. **Integration** — knowledge from separate episodes composes to solve tasks neither saw alone.
8. **Attributability** — every behavioral change traces to its causal experience and is reversible.

Meta: all eight simultaneously; all measured under evaluation noise smaller than claimed effects.

## Instantiated for AI research agents / long-horizon scientific discovery

1. **Compounding research skill.** Cost-per-validated-finding falls across a campaign; experiment
   designs improve (fewer confounds, better controls) on fresh questions. *Test: discovery rate
   and $/finding across campaign time.*
2. **Methodological stability.** Learning a new domain doesn't corrupt established methods or
   prior domain knowledge. *Test: standing methods/knowledge probes after each learning step.*
3. **Living literature + lab currency.** New papers or own results that contradict a working
   hypothesis flip it within bounded evidence; confidence in aging, unreplicated findings decays.
   *Test: assertion flips after contradicting result; hedging on stale unverified claims.*
4. **No re-falsification.** A falsified hypothesis or failed protocol is never re-run without
   new justification — the documented Co-Scientist failure ("cycled through variations of the
   same failed approaches", "confident rediscovery"). *Test: rediscovery rate of falsified
   hypotheses → 0 across sessions.*
5. **Experimental self-calibration.** Predicted effect sizes, runtimes, and costs converge to
   actuals; the agent knows which of ITS OWN findings will replicate. *Test: predicted-vs-actual
   curves; Brier score on replication predictions.*
6. **Cheap mastery.** Accumulated program knowledge gets cheaper to wield: a new question in a
   familiar field costs fewer tokens/papers each time (Kosmos re-reads ~1,500 papers per run —
   the anti-pattern). *Test: onboarding cost per new question at fixed quality, over time.*
7. **Cross-campaign synthesis.** A method from campaign A combines with an anomaly from campaign
   B into a hypothesis neither run produced — creativity as composition. *Test: cross-run
   compositional discovery tasks.*
8. **Evidence-traceable, retraction-propagating beliefs.** Every claim cites its papers/notebooks/
   runs; invalidate one source (retraction, discovered bug) and dependent claims flag or flip.
   *Test: the retraction-propagation test. No current system passes it.*
9. **(Discovery-specific) Frontier awareness.** Maintains an explicit registry of open questions
   and anomalies; allocates experiments by expected information gain, and checks claimed novelty
   against the literature. *Test: exploration choices beat greedy/random on information gain;
   novel-claim false-positive rate.*

Meta for science: the agent must hold its own learning to the same replication standard it holds
experiments — effects claimed about *itself* must exceed its measurement noise. (The property we
and the field kept failing.)

**Mechanisms proposed (see RESEARCH.md):** the metabolism (digest experience into verified
practice; integrate across context/cache/weights; excrete the superseded) and training on the
world's motion (next-diff prediction over literature streams, lab-result streams, and codebases)
— currency and calibration by construction. Current systems' violations, documented: Kosmos
(additive-only world model, 57.9% synthesis accuracy), Co-Scientist (repeats failed approaches,
loses insights across sessions), Sakana v2 (per-run tree, no cross-run memory), none with
retraction propagation.
