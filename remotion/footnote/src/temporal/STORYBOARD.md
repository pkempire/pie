# "The Clock You Cannot Feel" — storyboard

3b1b-style explainer. 1920×1080 @ 30fps. Deep-navy ground, restrained palette
(`theme.ts`). Each scene = one `<Sequence>` with its own local frame clock.
Script: `research/content/temporal-awareness-the-clock-you-cannot-feel.md`.

**Status:** S1 built (`scenes/IntroFlatDatabase.tsx`). S2–S7 specced below — each
is ~1–2 hrs to build from the existing component kit.

---

### S1 — The Flat Database  ·  0–30s  ·  BUILT
Two messages (angry/vegetarian) → identical points in an embedding plane → time
slider sweeps → half-life curves diverge wildly while the DB stores a flat 1.0 →
punch: "the store is flat, it has no time axis."
*Components used:* `ChatBubble`, `EmbeddingPlane`, `WriteStamp`, `TimeSlider`,
`DecayCurve`, `Caption`.

### S2 — Two words, one label: reasoning ≠ awareness  ·  30–54s
Split screen. LEFT a *calendar* (reasoning): dates, an arrow doing "3 days before
Friday → Tuesday," a green check. RIGHT a *clock face you can feel* (awareness):
a sweeping second hand, a body-like pulse ring expanding at ~1Hz. Caption builds
the metaphor: "A calendar is something you reason *about*. A heartbeat is
something you *have*."
*New components:* `CalendarGlyph` (date cells + morph arrow), `Heartbeat`
(expanding ring on a sine, reuse spring), `SplitFrame` (two-pane layout).

### S3 — Why the blindness is structural  ·  54–77s
A token stream marches left→right (monospace boxes). User msg #1 at 9:00, user
msg #2 at 17:00 — but between them, only **one** token of separation. Zoom the
gap: label "8 hours" collapses into a single box labeled `\n`. Caption: "the
eight hours are simply not in its world." Then a ghosted "background process"
lane appears and is struck through — "there is no clock ticking between turns."
*New components:* `TokenStream` (animated sequence of labeled boxes, RoPE-style
position ticks), `GapCollapse` (interpolate a wide span → one cell).

### S4 — The receipts  ·  77–100s
Three stat cards count up from 0 with `spring`: **65%** (TicToc alignment),
**4%** (deals closed, no clock), **11%** (Robotouille async). Each card has a
one-line caption. Then all three converge with arrows into a single node:
"one deficit — no proprioception of elapsed time." 4% morphs to 32% to preview S5.
*New components:* `StatCard` (count-up number + label + source tag), `Converge`
(arrows from N cards into 1).

### S5 — Urgency beats the countdown  ·  100–122s
Two negotiation panels side by side. LEFT fed `"47s remaining"` (a precise
numeric readout). RIGHT fed `"time is running short — move toward agreement"`
(a warm pulsing bar). A deal-closure meter fills under each; the RIGHT one fills
higher. Caption: "the model wanted the *feeling* of the deadline, not the
number." This is the proof beat — land it slowly.
*New components:* `NegotiationPanel` (prompt readout + outcome meter),
`UrgencyBar` (color shifts cool→hot on a curve).

### S6 — Three layers of the real fix  ·  122–152s
A stacked stack diagram, built bottom-up:
1. **Architecture** — interleaved 5Hz micro-turns; show time woven *into* the
   sequence (token stream with time baked into each cell). Stat: TimeSpeak
   **64.7 vs 4.3**, a 15× bar.
2. **Runtime harness** — an agent loop with a `heartbeat` source emitting clock
   tokens that interrupt the model. Animate a pulse entering the loop.
3. **Memory** — a retrieved fact card annotated with felt staleness
   ("learned 47d ago · usually changes monthly · probably still true").
*New components:* `LayerStack` (3 stacked panels, build-on), `HeartbeatLoop`
(agent loop diagram with injected pulse), reuse `DecayCurve` for the memory layer.

### S7 — Thesis / close  ·  152–170s
Black. One line writes on: "Did you give it a clock to *read*, or a clock it can
*feel*?" Then the channel card. Keep it austere — no motion noise.
*New components:* `TypeOn` (per-character reveal), `EndCard`.

---

## Build order (highest payoff first)
1. **S4 StatCard count-ups** — cheap, high-impact, reusable everywhere.
2. **S5 urgency proof** — the most novel beat; it's the argument.
3. **S2 calendar-vs-heartbeat** — sells the core distinction visually.
4. S3, S6, S7.

## Render / preview
```bash
cd remotion/footnote
npm start                      # Remotion Studio → pick "TemporalExplainer", scrub live
npm run build src/index.tsx TemporalExplainer out.mp4   # full render
npx remotion still src/index.tsx TemporalExplainer frame.png --frame=660  # one frame (the curves)
```
(If a fresh machine: `npm install` first — native bindings are arch-specific.)

## Asset notes
- Keep everything **programmatic/vector** — that's what makes it read as 3b1b and
  stay crisp at 4K. Avoid raster where a shape will do.
- Image-gen is worth it for exactly two things: the **thumbnail** (a flat glowing
  database slab with a missing clock-hand) and an optional textured background
  vignette. Not for the diagrams.
- Figma is the right tool for the **static thumbnail + title card** only; the
  motion lives here in Remotion.
