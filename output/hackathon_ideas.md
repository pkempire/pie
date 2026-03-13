# Ironsite Spatial Intelligence Hackathon — Ideas & Notes
## UMD Startup Shell x Ironsite | Feb 20-22, 2026

---

## THE DATA

POV construction footage from job sites. Chest/helmet cams. Wood-frame residential construction. What you're working with:
- Shaky, egocentric video — worker doing their job, not scanning for you
- No labels, no metadata, no camera intrinsics
- Wide-angle / fisheye distortion (visible in the test video)
- Workers' hands, tools, feet constantly in frame
- Partial coverage — camera looks where the worker looks, nowhere else
- Multiple workers sometimes visible, sometimes just the cam wearer

From the test video specifically: two workers building a curved/dome wood-frame house. Second-story elevated platform, OSB subfloor, open framing, scattered tools, ladder, nail guns. Multiple obvious safety issues (unguarded edges, floor openings, no PPE, trip hazards).

---

## PATH A: Spatial Memory — "The AI Forgot There's a Hole Behind You"

### Problem
VLMs are stateless. They see one frame at a time with no persistent spatial memory. A construction worker maintains a mental 3D map of hazards, tools, coworkers, and exit routes — even when they're not looking at them. Current AI completely loses spatial awareness the moment the camera turns away.

### The Failure Demo
1. Show frame 1 of the test video: floor opening/hole visible in the subfloor
2. Play forward 60 seconds. Camera has turned, worker is now measuring lumber, looking at completely different area
3. Ask Gemini/Claude/GPT-5: "Based on everything you've seen, is there a fall hazard near the camera wearer? Where is it relative to their current position?"
4. Model either says "I don't see any fall hazards" (only analyzing last frame) or can't localize the remembered hazard relative to current orientation
5. A human would say "yeah there's an open hole about 8 feet behind me and to the left"

### Benchmark Design
Spatial memory tasks at increasing difficulty:
- **Level 1:** Hazard visible in current frame → models do fine
- **Level 2:** Hazard visible 5 sec ago, camera panned 45° → models start failing
- **Level 3:** Hazard visible 30 sec ago, camera turned 180° → models fail hard
- **Level 4:** "How many distinct hazards have you seen in the last 2 minutes? Where is each one relative to current camera position?" → complete failure
- **Level 5:** Multi-hop: "Is there a clear path from the worker's current position to the ladder without passing near any unguarded edges?" → requires fusing spatial memory of both the ladder location AND all edge locations

### Technique: Spatial World Model (PIE for Physical Space)
Build a persistent spatial representation that accumulates over time, exactly like PIE tracks entity states over conversations:

**Frame-level processing:**
- Depth Anything V2 → metric depth per frame
- Grounding DINO + SAM2 → detect and segment objects/hazards
- DROID-SLAM (or simpler VO + depth fusion) → camera pose per frame

**World model accumulation:**
- Each detected object gets a 3D position in a global coordinate frame
- Objects persist even when not visible (spatial memory)
- Relationships are computed and updated: distances, directions, zones
- Temporal tracking: when was each object last seen? Has it likely moved? (Tools move, structural elements don't)

**Query-time context injection:**
- User asks a spatial question
- World model retrieves all relevant spatial entities + their positions relative to current camera pose
- Structured spatial context injected into VLM prompt
- VLM reasons with awareness of things it can't currently see

### Why This Path
- Directly parallels your PIE research (temporal entity tracking → spatial entity tracking)
- DeepVO gives you the ego-motion estimation foundation
- Clean, demonstrable failure that frontier models won't solve soon (requires architecture changes, not just better training)
- Safety framing is visceral: "the AI forgot there's a hole behind you" immediately clicks for judges
- Scientifically rigorous: measurable degradation curves across memory difficulty levels

### Risks
- DROID-SLAM setup can be finicky, especially on fisheye POV footage
- Fallback: skip full SLAM, use frame-to-frame optical flow (RAFT) + depth for approximate ego-motion. Less accurate but still demonstrates the concept.
- If the construction footage they provide is very short clips rather than continuous video, temporal memory tasks become less interesting

---

## PATH B: Inference-Time Spatial Compute — "Give VLMs a Tape Measure"

### Problem
VLMs cannot make metric spatial measurements. They recognize objects but can't tell you distances, heights, angles, or whether something fits in a space. Construction safety depends entirely on specific measurements (6-foot fall protection threshold, swing radius clearances, load path projections).

### The Failure Demo
1. Show a frame with worker near an unguarded edge
2. Ask all three frontier models: "Is this worker in compliance with OSHA's 6-foot fall protection requirement? How far are they from the nearest unguarded edge?"
3. Models either hedge ("it's difficult to determine exact distances") or hallucinate a specific wrong number
4. Your pipeline computes the actual distance: 1.3 meters / 4.3 feet. Below threshold. Violation.
5. Same model, same image, but now with spatial measurements injected → correct, specific, actionable answer

### Technique: Depth-Augmented VLM Prompting
**Pipeline:**
1. Depth Anything V2 (metric) → per-pixel depth in meters
2. Grounding DINO → detect objects by name ("worker", "floor edge", "ladder", "nail gun")
3. SAM2 → pixel-perfect segmentation masks
4. Back-project to 3D: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
5. Compute spatial facts: pairwise distances, heights above ground, angular relationships, containment
6. Inject structured spatial JSON into VLM prompt alongside the image

**Calibration using known objects:**
- Hard hats: ~25cm diameter
- 2x4 lumber: 3.8 x 8.9 cm cross-section
- Standard door frame: 2.03m tall
- Tape measure (visible in test video frame 90!): use the markings as ground truth scale

### Benchmark
Categories to test:
1. Metric distance estimation ("how far apart are X and Y?")
2. Height estimation ("how tall is this wall section?")
3. Safety compliance ("does this violate OSHA 6-foot rule?")
4. Fit/clearance ("could a 4x8 sheet of plywood fit through that opening?")
5. Angle/orientation ("is this beam level?")
6. Counting with spatial constraint ("how many joists in the left bay?")

Run all three frontier models zero-shot. Then with your spatial augmentation. Show per-category improvement.

### Why This Path
- Cleaner to implement in 36 hours — no SLAM needed, per-frame pipeline
- Very visual demo (depth maps, measurement overlays, before/after comparison)
- Modular: each layer works independently, graceful degradation
- "Context engineering for vision" is a crisp, novel framing
- The safety angle has real-world impact: construction is the deadliest US industry, 1000+ deaths/year, most from falls

### Risks
- If Gemini 2.5 Pro already does well on basic distance estimation (it's getting better), the improvement delta might be small
- Fisheye distortion from POV cameras will mess up depth estimation if not corrected
- Metric depth models have their own error margins — need to be honest about cascading uncertainties

---

## PATH C: Fine-Tuned Spatial Specialist — "LoRA for Spatial Intelligence"

### Problem
Same as Path B, but the technique is model adaptation rather than inference-time augmentation.

### Technique: Self-Supervised Spatial LoRA
The WiFiGPT pattern applied to spatial reasoning:

1. **Generate your own training data (hours 1-4):**
   - Run Depth Anything V2 on N construction frames → metric depth
   - Run Grounding DINO → object detections
   - Compute 3D positions and pairwise distances
   - Auto-generate spatial QA pairs:
     - "How far is [worker] from [edge]?" → "1.8 meters"
     - "Is [object A] higher than [object B]?" → "Yes, by 0.6 meters"
     - "What is the tallest visible structure?" → "[scaffolding] at 4.1m"
   - Generate hundreds of (image, question, answer) triples automatically

2. **LoRA fine-tune a small VLM (hours 5-10):**
   - Base model: Qwen2-VL-7B (strongest small open VLM for spatial tasks)
   - Freeze base, add LoRA adapters (r=16, α=32)
   - Fine-tune on your generated spatial QA data
   - A few hours on a single A100

3. **Optional RL (stretch goal):**
   - GRPO with depth-derived reward signal
   - Reward = -|predicted_distance - depth_computed_distance|
   - Model learns spatial calibration through grounded feedback

### Evaluation: Four-Way Comparison
| | Zero-shot Frontier | Inference-Time Compute | LoRA Fine-Tuned 7B | LoRA + Inference-Time |
|---|---|---|---|---|
| Metric distance | ? | ? | ? | ? |
| Relative position | ? | ? | ? | ? |
| Safety compliance | ? | ? | ? | ? |
| ... | | | | |

This IS the presentation. Fill in the table. Show ablations. Scientifically minded.

### Why This Path
- Directly replicates your WiFiGPT methodology (frozen LLM + LoRA + domain signal)
- "We generated our own spatial training data from depth estimation, then fine-tuned a 7B model that outperforms GPT-5 on construction spatial tasks" is a banger line
- Shows you can bake spatial intelligence INTO the model, not just prompt around it
- The self-supervised data generation (no human labeling needed) is a genuine contribution

### Risks
- GPU availability at the hackathon — need at least one A100 or equivalent
- LoRA training on VLMs can be fiddly (learning rate, data formatting, etc.)
- If the improvement is marginal, the story is weaker
- Time-intensive — less time for polish on the demo

---

## PATH D: Ego ↔ Exo Viewpoint Transformation

### Problem
Construction footage is egocentric. But spatial reasoning often requires exocentric understanding — bird's eye layouts, site maps, structural plans. Ask a VLM to generate a top-down layout from a POV walkthrough and it completely falls apart.

### The Failure Demo
Give the VLM 5-10 frames from a POV walkthrough of the construction site. Ask: "Based on these images, draw me a rough floor plan / top-down layout of this space."

The model will produce something incoherent because it can't perform the ego → exo viewpoint transformation. It'll place rooms in wrong relative positions, get proportions wildly wrong, contradict itself between frames.

A human who walked this same path could sketch a reasonable floor plan from memory.

### Technique
- SLAM the POV video → camera trajectory + 3D point cloud
- Project point cloud to top-down view → automated floor plan / site layout
- Feed BOTH the egocentric frames AND the generated top-down layout to the VLM
- Now it can answer layout questions: "which room is adjacent to which?", "what's the fastest route from A to B?", "where are the exits relative to the current work area?"

### Why This Path
- The ego→exo transformation is a genuinely unsolved problem for VLMs
- Very visual demo: show the crappy VLM-generated floor plan vs your SLAM-generated top-down view
- Novel angle — most teams will think about single-frame spatial QA, not viewpoint transformation
- Connects to DeepVO (ego-motion) and the broader embodied AI / robotics space

### Risks
- Requires SLAM to work well on the footage (same risk as Path A)
- The "floor plan from video" problem is well-studied in robotics — judges might want novelty beyond just applying existing SLAM
- If the construction footage is outdoors / open framing (like the test video), "floor plan" is less meaningful than for an enclosed building

---

## PATH E: Multi-Hop Spatial Reasoning (the complex one)

### Problem
Current models handle simple spatial queries ("is X near Y?") increasingly well. They completely fail at multi-hop spatial reasoning that requires chaining spatial facts, simulating transformations, or reasoning about counterfactuals.

### Failure Prompts (these will break every model)
- "If the crane swings 90° clockwise, will its boom pass over any workers?" (requires: locate crane, determine boom length and current orientation, simulate rotation, project swept path, check worker positions against path)
- "Can the worker carry a 12-foot board from the lumber stack to the far wall without hitting the scaffolding?" (requires: locate lumber stack, locate destination, trace path, estimate scaffolding clearance, check board length against clearance)
- "If this wall section falls outward, what's in its fall zone?" (requires: estimate wall height, compute fall radius, identify everything within that radius)
- "Is there a safe path from the current position to the exit that avoids all overhead hazards?" (requires: full spatial map, pathfinding with 3D constraints)

### Technique: Spatial Chain-of-Thought + World Model
Decompose complex spatial queries into atomic sub-tasks:
1. Parse the query into spatial primitives (locate, measure, simulate, check)
2. Execute each primitive using the depth/detection pipeline
3. Chain results through a reasoning framework
4. VLM synthesizes the final answer from the chain of spatial facts

This is the two-pass pipeline pattern from sponsorFind — fast spatial computation followed by deep LLM reasoning.

### Why This Path
- Highest novelty — nobody else will tackle multi-hop spatial reasoning
- Most impressive if you pull it off
- Connects to chain-of-thought reasoning research, inference-time compute, and agent architectures

### Risks
- Hardest to implement in 36 hours
- Each sub-task needs to work correctly for the chain to produce a good answer — error propagation
- Might be hard to find enough multi-hop examples in the footage to build a convincing benchmark

---

## RECOMMENDED COMBO

**Primary: Path A (Spatial Memory) as the core finding / narrative**
- This is the most novel angle and hardest for frontier models to solve
- "AI has no spatial memory" is a crisp, demonstrable, dramatic finding
- The PIE parallel gives YOU a unique perspective no other team has

**Technical backbone: Path B (Inference-Time Spatial Compute)**
- This is your engineering layer regardless of which path you present
- Depth + detection + 3D back-projection is needed for all paths
- Build this first, it enables everything else

**Stretch: Path C (LoRA fine-tuning) if you have GPU access**
- Generates the four-way ablation table that makes the project scientifically rigorous
- "7B model fine-tuned on self-supervised spatial data outperforms GPT-5" is a headline result

**Demo flow:**
1. Open with the failure: show frontier models failing at spatial memory on real construction footage
2. Show the spatial world model accumulating knowledge as the video plays (the PIE parallel)
3. Show the measurement overlay + safety violation detection (the practical impact)
4. Show the benchmark: degradation curves for frontier models across memory difficulty levels, your system maintaining accuracy
5. (If time) Show the LoRA fine-tuned model + ablation table

---

## QUICK REFERENCE: TOOLS & MODELS

**Depth estimation:** Depth Anything V2 (metric), Metric3D v2, UniDepth (estimates intrinsics too — useful for unknown cameras)

**Object detection:** Grounding DINO (open-vocabulary, text-prompted), YOLO-World (faster)

**Segmentation:** SAM2 (segment anything, use with detection boxes)

**SLAM:** DROID-SLAM (dense, deep learning based), ORB-SLAM3 (classical, lighter), or simple VO + depth fusion as fallback

**VLMs to benchmark:** Gemini 2.5 Pro, Claude Opus, GPT-5 (use their API credits), plus open models for fine-tuning: Qwen2-VL-7B, InternVL2-8B, Florence-2

**VLMs with spatial focus:** SpatialVLM (Google Research) — benchmark this, if it also fails your finding is stronger

**Fine-tuning:** LoRA via Hugging Face PEFT + TRL, GRPO for RL

**Optical flow (fallback for ego-motion):** RAFT, GMFlow

**Other:** OpenCV for fisheye undistortion (the POV cameras have barrel distortion — correct this before depth estimation)

---

## WHAT TO DO IN FIRST 2 HOURS AT THE HACKATHON

1. Look at the actual data they provide. How long are the clips? Indoor/outdoor? Multiple visits to same area? This determines whether Path A (spatial memory) is viable.
2. Run 10-15 spatial prompts against Gemini/Claude/GPT-5 on frames from their footage. Find the most dramatic failures. Record everything.
3. Test Depth Anything V2 on their frames. Does the fisheye distortion break it? If yes, undistort first.
4. Decide: if clips are long continuous POV → go Path A (spatial memory). If clips are short / disconnected → go Path B (inference-time compute) + Path C (LoRA).
5. Divide work across team.
