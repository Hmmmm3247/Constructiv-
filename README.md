# ICAP Coding Tutor

**Early warning for learning failure — catches gaps before they compound, predicts what future concepts are now at risk.**

Built for the Qwen Global AI Hackathon 2026.

---

## The Problem

Standard AI tutors answer the question. They don't know *why* you're asking it, whether the answer landed, or what you'll fail to learn next if the gap isn't caught now. A learner can appear to understand recursion, get a correct answer from GPT, and still fail the next three concepts because the understanding was shallow.

The gap isn't visible until the exam — by then it has compounded.

---

## What ICAP Changes

ICAP (Chi & Wylie, 2014) is a peer-reviewed framework classifying learner engagement into four levels:

| Level | What it looks like | What the system does |
|---|---|---|
| **Passive** | Reading, watching, asking "what is X?" | Teach first — no misconception yet, probing is wrong |
| **Active** | Copying, reproducing existing code | Surface the gap — push toward generating |
| **Constructive** | Generating new ideas, explaining in own words | Advance — deepen or bridge to next concept |
| **Interactive** | Defending, questioning, building on a prior exchange | Advance — they're teaching themselves |
| **Struggling** | Wrong output, syntax errors, stuck in a loop | Partial reveal — one clue, not the answer |

Constructive engagement produces measurably better retention than passive reading. The system forces learners up the ladder rather than giving them the answer.

---

## Architecture

Five specialized agents. One deterministic router. Zero guessing.

```
Learner message
      │
      ▼
┌─────────────┐
│  Classifier │  qwen-coder-plus — reads Python code + session history
│             │  → ICAP level + confidence score
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Router    │  Pure Python — zero LLM, zero token cost
│             │  Deterministic decision gate (every branch auditable)
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌────────────┐
│ Pro- │ │   Tutor    │  qwen-coder-plus
│ voc- │ │            │  modes: advance | deliver_concept
│ ateur│ │            │         partial_reveal | re_teach
│      │ └────────────┘
│qwen- │
│plus  │
└──────┘
       │
       ▼
┌─────────────┐
│  Trajectory │  Append-only event log — level history, failure modes
│  Store      │  consecutive_stuck() for re_teach escalation
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Auditor   │  qwen-plus — consequence projection
│             │  Ranks stuck concepts by downstream reach (deduplicated)
│             │  Confirmed gap (sandbox-verified) vs suspected (behavioral)
└─────────────┘
```

### Why the router uses no LLM

The routing decision is the system's only safety-critical gate. Using an LLM here would introduce latency, cost, and non-determinism where none is needed. Every branch is a named constant. Every decision is auditable in `router.py` without reading a prompt.

### The re_teach escalation

If a learner is stuck for 3+ consecutive turns (Passive/Active/Struggling without advancing), nudges and partial reveals are no longer pedagogy — they're a loop. The router escalates to `re_teach` mode, which forces the Tutor to use a different analogy or framing than anything already in `session_history`. The streak resets on a Constructive or Interactive win.

---

## Why Qwen — Specifically

Two different models for two different task types:

**`qwen-coder-plus` — Classifier and Tutor**
Both tasks require deep Python code understanding. The Classifier reads a learner's code and must distinguish "reproducing an example" (Active) from "generating a novel solution" (Constructive) — a semantic distinction that requires code-domain reasoning. The Tutor generates pedagogically correct Python snippets and must understand what a learner's broken code *intends* to do vs. what it does. `qwen-coder-plus` is purpose-built for exactly this.

**`qwen-plus` — Provocateur and Auditor**
The Provocateur generates natural-language probing questions (forcing questions, error plants, variant problems). The Auditor generates consequence narration from graph facts — one sentence per stuck concept, strictly constrained to names and numbers provided. Both are language tasks, not code tasks. `qwen-plus` handles these with lower latency.

**Structured JSON output is load-bearing**
Every agent returns structured JSON. The entire multi-agent pipeline depends on reliable JSON adherence — level, confidence, mode, consequence sentence. Qwen's instruction-following makes this work without fragile output parsing.

---

## The Novel Piece: Consequence Projection

The Auditor doesn't just report stuck concepts. It:

1. Traverses the curriculum dependency graph from each stuck concept
2. Ranks by downstream reach (deduplicated union — overlapping downstream sets counted once)
3. Classifies each gap as **confirmed** (sandbox caught a wrong output) or **suspected** (behavioral signal only)
4. Generates a one-sentence consequence: *"This confirmed gap in `functions_and_scope` blocks `list_comprehensions` and 11 other concepts — approximately 34 sessions of blocked learning."*
5. Returns a prescriptive summary: *"Fix `functions_and_scope` first — it unblocks the most downstream learning per unit effort."*

This means a teacher monitoring students sees failure before the exam, ranked by leverage, with a specific intervention point — not a generic "struggling" flag.

---

## Setup

```bash
git clone <repo>
cd qwen
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DASHSCOPE_API_KEY=your_key_here   # never commit this

python app.py
# → http://127.0.0.1:7860
```

---

## Demo Scenario (3 minutes)

**Turn 1 — First contact**
Ask: *"What is recursion?"*
→ Classifier: Passive. Router: `deliver_concept` (first contact rule). Tutor delivers IDEA + EXAMPLE + DO-step.

**Turn 2 — Active attempt**
Submit code that copies the example with minor changes.
→ Classifier: Active. Router: `expose_gap`. Provocateur probes: "What happens when n = 0?"

**Turns 3-5 — Stuck**
Submit wrong output repeatedly.
→ Classifier: Struggling × 3. Router escalates to `re_teach`. Tutor uses a different analogy — not the same explanation twice.

**Full Audit**
Click "Full Audit" → Consequence projection: confirmed gap ranked by downstream reach, prescriptive intervention recommendation.

---

## Classifier Accuracy

30-case harness covers all 5 ICAP levels including edge cases:
- RULE F: fresh start (zero concept history → always Passive regardless of phrasing)
- RULE C: cross-concept isolation (prior history on a different concept is ignored)
- Fluency trap: verbose answer that sounds Constructive but reproduces the example

Run: `python -m unittest tests.test_audit_fast -v`
Harness: `python icap_harness.py`

---

## Project Structure

```
agents/
  classifier.py      # ICAP classification (qwen-coder-plus)
  provocateur.py     # Gap exposure (qwen-plus)
  tutor.py           # Concept delivery + scaffolding (qwen-coder-plus)
  auditor.py         # Consequence projection (qwen-plus)
models.py            # Qwen model routing (single source of truth)
router.py            # Deterministic routing — pure Python, zero LLM
orchestrator.py      # Pipeline coordinator
store/
  trajectory.py      # Append-only event log + consecutive_stuck()
  concept_graph.py   # Curriculum dependency graph + reach ranking
  sandbox.py         # Safe Python execution for code verification
concept_packets/     # YAML concept definitions with test cases
demo/
  seed_trajectory.py # Seeds demo scenario with confirmed + suspected gap
tests/
  test_audit_fast.py
icap_harness.py      # 30-case classifier test suite
app.py               # Gradio UI with live decision trace
```
