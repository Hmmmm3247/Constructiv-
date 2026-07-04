"""
router.py — pure Python, no LLM, no token cost.

Reads Classifier output + sandbox result and decides which agent speaks and in
what mode. This is the system's only decision gate, so it must be deterministic
and auditable. Every branch is a single named constant.

Decision table
──────────────
  no_prior_attempt AND Passive                        → Tutor      / deliver_concept  ← teach first
  prior_count == 1 AND (Passive or Active)            → Challenger / challenge         ← drive to Constructive
  stuck_streak >= RETEACH_THRESHOLD (and not C/I)     → Tutor      / re_teach         ← nudges aren't working
  STRUGGLING                                          → Tutor      / partial_reveal
  Passive | Active,  conf >= CONF_THRESHOLD           → Provocateur/ expose_gap
  Passive | Active,  conf <  CONF_THRESHOLD           → Provocateur/ gentle_nudge
  Constructive | Interactive, conf >= CONF_THRESHOLD  → Tutor      / advance
  Constructive | Interactive, conf <  CONF_THRESHOLD  → Provocateur/ gentle_nudge

The challenge branch fires exactly once per concept — on the first follow-up turn
after deliver_concept. Prior_count == 1 means exactly one prior event exists for this
concept, which can only be the deliver_concept turn. Giving the learner a problem to
BUILD immediately after teaching drives them to Constructive level rather than waiting
for them to flounder through more Passive/Active turns first.

The low-confidence Constructive branch exists to defend against the fluency trap:
a learner who sounds Constructive but only at 0.5 confidence hasn't earned an advance.

The re_teach branch exists because partial_reveal and nudges loop forever on a learner
who genuinely lacks prerequisite understanding. After RETEACH_THRESHOLD consecutive
stuck turns, withholding the explanation is no longer pedagogy.
"""

from dataclasses import dataclass

CONF_THRESHOLD     = 0.65   # below this, don't trust a Constructive/Interactive label
RETEACH_THRESHOLD  = 3      # consecutive stuck turns before conceding nudges aren't working

STUCK_LEVELS   = {"Passive", "Active"}
HIGH_LEVELS    = {"Constructive", "Interactive"}


@dataclass(frozen=True)
class RouteDecision:
    agent: str          # "tutor" | "provocateur"
    mode: str           # "advance" | "partial_reveal" | "expose_gap" | "gentle_nudge"
    icap_level: str
    confidence: float
    failure_mode: str | None   # from sandbox; None if no code was submitted this turn
    reason: str = ""           # human-readable explanation of why this branch fired


def route(
    icap_level: str,
    confidence: float,
    failure_mode: str | None = None,
    no_prior_attempt: bool = False,
    stuck_streak: int = 0,
    prior_count: int = 0,
) -> RouteDecision:
    """
    icap_level      : one of Passive | Active | Constructive | Interactive | Struggling
    confidence      : 0.0–1.0 from the Classifier
    failure_mode    : from sandbox.evaluate()["failure_mode"], or None
    no_prior_attempt: True if the trajectory has no recorded events for this learner+concept
    stuck_streak    : consecutive prior turns (this concept) without advancing past
                      Passive/Active/Struggling — from TrajectoryStore.consecutive_stuck()
    prior_count     : total prior events for this concept (before this turn)
    """
    # Off-topic: not a programming question at all — redirect warmly, no trajectory recorded.
    if icap_level == "Off-topic":
        return RouteDecision("tutor", "redirect", icap_level, confidence, failure_mode,
                             reason="off-topic message — redirect without recording trajectory")

    # Curious first-time learner — no misconception exists yet, so probing is wrong.
    # Teach the concept, give a DO-step, then let the next turn probe if needed.
    if no_prior_attempt and icap_level == "Passive":
        return RouteDecision("tutor", "deliver_concept", icap_level, confidence, failure_mode,
                             reason="first contact — teaching before probing")

    # Immediately after deliver_concept, give the learner something to BUILD.
    # prior_count == 1 means exactly one prior event exists — the deliver_concept turn.
    # Passive/Active means they haven't generated yet. Challenge them now rather than
    # waiting for more passive turns.
    if prior_count == 1 and icap_level in STUCK_LEVELS:
        return RouteDecision("challenger", "challenge", icap_level, confidence, failure_mode,
                             reason="post-teach — generating a problem to drive Constructive level")

    # Nudges and partial reveals have had their chances and the learner still hasn't
    # advanced. A Constructive/Interactive turn here means they just got it — don't
    # override a win with a re-teach.
    if stuck_streak >= RETEACH_THRESHOLD and icap_level not in HIGH_LEVELS:
        return RouteDecision("tutor", "re_teach", icap_level, confidence, failure_mode,
                             reason=f"stuck {stuck_streak} turns — different explanation angle")

    if icap_level == "Struggling":
        return RouteDecision("tutor", "partial_reveal", icap_level, confidence, failure_mode,
                             reason="struggling — smallest unblocking clue only")

    if icap_level in STUCK_LEVELS:
        if confidence >= CONF_THRESHOLD:
            return RouteDecision("provocateur", "expose_gap", icap_level, confidence, failure_mode,
                                 reason=f"high confidence {confidence:.0%} — surfacing hidden gap")
        return RouteDecision("provocateur", "gentle_nudge", icap_level, confidence, failure_mode,
                             reason=f"low confidence {confidence:.0%} — nudging without assumption")

    if icap_level in HIGH_LEVELS:
        if confidence >= CONF_THRESHOLD:
            return RouteDecision("tutor", "advance", icap_level, confidence, failure_mode,
                                 reason="genuine progress — deepening or bridging forward")
        return RouteDecision("provocateur", "gentle_nudge", icap_level, confidence, failure_mode,
                             reason=f"low confidence {confidence:.0%} — one more probe before advancing")

    # unknown level — treat as low-confidence stuck; safe default
    return RouteDecision("provocateur", "gentle_nudge", icap_level, confidence, failure_mode,
                         reason="unrecognised level — safe default")


# ---------------------------------------------------------------------------
# quick self-test (python3 router.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        ("Passive",       0.80, None,              "provocateur", "expose_gap"),
        ("Active",        0.55, "wrong_formula",   "provocateur", "gentle_nudge"),
        ("Constructive",  0.82, None,              "tutor",       "advance"),
        ("Constructive",  0.50, None,              "provocateur", "gentle_nudge"),
        ("Interactive",   0.90, None,              "tutor",       "advance"),
        ("Struggling",    0.70, "missing_base_case","tutor",      "partial_reveal"),
    ]
    ok = 0
    for level, conf, fm, exp_agent, exp_mode in cases:
        d = route(level, conf, fm)
        passed = d.agent == exp_agent and d.mode == exp_mode
        ok += passed
        status = "OK" if passed else f"FAIL (got {d.agent}/{d.mode})"
        print(f"{status}  {level:15} conf={conf}  ->  {d.agent}/{d.mode}")
    print(f"\n{ok}/{len(cases)} passed")
