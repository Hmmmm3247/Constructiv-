import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Startup validation — fail loudly before the UI starts ─────────────────────
_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not _API_KEY:
    print(
        "\n[ICAP] ERROR: DASHSCOPE_API_KEY is not set.\n"
        "  Export it before starting:  export DASHSCOPE_API_KEY=sk-...\n"
        "  Or copy .env.example → .env and fill in your key.\n",
        file=sys.stderr,
    )
    sys.exit(1)

_ROOT         = Path(__file__).resolve().parent
_DATA_DIR     = Path(os.environ.get("ICAP_DATA_DIR", str(_ROOT / "data")))
_DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    from orchestrator import Orchestrator
except Exception as exc:
    print(f"[ICAP] Failed to import Orchestrator: {exc}", file=sys.stderr)
    sys.exit(1)

from openai import OpenAI
import gradio as gr
from models import QWEN_BASE_URL, QWEN_GENERAL

PACKETS_DIR     = str(_ROOT / "concept_packets")
TRAJECTORY_PATH = str(_DATA_DIR / "trajectory.jsonl")

orchestrator = Orchestrator(packets_dir=PACKETS_DIR, trajectory_path=TRAJECTORY_PATH)
_raw_client  = OpenAI(api_key=_API_KEY, base_url=QWEN_BASE_URL)

# ── Design tokens ──────────────────────────────────────────────────────────────

LEVEL_META = {
    "Passive":       {"icon": "🔴", "color": "#ef4444", "abbr": "P"},
    "Active":        {"icon": "🟡", "color": "#f59e0b", "abbr": "A"},
    "Constructive":  {"icon": "🟢", "color": "#22c55e", "abbr": "C"},
    "Interactive":   {"icon": "🔵", "color": "#3b82f6", "abbr": "I"},
    "Struggling":    {"icon": "🟠", "color": "#f97316", "abbr": "S"},
}

SIGNAL_COLORS = {"Pass": "#22c55e", "Fail": "#ef4444", "Error": "#f97316"}

ALL_CONCEPTS = [
    "variables_and_types",
    "conditionals",
    "functions_and_scope",
    "loops_iteration",
    "string_methods",
    "recursion_basics",
    "list_comprehensions",
    "error_handling",
    "sorting_and_searching",
    "classes_and_objects",
]

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

footer { display: none !important; }

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App header ── */
#app-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  border-radius: 14px;
  padding: 1.4rem 2rem;
  margin-bottom: 1rem;
}

/* ── Chat panel headers ── */
.panel-header-raw {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px 10px 0 0;
  padding: 0.65rem 1rem;
  border-bottom: none;
}
.panel-header-icap {
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 10px 10px 0 0;
  padding: 0.65rem 1rem;
  border-bottom: none;
}

/* ── Shared card base ── */
.ic-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.6rem;
  box-shadow: 0 1px 4px rgba(15,23,42,0.06);
}

/* ── Risk items ── */
.risk-item { padding: 0.55rem 0.75rem; border-radius: 8px; margin: 0.45rem 0; background: #fafafa; }
.risk-confirmed { border-left: 3px solid #dc2626; }
.risk-suspected { border-left: 3px solid #d97706; }

/* ── Sidebar labels ── */
.sidebar-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: #94a3b8;
  margin: 0.9rem 0 0.3rem;
}

/* ── Concept pill selector ── */
#concept-row .wrap { gap: 0.4rem !important; flex-wrap: wrap !important; }
#concept-row label {
  border-radius: 20px !important;
  border: 1.5px solid #e2e8f0 !important;
  padding: 0.25rem 0.8rem !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  background: #fff !important;
  color: #475569 !important;
  transition: background 0.15s, border-color 0.15s !important;
}
#concept-row input[type=radio]:checked + label,
#concept-row label:has(input[type=radio]:checked) {
  background: #6366f1 !important;
  border-color: #6366f1 !important;
  color: #ffffff !important;
}
#concept-row .block { padding: 0 !important; border: none !important; background: transparent !important; }

/* ── Message input ── */
#msg-box textarea {
  font-size: 0.95rem !important;
  border-radius: 10px !important;
  border: 1.5px solid #e2e8f0 !important;
}
#msg-box textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important; }

/* ── Buttons ── */
#submit-btn {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  min-height: 52px !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.28) !important;
}
#clear-btn, #audit-btn { border-radius: 8px !important; font-size: 0.83rem !important; font-weight: 500 !important; }

/* ── Chatbot panels ── */
#chatbox-raw  { border-radius: 0 0 12px 12px !important; border: 1px solid #e2e8f0 !important; border-top: none !important; }
#chatbox-icap { border-radius: 0 0 12px 12px !important; border: 1px solid #c7d2fe !important; border-top: none !important; }
"""

# ── Raw AI call ────────────────────────────────────────────────────────────────

def _raw_ai_response(message: str, code: str | None) -> str:
    """Call Qwen with no scaffolding — just answers the question directly."""
    content = message.strip()
    if code and code.strip():
        content += f"\n\nMy code:\n```python\n{code.strip()}\n```"
    resp = _raw_client.chat.completions.create(
        model=QWEN_GENERAL,
        messages=[
            {"role": "system", "content": "You are a helpful Python programming assistant. Answer the user's question directly and completely."},
            {"role": "user",   "content": content},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _level_track(level: str) -> str:
    ordered = [
        ("Passive",      "#ef4444", "P"),
        ("Active",       "#f59e0b", "A"),
        ("Constructive", "#22c55e", "C"),
        ("Interactive",  "#3b82f6", "I"),
    ]
    is_struggling = (level == "Struggling")
    parts = ['<div style="display:flex;align-items:center;gap:5px;margin:0.5rem 0 0.2rem">']
    for name, color, abbr in ordered:
        active = (level == name)
        bg     = color if active else "#f1f5f9"
        border = color if active else "#e2e8f0"
        txt    = "white" if active else "#94a3b8"
        weight = "700" if active else "500"
        parts.append(
            f'<div title="{name}" style="background:{bg};border:2px solid {border};color:{txt};'
            f'border-radius:6px;padding:2px 9px;font-size:0.76rem;font-weight:{weight};'
            f'min-width:28px;text-align:center">{abbr}</div>'
        )
        if name != "Interactive":
            parts.append('<span style="color:#cbd5e1;font-size:0.8rem;margin:0 1px">→</span>')
    if is_struggling:
        parts.append(
            '<div style="background:#f97316;color:white;border-radius:6px;'
            'padding:2px 9px;font-size:0.76rem;font-weight:700;margin-left:8px">⚠ Struggling</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def _chip(text: str, bg: str = "#f1f5f9", color: str = "#475569", border: str = "#e2e8f0") -> str:
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {border};'
        f'border-radius:5px;padding:1px 8px;font-size:0.74rem;font-family:monospace;font-weight:600">'
        f'{text}</span>'
    )


def _step(num: str, title: str, body: str) -> str:
    return (
        f'<div style="display:flex;gap:0.6rem;align-items:flex-start;margin:0.5rem 0">'
        f'<div style="background:#6366f1;color:white;border-radius:50%;width:20px;height:20px;'
        f'font-size:0.68rem;font-weight:700;display:flex;align-items:center;justify-content:center;'
        f'flex-shrink:0;margin-top:1px">{num}</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:0.67rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#94a3b8;font-weight:700;margin-bottom:3px">{title}</div>'
        f'{body}'
        f'</div></div>'
    )


def _icap_md(resp) -> str:
    meta        = LEVEL_META.get(resp.icap_level, {"icon": "⚪", "color": "#6b7280"})
    icon, color = meta["icon"], meta["color"]
    _agent_meta = {
        "provocateur": ("🎯", "Provocateur",  "qwen-plus"),
        "challenger":  ("🧩", "Challenger",   "qwen-coder-plus"),
    }.get(resp.agent_used, ("📖", "Tutor", "qwen-coder-plus"))
    agent_emoji, agent_name, model_name = _agent_meta

    signal_html = ""
    if getattr(resp, "signal", None):
        sig_color   = SIGNAL_COLORS.get(resp.signal, "#94a3b8")
        signal_html = f' {_chip(resp.signal, bg=sig_color+"18", color=sig_color, border=sig_color+"40")}'

    step1 = (
        f'<div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap">'
        f'<span style="color:{color};font-weight:700;font-size:0.95rem">{icon} {resp.icap_level}</span>'
        f'<span style="color:#94a3b8;font-size:0.8rem">· {resp.confidence:.0%}</span>'
        f'{signal_html}</div>'
        f'{_level_track(resp.icap_level)}'
        f'<div style="color:#475569;font-size:0.8rem;margin-top:0.3rem;line-height:1.45;font-style:italic">'
        f'"{resp.evidence}"</div>'
    )
    step2 = (
        f'<div style="margin-bottom:3px">{_chip(resp.mode, bg="#eef2ff", color="#6366f1", border="#c7d2fe")}</div>'
        f'<div style="color:#64748b;font-size:0.79rem">{getattr(resp, "route_reason", "")}</div>'
    )
    # Step 3 body — richer for Challenger (shows tool call explainability)
    if resp.agent_used == "challenger" and getattr(resp, "challenge_meta", None):
        m = resp.challenge_meta
        v_color = "#22c55e" if m.get("validation_passed") else "#ef4444"
        v_label = "✓ validated" if m.get("validation_passed") else f"✗ {m.get('validation_reason','failed')[:60]}"
        step3 = (
            f'<span style="font-weight:600;font-size:0.88rem;color:#0f172a">'
            f'{agent_emoji} {agent_name}</span>'
            f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:0.4rem">({model_name})</span>'
            f'<div style="margin-top:0.4rem;padding:0.5rem 0.6rem;background:#f8fafc;'
            f'border:1px solid #e2e8f0;border-radius:7px;font-size:0.76rem">'
            f'<div style="color:#64748b;margin-bottom:3px">'
            f'<b>Tool called:</b> <code style="font-size:0.73rem">create_challenge</code></div>'
            f'<div style="color:#64748b;margin-bottom:3px">'
            f'<b>Constraint:</b> {m.get("constraint","—")}</div>'
            f'<div style="color:{v_color};font-weight:600">{v_label}</div>'
            f'</div>'
        )
    else:
        step3 = (
            f'<span style="font-weight:600;font-size:0.88rem;color:#0f172a">'
            f'{agent_emoji} {agent_name}</span>'
            f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:0.4rem">({model_name})</span>'
        )

    divider = '<div style="height:1px;background:#f1f5f9;margin:0.2rem 0"></div>'
    return (
        f'<div class="ic-card">'
        f'<div style="font-size:0.67rem;text-transform:uppercase;letter-spacing:0.09em;'
        f'color:#94a3b8;font-weight:700;margin-bottom:0.4rem">⚡ Decision Trace</div>'
        f'{_step("1", "Classify", step1)}{divider}'
        f'{_step("2", "Route",    step2)}{divider}'
        f'{_step("3", "Agent",    step3)}'
        f'</div>'
    )


def _audit_report_md(report) -> str:
    if not report.risks:
        return (
            '<div class="ic-card">'
            '<span style="color:#22c55e;font-weight:700">✓ On track</span>'
            '<span style="color:#64748b;font-size:0.85rem"> — no stuck concepts.</span>'
            '</div>'
        )
    lines = ['<div class="ic-card">']
    if report.summary:
        lines.append(
            f'<div style="color:#0f172a;font-size:0.9rem;font-weight:600;line-height:1.6;'
            f'border-left:3px solid #f59e0b;padding-left:0.7rem;margin-bottom:0.8rem">'
            f'{report.summary}</div>'
        )
    else:
        lines.append('<div style="font-weight:600;margin-bottom:0.8rem">⚠ Risk Report</div>')

    lines.append(
        '<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;'
        'color:#94a3b8;font-weight:700;margin-bottom:0.4rem">Gap Breakdown</div>'
    )
    for i, r in enumerate(report.risks):
        c_color = "#dc2626" if r.confirmed else "#d97706"
        c_label = "confirmed" if r.confirmed else "suspected"
        c_class = "risk-confirmed" if r.confirmed else "risk-suspected"
        lines.append(
            f'<div class="risk-item {c_class}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:600;font-size:0.86rem;color:#0f172a">{r.concept_id}</span>'
            f'<span style="background:{c_color};color:white;border-radius:4px;'
            f'padding:1px 7px;font-size:0.68rem;font-weight:600">{c_label}</span>'
            f'</div>'
            f'<div style="color:#64748b;font-size:0.79rem;margin-top:2px">'
            f'{r.downstream_reach} downstream concepts blocked</div>'
        )
        if r.consequence:
            lines.append(
                f'<div style="color:#475569;font-size:0.8rem;margin-top:0.35rem;'
                f'line-height:1.45;font-style:italic">{r.consequence}</div>'
            )
        else:
            lines.append(
                '<div style="color:#94a3b8;font-size:0.78rem;margin-top:0.3rem;font-style:italic">'
                'Click Full Audit for consequence analysis</div>'
            )
        lines.append('</div>')
        if i < len(report.risks) - 1:
            lines.append('<div style="height:1px;background:#f1f5f9;margin:0.3rem 0"></div>')

    lines.append('</div>')
    return "".join(lines)


def _audit_md(resp) -> str:
    return _audit_report_md(resp.audit_report)


# ── Event handlers ──────────────────────────────────────────────────────────────

def run_full_audit():
    report = orchestrator._auditor.audit_full("demo_learner")
    return _audit_report_md(report)


def submit(message, code, concept_id, history_raw, history_icap):
    if not message.strip() and not (code and code.strip()):
        return history_raw, history_icap, message, code, gr.update(), gr.update(), gr.update()

    user_content = message.strip()
    if code and code.strip():
        snippet      = f"\n```python\n{code.strip()}\n```"
        user_content = (user_content + snippet) if user_content else snippet.strip()

    # ── Raw AI (no scaffolding) ────────────────────────────────────────────
    raw_text = _raw_ai_response(message, code)
    history_raw = list(history_raw)
    history_raw.append({"role": "user",      "content": user_content})
    history_raw.append({"role": "assistant", "content": raw_text})

    # ── ICAP pipeline ─────────────────────────────────────────────────────
    resp = orchestrator.run(
        learner_id="demo_learner",
        concept_id=concept_id,
        learner_message=message,
        code=code.strip() if code and code.strip() else None,
    )
    history_icap = list(history_icap)
    history_icap.append({"role": "user",      "content": user_content})
    history_icap.append({"role": "assistant", "content": resp.response_text})

    # ── Code editor: pre-populate when Challenger fires ────────────────────
    if resp.starter_code:
        code_update      = gr.update(value=resp.starter_code)
        accordion_update = gr.update(open=True)
    else:
        code_update      = gr.update(value="")
        accordion_update = gr.update()

    return history_raw, history_icap, "", code_update, _icap_md(resp), _audit_md(resp), accordion_update


def clear_session():
    empty_icap  = '<div class="ic-card" style="color:#94a3b8;font-size:0.88rem">Submit your first message to begin.</div>'
    empty_audit = '<div class="ic-card" style="color:#94a3b8;font-size:0.88rem">No data yet.</div>'
    return [], [], "", "", empty_icap, empty_audit, gr.update(open=False)


# ── Layout ─────────────────────────────────────────────────────────────────────

_EMPTY_ICAP  = '<div class="ic-card" style="color:#94a3b8;font-size:0.88rem">Submit your first message to begin.</div>'
_EMPTY_AUDIT = '<div class="ic-card" style="color:#94a3b8;font-size:0.88rem">No data yet.</div>'

with gr.Blocks(title="ICAP Coding Tutor") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.8rem">
        <div>
          <div style="color:#ffffff;font-size:1.65rem;font-weight:800;letter-spacing:-0.025em;line-height:1.1">
            ⚡ ICAP Coding Tutor
          </div>
          <div style="color:#94a3b8;font-size:0.86rem;margin-top:0.35rem;font-weight:400;max-width:520px;line-height:1.5">
            Early warning for learning failure — catches gaps before they compound,
            predicts what future concepts are now at risk.
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <div style="display:flex;flex-direction:column;gap:0.35rem;align-items:flex-end">
            <div>
              <span style="background:#6366f1;color:white;border-radius:6px;padding:0.28rem 0.85rem;
                           font-size:0.76rem;font-weight:700">qwen-coder-plus</span>
              <span style="background:#334155;color:#94a3b8;border-radius:6px;padding:0.28rem 0.85rem;
                           font-size:0.76rem;font-weight:600;margin-left:0.35rem">qwen-plus</span>
            </div>
            <div style="color:#475569;font-size:0.72rem;font-weight:500">
              Qwen Global AI Hackathon 2026
            </div>
          </div>
        </div>
      </div>
    </div>
    """)

    # ── Main row: [Raw AI] | ICAP Tutor | Sidebar ─────────────────────────────
    # Raw AI panel is hidden by default — toggle with the Compare button.
    with gr.Row(equal_height=False):

        # ── Left: Raw AI (hidden by default) ──────────────────────────────
        with gr.Column(scale=4, min_width=280, visible=False) as raw_col:
            gr.HTML("""
            <div class="panel-header-raw">
              <span style="font-weight:700;color:#475569;font-size:0.9rem">🤖 Raw AI</span>
              <span style="color:#94a3b8;font-size:0.78rem;margin-left:0.5rem">qwen-plus · no scaffolding</span>
            </div>
            """)
            chatbot_raw = gr.Chatbot(
                height=420,
                show_label=False,
                elem_id="chatbox-raw",
                placeholder=(
                    "<div style='text-align:center;padding:2rem 1rem;color:#94a3b8'>"
                    "<div style='font-size:1.5rem;margin-bottom:0.4rem'>🤖</div>"
                    "<div style='font-size:0.85rem'>Answers directly — no awareness of your level</div>"
                    "</div>"
                ),
            )

        # ── Right: ICAP Tutor ──────────────────────────────────────────────
        with gr.Column(scale=4, min_width=280):
            gr.HTML("""
            <div class="panel-header-icap">
              <span style="font-weight:700;color:#6366f1;font-size:0.9rem">⚡ ICAP Tutor</span>
              <span style="color:#6366f1;font-size:0.78rem;margin-left:0.5rem;opacity:0.7">5-agent pipeline · level-aware</span>
            </div>
            """)
            chatbot_icap = gr.Chatbot(
                height=420,
                show_label=False,
                elem_id="chatbox-icap",
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=icap-tutor"),
                placeholder=(
                    "<div style='text-align:center;padding:2rem 1rem;color:#6366f1'>"
                    "<div style='font-size:1.5rem;margin-bottom:0.4rem'>⚡</div>"
                    "<div style='font-size:0.85rem'>Classifies your level, routes to the right agent, tracks risk</div>"
                    "</div>"
                ),
            )

        # ── Sidebar ────────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=220):
            gr.HTML('<div class="sidebar-label">Decision Trace</div>')
            icap_panel  = gr.HTML(_EMPTY_ICAP)

            gr.HTML('<div class="sidebar-label">Risk Monitor</div>')
            audit_panel = gr.HTML(_EMPTY_AUDIT)

            full_audit_btn = gr.Button("🔍  Full Audit", variant="secondary", size="sm", elem_id="audit-btn")

            gr.HTML("""
            <div style="margin-top:0.9rem;padding:0.75rem;background:#f8fafc;
                        border-radius:10px;border:1px solid #e2e8f0">
              <div style="font-size:0.67rem;text-transform:uppercase;letter-spacing:0.08em;
                          color:#94a3b8;font-weight:700;margin-bottom:0.45rem">ICAP Scale</div>
              <div style="font-size:0.79rem;color:#475569;line-height:2">
                🔴 <b>P</b>assive &nbsp;→&nbsp; 🟡 <b>A</b>ctive<br>
                🟢 <b>C</b>onstructive &nbsp;→&nbsp; 🔵 <b>I</b>nteractive<br>
                🟠 <b>Struggling</b> — intervention mode
              </div>
              <div style="margin-top:0.5rem;font-size:0.78rem;display:flex;gap:0.75rem">
                <span><span style="color:#dc2626;font-size:1rem">▌</span>
                  <span style="color:#64748b">confirmed</span></span>
                <span><span style="color:#d97706;font-size:1rem">▌</span>
                  <span style="color:#64748b">suspected</span></span>
              </div>
            </div>
            """)

    # ── Input section (full width) ────────────────────────────────────────────
    gr.HTML('<div style="margin-top:0.75rem"></div>')

    concept = gr.Radio(
        choices=ALL_CONCEPTS,
        value="recursion_basics",
        label="Topic",
        elem_id="concept-row",
    )

    msg = gr.Textbox(
        placeholder="What's confusing? Describe what you tried or ask a question...",
        lines=2,
        max_lines=6,
        show_label=False,
        elem_id="msg-box",
    )

    with gr.Accordion("📎  Paste code (optional)", open=False) as code_accordion:
        code = gr.Code(language="python", show_label=False, lines=7)

    with gr.Row():
        clear_btn      = gr.Button("🗑  Clear",              variant="secondary", scale=1, elem_id="clear-btn")
        compare_btn    = gr.Button("🔀 Compare with Raw AI", variant="secondary", scale=2, elem_id="compare-btn")
        submit_btn     = gr.Button("Send  ↵",                variant="primary",   scale=4, elem_id="submit-btn")

    compare_visible = gr.State(False)

    # ── Event wiring ──────────────────────────────────────────────────────────
    outputs = [chatbot_raw, chatbot_icap, msg, code, icap_panel, audit_panel, code_accordion]
    submit_btn.click(fn=submit,        inputs=[msg, code, concept, chatbot_raw, chatbot_icap], outputs=outputs)
    msg.submit(       fn=submit,        inputs=[msg, code, concept, chatbot_raw, chatbot_icap], outputs=outputs)
    clear_btn.click(  fn=clear_session, inputs=[],                                              outputs=outputs)
    full_audit_btn.click(fn=run_full_audit, inputs=[], outputs=[audit_panel])

    def toggle_compare(visible):
        new = not visible
        label = "✕ Hide Comparison" if new else "🔀 Compare with Raw AI"
        return gr.update(visible=new), new, gr.update(value=label)

    compare_btn.click(
        fn=toggle_compare,
        inputs=[compare_visible],
        outputs=[raw_col, compare_visible, compare_btn],
    )


if __name__ == "__main__":
    # server_name="0.0.0.0" is required for the app to be reachable
    # outside a container or VM — localhost only won't bind to the public interface.
    # PORT env var lets the platform override 7860 (Alibaba FC, ECS, ModelScope all set it).
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        theme=gr.themes.Soft(),
        css=CSS,
        show_error=True,
        server_name="0.0.0.0",
        server_port=port,
    )
