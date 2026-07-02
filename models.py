"""
Qwen model routing — different subtasks get the right model.

Classifier and Tutor handle Python code directly:
  → qwen-coder-plus: code-specialized reasoning, better at understanding
    syntax, semantics, and what a learner's code actually does vs. intends.

Provocateur and Auditor work in natural language:
  → qwen-plus: general-purpose, strong instruction-following for
    questioning strategies and consequence narration.

The deterministic router between agents uses zero LLM calls — all
routing logic lives in router.py as plain Python conditionals.
"""

QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

QWEN_CODE    = "qwen-coder-plus"   # classifier, tutor — code-domain tasks
QWEN_GENERAL = "qwen-plus"         # provocateur, auditor — language tasks
