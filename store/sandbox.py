"""
executor/sandbox.py
Active-Learning Agent Society — code execution + signal disambiguation

THREE JOBS, IN ORDER:
  1. run_code()      — execute learner code against packet test cases, SAFELY.
  2. classify_signal — map the run result to an executor_signal string.
  3. disambiguate()  — when a signal is shared by >1 failure mode (e.g.
                       RecursionError from both missing_base_case and
                       no_progress), use AST structure to pick the right one.

WHY THE DISAMBIGUATOR EXISTS
  The concept packet maps executor_signal -> failure_mode -> provocateur_prompt.
  But RecursionError is produced by TWO structurally different bugs. Without
  disambiguation the Provocateur asks "when does it stop?" to a learner who has
  a base case but isn't progressing toward it — wrong question, broken-looking
  tutor. The run tells you THAT it failed; the AST tells you WHICH bug.

SECURITY NOTE
  A timeout is NOT a sandbox. This runs learner code in a subprocess with:
    - empty environment (cannot read DASHSCOPE_API_KEY or anything else)
    - a temp working dir it is started in
    - CPU + memory + file-size rlimits (blocks fork bombs / memory balloons)
    - a hard wall-clock timeout
  This is defensible for a hackathon demo with controlled input. It is NOT a
  true sandbox — a determined attacker can still escape. For a PUBLIC live URL
  running untrusted input, run this inside a container instead.
"""

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
import textwrap


# ---------------------------------------------------------------------------
# 1. EXECUTION
# ---------------------------------------------------------------------------
CPU_SECONDS = 2          # rlimit CPU time
MEM_BYTES = 256 * 1024 * 1024   # 256 MB address space cap
FSIZE_BYTES = 1024 * 1024       # 1 MB max file write
WALL_TIMEOUT = 5         # subprocess wall-clock kill


def _set_limits():
    """
    Called in the child via preexec_fn — applies hard resource caps.
    Each limit is set independently so a restricted container environment
    (e.g. no RLIMIT_AS due to cgroup v2) degrades gracefully rather than
    killing the whole process.
    """
    for limit, value in [
        (resource.RLIMIT_CPU,   (CPU_SECONDS, CPU_SECONDS)),
        (resource.RLIMIT_AS,    (MEM_BYTES,   MEM_BYTES)),
        (resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES)),
    ]:
        try:
            resource.setrlimit(limit, value)
        except (ValueError, resource.error):
            pass  # container doesn't allow this limit — wall-clock timeout still applies


def run_code(code: str, func_name: str, test_cases: list) -> dict:
    """
    Execute `code`, then call func_name against each test case.
    test_cases: [{"input": <arg or [args]>, "expected": <value>}, ...]

    Returns a structured result the signal classifier consumes:
      {
        "ran": bool,                # did it execute without crashing?
        "error_type": str|None,     # e.g. "RecursionError", "TypeError"
        "results": [                # per test case (only if ran)
            {"input":..., "expected":..., "got":..., "passed": bool}
        ],
        "all_passed": bool,
        "stderr": str
      }
    """
    harness = _build_harness(code, func_name, test_cases)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "submission.py")
        with open(tmpfile, "w") as fh:
            fh.write(harness)
        try:
            proc = subprocess.run(
                [sys.executable, tmpfile],
                cwd=tmpdir,                 # confined working dir
                env={"PATH": "/usr/bin:/bin"},  # stripped env — no API keys leak
                capture_output=True,
                text=True,
                timeout=WALL_TIMEOUT,
                preexec_fn=_set_limits,     # CPU/mem/fsize caps in child
            )
        except subprocess.TimeoutExpired:
            return {"ran": False, "error_type": "Timeout", "results": [],
                    "all_passed": False, "stderr": "wall-clock timeout"}

    return _parse_output(proc.stdout, proc.stderr)


def _build_harness(code: str, func_name: str, test_cases: list) -> str:
    """
    Build a runnable script: learner code at MODULE LEVEL (preserves its own
    indentation), then a separately-indented test runner. Defining the learner
    code inside a try block corrupts indentation, so we don't — a syntax error
    in the learner code surfaces as a normal import-time exception instead.
    """
    prelude = "import json, sys\nsys.setrecursionlimit(200)\n"
    runner = textwrap.dedent(f"""
        _RESULTS = []
        _ERR = None
        _tests = {json.dumps(test_cases)}
        for _t in _tests:
            _inp = _t["input"]
            try:
                _got = {func_name}(*_inp) if isinstance(_inp, list) else {func_name}(_inp)
                _RESULTS.append({{"input": _inp, "expected": _t["expected"],
                                  "got": _got, "passed": _got == _t["expected"]}})
            except RecursionError:
                _ERR = "RecursionError"; break
            except Exception as _e:
                _ERR = type(_e).__name__; break
        print(json.dumps({{"error_type": _ERR, "results": _RESULTS}}))
    """)
    # learner code sits flush at module level between prelude and runner
    return prelude + "\n" + code.strip() + "\n" + runner


def _parse_output(stdout: str, stderr: str) -> dict:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, IndexError):
        return {"ran": False, "error_type": "CrashedNoOutput", "results": [],
                "all_passed": False, "stderr": stderr[-500:]}
    results = payload.get("results", [])
    err = payload.get("error_type")
    all_passed = bool(results) and all(r["passed"] for r in results) and err is None
    return {"ran": err is None, "error_type": err, "results": results,
            "all_passed": all_passed, "stderr": stderr[-500:]}


# ---------------------------------------------------------------------------
# 2. SIGNAL CLASSIFICATION
# ---------------------------------------------------------------------------
def classify_signal(run_result: dict) -> str:
    """
    Map a run result to a coarse executor_signal. Some signals are AMBIGUOUS
    (one signal, multiple possible failure modes) and need disambiguation.
    """
    if run_result["error_type"] == "Timeout":
        return "Timeout"
    if run_result["error_type"] == "RecursionError":
        return "RecursionError"          # AMBIGUOUS -> needs disambiguate()
    if run_result["error_type"]:
        return f"Error:{run_result['error_type']}"
    if run_result["all_passed"]:
        return "all_passed"
    # ran clean but some test failed — which one matters (edge case vs general)
    failed = [r for r in run_result["results"] if not r["passed"]]
    if failed and all(_is_edge(r["input"]) for r in failed):
        return "passes_common_fails_edge"   # e.g. off-by-one base case
    return "wrong_answer"


def _is_edge(inp):
    """Edge inputs that distinguish off-by-one base cases (0, '', [])."""
    val = inp[0] if isinstance(inp, list) and inp else inp
    return val in (0, "", []) or val == []


# ---------------------------------------------------------------------------
# 3. STATIC DISAMBIGUATION  (the piece the build doc omitted)
# ---------------------------------------------------------------------------
def disambiguate(code: str, signal: str, func_name: str) -> str:
    """
    For shared signals, inspect code STRUCTURE to pick the precise failure mode.
    Currently handles the RecursionError fork:
        missing_base_case  — no return path that avoids the recursive call
        no_progress        — base case exists, but recursive call doesn't shrink arg
    Returns a failure_mode id the packet understands; falls back to the raw
    signal if it can't parse or the signal isn't ambiguous.
    """
    if signal != "RecursionError":
        return signal   # not ambiguous — passthrough

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "missing_base_case"   # unparseable + recursion ~ treat as no base

    func = _find_func(tree, func_name)
    if func is None:
        return signal

    has_base = _has_nonrecursive_return(func, func_name)
    progresses = _recursive_call_shrinks_arg(func, func_name)

    if not has_base:
        return "missing_base_case"
    if has_base and not progresses:
        return "no_progress"
    # base case present and arg reduces but still RecursionError — the base case
    # exists but doesn't cover the boundary input (e.g. n==1 misses n==0).
    return "wrong_base_case"


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _has_nonrecursive_return(func, fname):
    """True if there's a return statement that does NOT contain a call to fname."""
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            if not _contains_call(node.value, fname):
                return True
    return False


def _recursive_call_shrinks_arg(func, fname):
    """
    Heuristic: in a recursive call fname(<arg>), is at least one argument a
    strictly-reducing expression of a parameter (e.g. n-1, n//10, n[1:])?
    """
    params = {a.arg for a in func.args.args}
    for node in ast.walk(func):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == fname):
            for arg in node.args:
                if _is_reducing(arg, params):
                    return True
            return False   # found the recursive call, no reducing arg
    return False


def _is_reducing(node, params):
    # n - <something>
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.FloorDiv)):
        return _mentions_param(node.left, params)
    # n[1:] style slice
    if isinstance(node, ast.Subscript):
        return _mentions_param(node.value, params)
    return False


def _mentions_param(node, params):
    return any(isinstance(n, ast.Name) and n.id in params for n in ast.walk(node))


def _contains_call(node, fname):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == fname for n in ast.walk(node))


# ---------------------------------------------------------------------------
# convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def evaluate(code: str, func_name: str, test_cases: list) -> dict:
    run = run_code(code, func_name, test_cases)
    sig = classify_signal(run)
    mode = disambiguate(code, sig, func_name)
    return {"run": run, "signal": sig, "failure_mode": mode}


if __name__ == "__main__":
    print(__doc__.strip().splitlines()[0])