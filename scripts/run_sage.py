"""Find your SAGE: the Small And Good Enough model.

For every model in the registry that is currently being served, this:

  1. runs a Phoenix experiment over your golden dataset (traced into Phoenix),
  2. scores each example with the right evaluator for its `scoring` kind
     (exact / structured / local LLM-as-judge),
  3. records real pass rate per task type, aggregate score, and mean latency,
  4. selects the SMALLEST model whose aggregate clears the threshold — the SAGE,
  5. writes reports/sage_report.md.

Every number in the report comes from a real run. If no model clears the bar,
the report says so plainly. Nothing is fabricated.

    # 1. start Phoenix (pip install arize-phoenix && phoenix serve)
    # 2. build the dataset: python scripts/build_dataset.py
    # 3. serve some models: ./scripts/serve.sh qwen2.5-1.5b   (etc.)
    # 4. serve the judge:  ./scripts/serve.sh judge
    python scripts/run_sage.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

import yaml

DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIR / "scripts"))
from openai_client import LocalLLM, ModelSpec, load_registry  # noqa: E402


# --------------------------------------------------------------------------- #
# Scoring primitives (used by the evaluators)
# --------------------------------------------------------------------------- #
def _strip_json(text: str) -> str:
    """Pull a JSON object/array out of a model reply, tolerating code fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=0)
    return text[start:]


def score_exact(output: str, reference) -> float:
    return 1.0 if output.strip().lower() == str(reference).strip().lower() else 0.0


def score_structured(output: str, reference) -> float:
    """1.0 only if the parsed JSON matches the reference dict (numeric-tolerant)."""
    try:
        parsed = json.loads(_strip_json(output))
    except (json.JSONDecodeError, ValueError):
        return 0.0
    if not isinstance(reference, dict) or not isinstance(parsed, dict):
        return 1.0 if parsed == reference else 0.0
    for key, want in reference.items():
        got = parsed.get(key)
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(want) - float(got)) > 1e-6:
                return 0.0
        elif str(got).strip().lower() != str(want).strip().lower():
            return 0.0
    return 1.0


_JUDGE_TEMPLATE = """You are a strict grader. Decide if the RESPONSE satisfies the RUBRIC.
Answer with exactly one word on the first line: PASS or FAIL. Then one short reason.

TASK:
{task}

RESPONSE:
{response}

RUBRIC:
{rubric}
"""


def score_judge(judge: LocalLLM, task: str, output: str, rubric: str) -> tuple[float, str]:
    verdict = judge.chat(_JUDGE_TEMPLATE.format(task=task, response=output, rubric=rubric))
    passed = verdict.strip().upper().startswith("PASS")
    return (1.0 if passed else 0.0), verdict.strip()[:200]


# --------------------------------------------------------------------------- #
# Benchmark one model
# --------------------------------------------------------------------------- #
def benchmark_model(model: LocalLLM, spec: ModelSpec, dataset, judge: LocalLLM | None):
    """Run one Phoenix experiment for `spec` and return aggregated real metrics."""
    from phoenix.client.experiments import create_evaluator, run_experiment

    latencies: dict[str, float] = {}
    per_example: list[dict] = []
    lock = threading.Lock()

    def task(input, example):
        t0 = time.perf_counter()
        out = model.chat(input["task"])
        with lock:
            latencies[str(example.id)] = time.perf_counter() - t0
        return out

    @create_evaluator(kind="CODE", name="good-enough")
    def good_enough(output, expected, metadata, example):
        scoring = (metadata or {}).get("scoring")
        reference = (expected or {}).get("reference")
        if scoring == "exact":
            score = score_exact(output, reference)
        elif scoring == "structured":
            score = score_structured(output, reference)
        elif scoring == "judge":
            if judge is None:
                return None  # no judge served -> leave unscored, do not guess
            score, _reason = score_judge(
                judge,
                task=(example.input or {}).get("task", ""),
                output=output,
                rubric=(metadata or {}).get("rubric", ""),
            )
        else:
            return None
        with lock:
            per_example.append({"task_type": (metadata or {}).get("task_type"), "score": score})
        return score

    experiment = run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[good_enough],
        experiment_name=f"sage-{spec.name}",
        experiment_description=f"Personal-benchmark run for {spec.display} ({spec.params_b}B)",
    )

    scored = [r for r in per_example if r["score"] is not None]
    overall = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
    by_type: dict[str, list[float]] = {}
    for r in scored:
        by_type.setdefault(r["task_type"], []).append(r["score"])
    per_type = {k: sum(v) / len(v) for k, v in by_type.items()}
    mean_latency = sum(latencies.values()) / len(latencies) if latencies else 0.0
    return {
        "name": spec.name,
        "display": spec.display,
        "params_b": spec.params_b,
        "overall": overall,
        "per_type": per_type,
        "mean_latency_s": mean_latency,
        "n_scored": len(scored),
        "experiment_id": experiment.get("experiment_id") if isinstance(experiment, dict) else None,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(
    results: list[dict], threshold: float, sage: dict | None, endpoint: str, out: Path
):
    lines = ["# SAGE report", ""]
    lines.append(
        f"- Dataset scored live against a threshold of **{threshold:.0%}** overall pass rate."
    )
    lines.append(f"- Every number below is from a real run. Traces are in Phoenix at {endpoint}.")
    lines.append("")
    task_types = sorted({t for r in results for t in r["per_type"]})
    header = ["Model", "Size (B)", "Overall", *task_types, "Mean latency (s)", "SAGE?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in sorted(results, key=lambda x: x["params_b"]):
        row = [
            r["display"],
            f"{r['params_b']:g}",
            f"{r['overall']:.0%}",
            *[
                f"{r['per_type'].get(t, float('nan')):.0%}" if t in r["per_type"] else "—"
                for t in task_types
            ],
            f"{r['mean_latency_s']:.2f}",
            "✅" if sage and r["name"] == sage["name"] else "",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if sage:
        lines.append(
            f"**SAGE = {sage['display']} ({sage['params_b']:g}B)** — the smallest model "
            f"that cleared {threshold:.0%} (scored {sage['overall']:.0%})."
        )
    else:
        lines.append(
            f"**No model cleared {threshold:.0%}.** Raise the dataset's difficulty, "
            "lower the bar, or try the GEPA prompt-optimization stretch goal."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = yaml.safe_load((DIR / "config.yaml").read_text())
    endpoint = cfg["phoenix"]["endpoint"]
    threshold = float(cfg["threshold"])
    host = cfg["serving"]["host"]

    # Trace everything into Phoenix.
    from phoenix.otel import register

    register(
        endpoint=f"{endpoint}/v1/traces",
        project_name=cfg["phoenix"]["project_name"],
        auto_instrument=True,  # instruments the OpenAI client used by LocalLLM
    )

    from phoenix.client import Client

    client = Client(base_url=endpoint)
    dataset = client.datasets.get_dataset(dataset=cfg["dataset"]["name"])

    # Judge (optional but needed for subjective tasks).
    judge = None
    jcfg = cfg.get("judge", {})
    if jcfg:
        jllm = LocalLLM(
            base_url=jcfg["base_url"],
            api_key=jcfg.get("api_key", "x"),
            model=jcfg.get("model", "judge"),
        )
        if jllm.is_up():
            judge = jllm
            print(f"Judge ready at {jcfg['base_url']}")
        else:
            print(f"WARNING: judge at {jcfg['base_url']} not up — judge tasks left unscored.")

    specs = load_registry(DIR / cfg["serving"]["registry"], host=host)
    results = []
    for spec in specs:
        llm = LocalLLM(base_url=spec.base_url, model=spec.name)
        if not llm.is_up():
            print(f"skip {spec.name}: not served at {spec.base_url} (start it with serve.sh)")
            continue
        print(f"benchmarking {spec.display} ({spec.params_b}B) ...")
        results.append(benchmark_model(llm, spec, dataset, judge))

    if not results:
        print("No served models found. Start at least one with ./scripts/serve.sh <name>.")
        return

    candidates = [r for r in results if r["overall"] >= threshold]
    sage = min(candidates, key=lambda r: r["params_b"]) if candidates else None

    out = DIR / "reports" / "sage_report.md"
    write_report(results, threshold, sage, endpoint, out)
    print(f"\nWrote {out}")
    if sage:
        print(f"SAGE = {sage['display']} ({sage['params_b']:g}B), scored {sage['overall']:.0%}")
    else:
        print(f"No model cleared {threshold:.0%}.")


if __name__ == "__main__":
    main()
