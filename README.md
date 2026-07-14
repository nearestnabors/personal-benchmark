# Build Your Own Benchmark: Find Your SAGE

> Turn your real Claude and ChatGPT prompts into a personal eval set, then use [Phoenix](https://github.com/Arize-ai/phoenix) experiments to find your **SAGE** — the Smallest model that's Good Enough for the work you actually do.

Public benchmarks measure a model against *everyone's* work. This repo builds a benchmark that measures models against **your** work — the tasks you actually hand to an assistant every day — and then finds your **SAGE**:

> **SAGE — the Smallest model that's Good Enough.** The smallest, cheapest local model that clears a bar *you* set, on tasks *you* care about.

Why bother? Because the frontier model you reach for by reflex is often overkill for "extract these fields to JSON" or "summarize this thread." A 1.5B model running locally on your laptop may clear your bar for those — for free, offline, and privately. The only way to know is to measure.

You will:

1. Extract a golden dataset from the agent you're already using.
2. Use Phoenix to compare different models (small to large) and find one that is SAGE (small and good enough).
3. Add said model to Goose — boom! A local inference replacement, or keep using big inference but on a smaller/cheaper model.

> **Note:** Everything here runs **offline** on a committed `SAMPLE` dataset, so you can follow along with zero exports and no internet. Local inference is **llama.cpp** (`llama-server`) — no Ollama. And no number in any report is invented: every score comes from a real run, or is labeled `SAMPLE`.

## Repo layout

```
personal-benchmark/
├── config.yaml            # endpoint, threshold, judge, serving
├── models.yaml            # size ladder of tool-calling GGUFs (fill in real paths)
├── requirements.txt
├── data/
│   └── sample_dataset.jsonl   # 12-example SAMPLE golden set (objective + judge)
├── fixtures/              # fake harvest data across all 3 CLI sources
├── scripts/
│   ├── redact.py          # PII/secret redaction
│   ├── harvest.py         # Claude/ChatGPT/Codex usage -> normalized JSONL
│   ├── taxonomy.py        # rank your recurring task types
│   ├── build_dataset.py   # upload the golden set to Phoenix
│   ├── openai_client.py   # thin OpenAI-compatible client (endpoint-agnostic)
│   ├── serve.sh / stop.sh # launch llama-server + startup probe
│   └── run_sage.py        # experiments across models -> pick the SAGE
└── tests/
    └── test_harvest.py    # runs against fixtures only
```

## Before you start

1. **Get the code**
   ```bash
   git clone https://github.com/nearestnabors/personal-benchmark
   cd personal-benchmark
   uv venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
2. **[Install llama.cpp](https://llama-cpp.com/).** You need `llama-server` on your PATH (Homebrew: `brew install llama.cpp`, or build from source). This is the only inference engine used here.
3. **Run Phoenix locally**
   ```bash
   pip install arize-phoenix
   phoenix serve            # UI at http://localhost:6006
   ```

> **⚠️ Three preflight gotchas** that bite people mid-workshop:
>
> 1. **Async exports are slow.** The ChatGPT and Claude *app* data exports arrive by email and can take hours. If you want to use your app history, request it **before** you sit down. (The CLI history in Step 1 is instant, so you can also just use that.)
> 2. **Tool-calling is template-dependent.** Models must be served with `--jinja` so their own tool template is applied. The serve script probes this and fails loudly.
> 3. **Context size matters.** Serve with `-c 8192` or larger — too small and downstream agents silently ignore their system prompt.

## Step 1 — Harvest your prompts

Your best source of "what do I actually ask for" is your own history. The `harvest.py` script reads each source and normalizes every message to one record shape:

```json
{"source": "claude_history", "timestamp": "2026-01-01T09:00:00+00:00",
 "role": "user", "text": "Summarize this thread into 3 bullets.",
 "session_id": "abc", "project": "/Users/you/inbox-tools"}
```

| Source | Where it lives | Notes |
| --- | --- | --- |
| **Claude Code (CLI)** | `~/.claude/history.jsonl` | Your raw prompts. Cleanest source — start here. |
| **Claude Code transcripts** | `~/.claude/projects/<proj>/*.jsonl` | Defensive fallback; format is internal and changes between versions. |
| **Codex (CLI)** | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl[.zst]` | First line is a `session_meta` block; newer sessions are `zstd`-compressed. |
| **ChatGPT app / web** | `conversations.json` inside the emailed export `.zip` | Settings → Data Controls → Export data. |
| **Claude app / web** | `conversations.json` inside the emailed export `.zip` | Settings → Export data. |

> **Note:** The CLI transcript formats are **internal and change between tool versions** — this is documented by both tools. `harvest.py` therefore parses *defensively*: unknown line shapes are skipped, never assumed. For Claude, it prefers `history.jsonl` (your raw prompts) over the noisier project transcripts.

### Redaction happens first

Before any text is written to disk, it passes through a redaction pass so API keys, tokens, emails, and other PII never land in your dataset in the clear:

```python
# scripts/redact.py (excerpt)
_RULES = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_OPENAI_KEY]"),
    ("email",      re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    # ... bearer tokens, JWTs, AWS/GCP/GitHub keys, credit cards, SSNs, assigned secrets
]
```

### Run it

The repo ships **fixtures** (fake data across all three CLI sources) so you can run everything with no private data:

```bash
python scripts/harvest.py --fixtures --out data/harvested
```

```text
Harvested 17 records (15 user) from 3 source(s):
  claude_history          12 records (12 user)
  claude_transcript        2 records (1 user)
  codex                    3 records (2 user)
Wrote data/harvested/all.jsonl
```

When you're ready for the real thing, drop `--fixtures` and it auto-detects your local sources; add `--chatgpt-zip ~/Downloads/export.zip` to fold in an app export.

## Step 2 — Find your recurring task types

You can't recall your recurring tasks from memory — that's the whole point of harvesting. `taxonomy.py` groups your prompts into task types and ranks them by how often you actually do them:

```bash
python scripts/taxonomy.py --in data/harvested/all.jsonl
```

```text
Ranked task types from 15 prompts (keyword):

   1. extract_fields      5×
        - Extract the name, email, and company into JSON: 'Jordan Lee, [REDACTED_EMAIL], Acme Corp'
   2. triage_error        4×
        - My deploy failed with this log, triage it and tell me if it's my code or infra.
   3. summarize           3×
        - Summarize this Slack thread into 3 bullet points for my standup.
   4. rewrite             2×
        - Rewrite this email to be more concise and friendly but keep the deadline.
```

The default `keyword` method is instant and fully offline, but it's **crude** — on real history it mislabels and dumps most prompts into `other`. Treat it as a rough first pass you curate by hand, not ground truth.

Grouping messy real usage is a job for a **big model**. Two better options:

```bash
# Recruit Claude/ChatGPT — no API key. Writes a copy-paste-ready prompt
# (your already-redacted prompts + a strict JSON schema). Paste it into
# Claude or ChatGPT, then save its JSON reply as data/taxonomy.json.
python scripts/taxonomy.py --in data/harvested/all.jsonl --method paste

# Or label each prompt via an OpenAI-compatible endpoint. This makes ONE call
# per prompt, so for a hosted big model prefer `paste` above (one call, no key).
python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \
    --base-url http://127.0.0.1:8102/v1                                   # local llama-server

python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY

python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \
    --base-url https://api.anthropic.com/v1 --model claude-haiku-4-5-20251001 --api-key $ANTHROPIC_API_KEY
```

Claude works here because Anthropic exposes an [OpenAI-compatible endpoint](https://docs.anthropic.com/en/api/openai-sdk) — same client, just a different `--base-url`, `--model`, and key. (`--api-key` also falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in your environment.)

Your prompts are redacted before they ever leave `harvest.py`, but `paste`/hosted `llm` do send them to a third-party model — that's your call to make.

> **Note:** This step is human-in-the-loop by design. It only *prints* the ranked list and writes `data/taxonomy.json`. **You** decide which task types are worth an eval — the script never auto-decides for you.

## Step 3 — Build your golden dataset

Now turn your top task types into eval examples. Each example needs an `input`, a `scoring` method, and either a `reference` (the correct answer) or a `rubric` (for subjective tasks).

**Be honest about "golden."** Only tasks with a single correct answer get an exact/structured reference. Subjective tasks get an LLM-judge rubric and are labeled as such — don't dress a subjective task up as objective.

| `scoring` | Use when… | Needs | Example task |
| --- | --- | --- | --- |
| `exact` | one exact string is correct | `reference` (string) | classify intent as `billing`/`technical`/`account` |
| `structured` | a specific JSON is correct | `reference` (object) | extract name/email/company to JSON |
| `judge` | quality is subjective | `rubric` (text) | rewrite this email to be concise but keep the deadline |

The committed SAMPLE set (`data/sample_dataset.jsonl`, 12 examples) spans all three. One line looks like:

```json
{"task_type": "summarize_thread", "scoring": "judge",
 "input": "Summarize this thread into exactly 3 bullet points for a standup. Thread: ...",
 "reference": null,
 "rubric": "Passing only if the summary is 3 bullets, captures the outage + rollback + fix, and invents no facts.",
 "label": "SAMPLE"}
```

Upload it to Phoenix as a versioned dataset with the current client API:

```python
# scripts/build_dataset.py (core)
from phoenix.client import Client

client = Client(base_url="http://localhost:6006")

dataset = client.datasets.create_dataset(
    name="personal-benchmark-SAMPLE",
    inputs=[{"task": r["input"]} for r in rows],
    outputs=[{"reference": r.get("reference")} for r in rows],
    metadata=[
        {"task_type": r["task_type"], "scoring": r["scoring"],
         "rubric": r.get("rubric"), "label": r.get("label")}
        for r in rows
    ],
)
```

```bash
python scripts/build_dataset.py
```

```text
Uploaded dataset 'personal-benchmark-SAMPLE'  (id=RGF0YXNldDox, version=...)
  12 examples: 5 exact, 4 judge, 3 structured
```

## Step 4 — Serve small local models with llama.cpp

The `models.yaml` registry is a **size ladder** of small, tool-calling-capable instruct models. Fill in real local `.gguf` paths — nothing is downloaded for you:

```yaml
models:
  - {name: qwen2.5-0.5b, display: "Qwen2.5 0.5B Instruct", params_b: 0.5, port: 8101, gguf: "~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"}
  - {name: qwen2.5-1.5b, display: "Qwen2.5 1.5B Instruct", params_b: 1.5, port: 8102, gguf: "~/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"}
  - {name: llama3.2-3b,  display: "Llama 3.2 3B Instruct",  params_b: 3.0, port: 8103, gguf: "~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"}
```

Serve one model at a time. Under the hood this is just:

```bash
llama-server -m <model>.gguf -c 8192 --jinja --host 127.0.0.1 --port 8102
```

`--jinja` applies the model's tool-calling template; `-c 8192` keeps the context large enough. The `serve.sh` wrapper adds a **startup probe** that waits for `/v1/models` and then checks a real tool-call roundtrip, so you never benchmark a broken endpoint:

```bash
./scripts/serve.sh qwen2.5-1.5b
```

```text
>> Serving qwen2.5-1.5b on 127.0.0.1:8102  (ctx=8192)
>> Waiting for http://127.0.0.1:8102/v1/models ...
>> Probing a tool-call round-trip ...
>> OK: tool-calling works. qwen2.5-1.5b is ready at http://127.0.0.1:8102
```

The rest of the pipeline talks to these through one thin, endpoint-agnostic client (`openai_client.py`), so swapping in `llamafile` or LM Studio later only changes a host and port.

## Step 5 — Run experiments and find your SAGE

This is where Phoenix does the heavy lifting. `run_sage.py` runs one **experiment per served model** over your dataset, scoring each example with the right evaluator for its kind, and traces everything into Phoenix.

The **task** just runs the model on each input; the **evaluator** dispatches on the example's `scoring` metadata:

```python
# scripts/run_sage.py (core)
from phoenix.client.experiments import create_evaluator, run_experiment

def task(input, example):
    return model.chat(input["task"])          # your local model

@create_evaluator(kind="CODE", name="good-enough")
def good_enough(output, expected, metadata, example):
    scoring   = metadata["scoring"]
    reference = (expected or {}).get("reference")
    if scoring == "exact":
        return score_exact(output, reference)          # normalized string match
    if scoring == "structured":
        return score_structured(output, reference)     # parse JSON, compare fields
    if scoring == "judge":
        return score_judge(judge, task=..., output=output, rubric=metadata["rubric"])
    return None

experiment = run_experiment(
    dataset=dataset, task=task, evaluators=[good_enough],
    experiment_name=f"sage-{spec.name}",
)
```

The **judge is a local model too** (configured in `config.yaml`) — offline by default, no API key. A hosted judge is an opt-in fallback, never the default. If the judge isn't running, judge tasks are left *unscored* rather than guessed.

Set your bar in `config.yaml`:

```yaml
threshold: 0.7   # a model is a SAGE candidate only if overall pass rate >= this
```

Then run it and read the report:

```bash
python scripts/run_sage.py
cat reports/sage_report.md
```

The report ranks models by size and marks the SAGE — **the smallest model that clears the bar**:

> **Note:** The table below shows the **shape** of the output with `SAMPLE` placeholder numbers. Real numbers only appear after you run it against served models on your machine. If no model clears the bar, the report says so plainly.

```markdown
# SAGE report  (SAMPLE — illustrative shape, not a real run)

| Model                  | Size (B) | Overall | extract | classify | summarize | Mean latency (s) | SAGE? |
| ---                    | ---      | ---     | ---     | ---      | ---       | ---              | ---   |
| Qwen2.5 0.5B Instruct  | 0.5      | 58%     | 40%     | 67%      | 50%       | 0.4              |       |
| Qwen2.5 1.5B Instruct  | 1.5      | 75%     | 80%     | 100%     | 50%       | 0.7              | ✅    |
| Llama 3.2 3B Instruct  | 3.0      | 83%     | 80%     | 100%     | 75%       | 1.3              |       |

**SAGE = Qwen2.5 1.5B Instruct (1.5B)** — the smallest model that cleared 70%.
```

Open the experiments in the Phoenix UI to compare them side by side, drill into any failed example, and see the exact model output and judge explanation.

## Appendix — Deploy your SAGE

Once you've found your SAGE, wire that `(model, prompt)` into a real agent. **Goose** points at any OpenAI-compatible endpoint, so your `llama-server` works directly:

```yaml
# ~/.config/goose/config.yaml
GOOSE_PROVIDER: openai
OPENAI_HOST: http://127.0.0.1:8102
OPENAI_BASE_PATH: v1/chat/completions
GOOSE_MODEL: qwen2.5-1.5b
```

Put the winning prompt in a `.goosehints` file, then drive it headless:

```bash
goose run -t "Extract the name, email, and company from this signature into JSON: ..."
```

Prefer a *coding*-focused harness? **Aider** is the tightest scriptable loop — point it at the same endpoint with `OPENAI_API_BASE=http://127.0.0.1:8102/v1` and `--model openai/qwen2.5-1.5b`. Goose is the more general local-first agent; Aider/OpenCode are better when you specifically want a coding agent. Verify flags against each tool's current docs.

## Bring your own data

Everything above ran on the SAMPLE set. To make it *yours*:

1. Re-run **Step 1** without `--fixtures` to harvest your real history (request app exports ahead of time).
2. Use **Step 2** to find your true recurring tasks.
3. Curate them into a golden set in `data/` — objective answers where you can, honest judge rubrics where you can't.
4. Re-run **Steps 3–5**. Your SAGE is the smallest model that's good enough for *your* work.

> **Stretch goal:** if no small model clears your bar, don't give up on it yet — optimize the *prompt* against your dataset with [GEPA](https://github.com/gepa-ai/gepa) or `dspy.GEPA`, trace the optimization into Phoenix, and re-benchmark. The deliverable there is a better prompt, not a bigger model.

## Tests

```bash
pytest tests/ -q     # runs against fixtures only — no private data needed
```
