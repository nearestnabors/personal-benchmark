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

**Every step below follows the same rhythm: _Run this_ first, _what you'll see_ second, _what it's doing_ third.**

## Repo layout

```
personal-benchmark/
├── config.yaml            # endpoint, threshold, judge, serving
├── models.yaml            # size ladder of tool-calling GGUFs (auto-downloads via -hf)
├── requirements.txt
├── data/
│   └── sample_dataset.jsonl   # 12-example SAMPLE golden set (objective + judge)
├── fixtures/              # fake harvest data across all 3 CLI sources
├── scripts/
│   ├── redact.py          # PII/secret redaction
│   ├── harvest.py         # Claude/ChatGPT/Codex usage -> normalized JSONL
│   ├── taxonomy.py        # rank your recurring task types
│   ├── make_dataset.py    # draft a golden dataset from your REAL prompts
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
2. **[Install llama.cpp](https://github.com/ggml-org/llama.cpp).** You need `llama-server` on your PATH (Homebrew: `brew install llama.cpp`, or build from source). This is the only inference engine used here.
3. **Run Phoenix locally**
   ```bash
   pip install arize-phoenix
   phoenix serve            # UI at http://localhost:6006
   ```

> **⚠️ Three preflight gotchas** that bite people mid-workshop:
>
> 1. **Async exports are slow.** The ChatGPT and Claude *app* data exports arrive by email and can take hours. Request them **before** you sit down. (The CLI history in Step 1 is instant, so you can also just use that.)
> 2. **Tool-calling is template-dependent.** Models must be served with `--jinja`. The serve script probes this and fails loudly.
> 3. **Context size matters.** Serve with `-c 8192` or larger — too small and downstream agents silently ignore their system prompt.

## Step 1 — Harvest your prompts

**Run this:**

```bash
# Workshop demo — committed fake data, no private history needed:
python scripts/harvest.py --fixtures --out data/harvested

# The real thing — auto-detects ~/.claude and ~/.codex; add app exports with
# --chatgpt-zip / --claude-zip:
python scripts/harvest.py --out data/harvested
```

**What you'll see:**

```text
Harvested 17 records (15 user) from 3 source(s):
  claude_history          12 records (12 user)
  claude_transcript        2 records (1 user)
  codex                    3 records (2 user)
Wrote data/harvested/all.jsonl
```

**What it's doing:** it reads your assistant history from each source and normalizes every message into one record. Each line of `data/harvested/all.jsonl` is an *output* like this:

```json
{"source": "claude_history", "timestamp": "2026-01-01T09:00:00+00:00",
 "role": "user", "text": "Summarize this thread into 3 bullets.",
 "session_id": "abc", "project": "/Users/you/inbox-tools"}
```

Sources it pulls from:

| Source | Where it lives | Notes |
| --- | --- | --- |
| **Claude Code (CLI)** | `~/.claude/history.jsonl` | Your raw prompts. Cleanest source — start here. |
| **Claude Code transcripts** | `~/.claude/projects/<proj>/*.jsonl` | Defensive fallback; format is internal and changes between versions. |
| **Codex (CLI)** | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl[.zst]` | First line is a `session_meta` block; newer sessions are `zstd`-compressed. |
| **ChatGPT app / web** | `conversations.json` inside the emailed export `.zip` | Settings → Data Controls → Export data. |
| **Claude app / web** | `conversations.json` inside the emailed export `.zip` | Settings → Export data. |

Two things it does before anything touches disk: it **redacts** secrets and PII (API keys, tokens, emails), and it **parses defensively** — the CLI transcript formats are internal and change between tool versions, so unknown line shapes are skipped rather than trusted.

## Step 2 — Find your recurring task types

**Run this:**

```bash
python scripts/taxonomy.py --in data/harvested/all.jsonl
```

**What you'll see:**

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

**What it's doing:** you can't recall your recurring tasks from memory — this ranks them by how often you actually make each kind of request, and writes `data/taxonomy.json`. It's human-in-the-loop: it only *prints and writes* the ranked list; **you** decide which task types are worth an eval.

The default `keyword` method is instant and fully offline, but it's **crude** — on real history it mislabels and dumps most prompts into `other`. Treat it as a rough first pass. For real grouping, recruit a **big model**:

```bash
# Recruit Claude/ChatGPT — no API key. Writes a copy-paste prompt (your redacted
# prompts + a strict JSON schema); paste it in, save the reply as data/taxonomy.json.
python scripts/taxonomy.py --in data/harvested/all.jsonl --method paste

# Or call an OpenAI-compatible endpoint directly (ONE call per prompt):
python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY
python scripts/taxonomy.py --in data/harvested/all.jsonl --method llm \
    --base-url https://api.anthropic.com/v1 --model claude-haiku-4-5-20251001 --api-key $ANTHROPIC_API_KEY
```

Claude works here because Anthropic exposes an [OpenAI-compatible endpoint](https://docs.anthropic.com/en/api/openai-sdk) — same client, just a different `--base-url`, `--model`, and key. Your prompts are redacted before they leave `harvest.py`, but `paste`/hosted `llm` do send them to a third-party model — that's your call to make.

## Step 3 — Build your golden dataset

**Run this:**

```bash
# Draft a dataset from your REAL prompts (no API key): writes a prompt to paste
# into Claude/ChatGPT. Save its JSON-Lines reply as data/dataset.jsonl.
python scripts/make_dataset.py --in data/harvested/all.jsonl --max 40

# ...review the draft, then upload it to Phoenix:
python scripts/build_dataset.py --path data/dataset.jsonl
```

**What you'll see:**

```text
Offering 40 of your real requests to the model.
Wrote data/dataset_prompt.txt
Next:
  1. Paste it into Claude or ChatGPT.
  2. Save the JSON-Lines reply as data/dataset.jsonl.
  3. REVIEW it — fix any wrong references, drop weak rows.
  4. python scripts/build_dataset.py --path data/dataset.jsonl
```

Then `build_dataset.py` uploads it (numbers illustrative):

```text
Uploaded dataset 'personal-benchmark-SAMPLE'  (id=RGF0YXNldDox, version=...)
  12 examples: 5 exact, 4 judge, 3 structured
```

**What it's doing:** each dataset row pairs one of *your real requests* (the `input`, kept verbatim — the model never invents tasks) with a way to score it:

| `scoring` | Use when… | Needs | Example task |
| --- | --- | --- | --- |
| `exact` | one exact string is correct | `reference` (string) | classify intent as `billing`/`technical`/`account` |
| `structured` | a specific JSON is correct | `reference` (object) | extract name/email/company to JSON |
| `judge` | quality is subjective | `rubric` (text) | rewrite this email to be concise but keep the deadline |

`make_dataset.py` feeds your harvested requests to a model that keeps each one verbatim as `input` and only adds the scoring + reference/rubric, skipping requests it can't grade on their own (ones that point at a paste or file). A row looks like:

```json
{"task_type": "summarize_thread", "scoring": "judge",
 "input": "Summarize this thread into exactly 3 bullet points for a standup. Thread: ...",
 "reference": null,
 "rubric": "Passing only if the summary is 3 bullets, captures the outage + rollback + fix, and invents no facts.",
 "label": "DRAFT"}
```

> **⚠️ Review before you trust it.** A drafted dataset is *not* golden until you read it. Models get "correct" answers wrong, and a single bad `reference` silently corrupts every experiment. Skim the draft, fix references, drop weak rows — *then* upload. Delegating the typing is fine; delegating the judgment is not. (In a pinch, the committed `data/sample_dataset.jsonl` is a ready-made 12-example `SAMPLE` set — just run `python scripts/build_dataset.py`.)

## Step 4 — Serve small local models with llama.cpp

**Run this:**

```bash
./scripts/serve.sh qwen2.5-1.5b     # a candidate from models.yaml
./scripts/serve.sh judge            # the local judge
```

**What you'll see:**

```text
>> Serving qwen2.5-1.5b via Hugging Face: Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
   (first run downloads several GB and caches it; later runs are instant/offline)
>> on 127.0.0.1:8102  (ctx=8192)
>> Waiting up to 900s for http://127.0.0.1:8102/v1/models (downloading if needed) ...
>> Probing a tool-call round-trip ...
>> OK: tool-calling works. qwen2.5-1.5b is ready at http://127.0.0.1:8102
```

**What it's doing:** `serve.sh` reads one model from `models.yaml` and launches it under the hood as:

```bash
llama-server -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M -c 8192 --jinja --host 127.0.0.1 --port 8102
```

`models.yaml` is a **size ladder** of small, tool-calling instruct models. Each entry has a local `gguf` path *and* an `hf` spec — if the file isn't present, llama.cpp auto-downloads it from Hugging Face and caches it, so you need no manual downloads. `--jinja` applies the model's tool template; `-c 8192` keeps context large enough. The startup probe waits for `/v1/models` and checks a real tool-call roundtrip, so you never benchmark a broken endpoint. Everything downstream talks to these through one thin, endpoint-agnostic client, so swapping in `llamafile` or LM Studio later only changes a host and port.

## Step 5 — Run experiments and find your SAGE

**Run this:**

```bash
python scripts/run_sage.py
cat reports/sage_report.md
```

**What you'll see** (SAMPLE — illustrative shape, not a real run):

```markdown
# SAGE report

| Model                  | Size (B) | Overall | extract | classify | summarize | Mean latency (s) | SAGE? |
| ---                    | ---      | ---     | ---     | ---      | ---       | ---              | ---   |
| Qwen2.5 0.5B Instruct  | 0.5      | 58%     | 40%     | 67%      | 50%       | 0.4              |       |
| Qwen2.5 1.5B Instruct  | 1.5      | 75%     | 80%     | 100%     | 50%       | 0.7              | ✅    |
| Qwen2.5 3B Instruct    | 3.0      | 83%     | 80%     | 100%     | 75%       | 1.3              |       |

**SAGE = Qwen2.5 1.5B Instruct (1.5B)** — the smallest model that cleared 70%.
```

**What it's doing:** for every served model, `run_sage.py` runs one Phoenix experiment over your dataset and traces it into Phoenix. The **task** runs the model on each input; the **evaluator** scores each example by its `scoring` kind:

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

experiment = run_experiment(dataset=dataset, task=task, evaluators=[good_enough])
```

The **judge is a local model too** (offline by default, no API key); if it isn't running, judge tasks are left *unscored* rather than guessed. You set the bar in `config.yaml` (`threshold: 0.7`), and the **SAGE is the smallest model that clears it**. If none does, the report says so plainly. Open the experiments in the Phoenix UI to compare them side by side and drill into any failed example.

## Appendix — Deploy your SAGE

**Run this:**

```bash
# point Goose at your llama-server, then drive it headless
goose run -t "Extract the name, email, and company from this signature into JSON: ..."
```

**What you'll need** — Goose pointed at the winning model's endpoint:

```yaml
# ~/.config/goose/config.yaml
GOOSE_PROVIDER: openai
OPENAI_HOST: http://127.0.0.1:8102
OPENAI_BASE_PATH: v1/chat/completions
GOOSE_MODEL: qwen2.5-1.5b
```

**What it's doing:** Goose speaks the OpenAI-compatible protocol, so your `llama-server` endpoint works directly — put the winning prompt in a `.goosehints` file and you've replaced a frontier model with your SAGE for that task. Prefer a *coding* harness? **Aider** points at the same endpoint with `OPENAI_API_BASE=http://127.0.0.1:8102/v1` and `--model openai/qwen2.5-1.5b`.

## Bring your own data

Everything above runs on the SAMPLE/fixture data. To make it yours: re-run **Step 1** without `--fixtures`, use **Step 2** to find your real task types, let **Step 3** draft a dataset from your real prompts (and review it), then re-run **Steps 4–5**. Your SAGE is the smallest model that's good enough for *your* work.

> **Stretch goal:** if no small model clears your bar, optimize the *prompt* against your dataset with [GEPA](https://github.com/gepa-ai/gepa) or `dspy.GEPA`, trace the optimization into Phoenix, and re-benchmark. The deliverable there is a better prompt, not a bigger model.

## Tests

```bash
pytest tests/ -q     # runs against fixtures only — no private data needed
```
