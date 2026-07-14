#!/usr/bin/env bash
# Serve ONE model from models.yaml with llama.cpp's OpenAI-compatible server,
# then probe it so we fail loudly instead of benchmarking a dead endpoint.
#
#   ./scripts/serve.sh qwen2.5-1.5b        # a candidate from models.yaml
#   ./scripts/serve.sh judge               # the local judge model
#
# Requires: llama-server (llama.cpp) on PATH, yq or python for YAML, curl, jq.
# No Ollama. If you prefer llamafile, run the .llamafile with the same flags.
set -euo pipefail

MODEL_KEY="${1:?usage: serve.sh <model-name-from-models.yaml>}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$DIR/models.yaml"
CTX="${CTX:-8192}"
HOST="${HOST:-127.0.0.1}"

# Read one model's port + gguf from models.yaml (python is always available).
read_field() {
  python3 - "$REGISTRY" "$MODEL_KEY" "$1" <<'PY'
import sys, yaml
reg, key, field = sys.argv[1], sys.argv[2], sys.argv[3]
data = yaml.safe_load(open(reg))
entries = list(data.get("models", []))
if isinstance(data.get("judge"), dict):
    entries.append(data["judge"])
for m in entries:
    if m.get("name") == key:
        print(m.get(field, "")); break
else:
    sys.exit(f"model '{key}' not found in {reg}")
PY
}

PORT="$(read_field port)"
GGUF="$(eval echo "$(read_field gguf)")" # expand ~
HF="$(read_field hf)"

# Prefer a local .gguf; otherwise let llama.cpp download+cache from Hugging Face.
if [[ -f "$GGUF" ]]; then
  MODEL_ARGS=(-m "$GGUF")
  echo ">> Serving $MODEL_KEY from local file: $GGUF"
elif [[ -n "$HF" ]]; then
  MODEL_ARGS=(-hf "$HF")
  echo ">> Serving $MODEL_KEY via Hugging Face: $HF"
  echo "   (first run downloads several GB and caches it; later runs are instant/offline)"
else
  echo "ERROR: '$MODEL_KEY' has no local gguf ($GGUF) and no 'hf' spec in models.yaml." >&2
  exit 1
fi

echo ">> on $HOST:$PORT  (ctx=$CTX)"
# --jinja is REQUIRED so the model's own tool-calling template is applied.
llama-server "${MODEL_ARGS[@]}" -c "$CTX" --jinja --host "$HOST" --port "$PORT" \
  > "/tmp/llama-$MODEL_KEY.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "/tmp/llama-$MODEL_KEY.pid"

# --- Startup probe: /v1/models must come up, and a tool-call must round-trip ---
# WAIT is generous because a first-run Hugging Face download can take minutes.
BASE="http://$HOST:$PORT"
WAIT="${WAIT:-900}"
echo ">> Waiting up to ${WAIT}s for $BASE/v1/models (downloading if needed) ..."
for i in $(seq 1 "$WAIT"); do
  if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: llama-server died on startup. See /tmp/llama-$MODEL_KEY.log" >&2; exit 1
  fi
  if (( i % 15 == 0 )); then echo "   still waiting (${i}s) — see /tmp/llama-$MODEL_KEY.log"; fi
  sleep 1
done
[[ "${ready:-0}" == 1 ]] || { echo "ERROR: /v1/models never came up" >&2; exit 1; }

echo ">> Probing a tool-call round-trip ..."
TOOL_RESP="$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_KEY"'",
  "messages": [{"role":"user","content":"What is the weather in Dublin? Use the tool."}],
  "tools": [{"type":"function","function":{"name":"get_weather",
    "description":"Get weather for a city",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice": "auto"
}' || true)"

if echo "$TOOL_RESP" | jq -e '.choices[0].message.tool_calls[0].function.name' >/dev/null 2>&1; then
  echo ">> OK: tool-calling works. $MODEL_KEY is ready at $BASE"
else
  echo "WARNING: no tool_calls parsed for $MODEL_KEY. It may lack a tool template" >&2
  echo "         or need --jinja support in your llama.cpp build. Chat still works." >&2
fi
echo ">> pid $SERVER_PID (stop with: ./scripts/stop.sh $MODEL_KEY)"
