#!/usr/bin/env bash
# Stop a model started by serve.sh.  ./scripts/stop.sh qwen2.5-1.5b  (or: all)
set -euo pipefail
KEY="${1:?usage: stop.sh <model-name|all>}"

stop_one() {
  local pidfile="/tmp/llama-$1.pid"
  if [[ -f "$pidfile" ]]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "stopped $1" || echo "$1 not running"
    rm -f "$pidfile"
  else
    echo "no pidfile for $1"
  fi
}

if [[ "$KEY" == "all" ]]; then
  for f in /tmp/llama-*.pid; do [[ -e "$f" ]] || continue; stop_one "$(basename "$f" .pid | sed 's/^llama-//')"; done
else
  stop_one "$KEY"
fi
