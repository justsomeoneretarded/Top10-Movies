#!/usr/bin/env bash
set -euo pipefail
# Ollama is a local inference server. No account or paid API is used.
if ! command -v ollama >/dev/null; then
  curl --fail --show-error --location --retry 3 https://ollama.com/install.sh -o "$RUNNER_TEMP/ollama-install.sh"
  sh "$RUNNER_TEMP/ollama-install.sh"
fi
ollama serve > "$RUNNER_TEMP/ollama.log" 2>&1 &
for attempt in $(seq 1 60); do
  if curl --fail --silent "${OLLAMA_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
    ollama pull "${SCRIPT_MODEL:-qwen3:1.7b}"
    exit 0
  fi
  sleep 1
done
echo "Ollama did not start" >&2
exit 1
