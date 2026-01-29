#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for vLLM OpenAI-compatible tool-calling.
#
# Usage:
#   chmod +x /root/run_tool_call_cases.sh
#   /root/run_tool_call_cases.sh
#
# Optional env:
#   VLLM_URL="http://localhost:8000/v1/chat/completions"
#   MODEL="openai/gpt-oss-120b"
#   CASE="tool_choice_required_batch"   # optional
#   N=100                                # optional (for batch cases)
#   SEED=1337                            # optional (for batch cases)
#   RUN_REQUIRED=1                       # default: 1 (run required batch after smoke tests)

URL="${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
RUN_REQUIRED="${RUN_REQUIRED:-1}"

extra_args=()
if [ -n "${CASE:-}" ]; then
  extra_args+=(--case "$CASE")
fi
if [ -n "${N:-}" ]; then
  extra_args+=(--n "$N")
fi
if [ -n "${SEED:-}" ]; then
  extra_args+=(--seed "$SEED")
fi

# If the user explicitly chose a case (or provided CLI args), run exactly once.
if [ -n "${CASE:-}" ] || [ "$#" -gt 0 ]; then
  python3 /root/run_tool_call_cases.py --url "$URL" --model "$MODEL" "${extra_args[@]}" "$@"
  exit 0
fi

# Default behavior:
# 1) Run lightweight single-request smoke tests.
python3 /root/run_tool_call_cases.py --url "$URL" --model "$MODEL"

# 2) Run the tool_choice="required" batch test (default: 100 runs).
if [ "$RUN_REQUIRED" = "1" ]; then
  python3 /root/run_tool_call_cases.py \
    --url "$URL" \
    --model "$MODEL" \
    --case tool_choice_required_batch \
    ${N:+--n "$N"} \
    ${SEED:+--seed "$SEED"}
fi
