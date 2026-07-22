#!/bin/bash
# aspida canary watchdog: greedy determinism probe. If the engine's greedy
# output for the reference prompt drifts from the known-good sha (silent
# state-poisoning, see aspida commit 9dbc129 investigation) or the request
# fails outright, restart the engine. Logs to journald (systemd unit).
set -u
GOOD="8752e132c193abbe"
OUT=$(curl -s -m 90 http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hura:latest","messages":[{"role":"user","content":"Write one sentence about the ocean."}],"max_tokens":48,"temperature":0}' \
  | python3 -c 'import json,sys,hashlib;c=json.load(sys.stdin)["choices"][0]["message"]["content"];print(hashlib.sha256(c.encode()).hexdigest()[:16])' 2>/dev/null)
if [ "$OUT" = "$GOOD" ]; then
  echo "canary ok ($OUT)"
  exit 0
fi
# One retry: a busy engine can time out the probe without being poisoned.
sleep 20
OUT2=$(curl -s -m 90 http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hura:latest","messages":[{"role":"user","content":"Write one sentence about the ocean."}],"max_tokens":48,"temperature":0}' \
  | python3 -c 'import json,sys,hashlib;c=json.load(sys.stdin)["choices"][0]["message"]["content"];print(hashlib.sha256(c.encode()).hexdigest()[:16])' 2>/dev/null)
if [ "$OUT2" = "$GOOD" ]; then
  echo "canary ok on retry ($OUT2)"
  exit 0
fi
# Empty sha = failed request. Two causes with OPPOSITE correct responses:
#   1. Cold start: engine is mid model-load (~30s), returns a fast empty/503,
#      NOT a 90s timeout. Recovers on its own — restarting just loops.
#   2. Wedge: engine is up but its CUDA/inference state is dead (e.g. the
#      dual-backend illegal-access hang) — empty forever, MUST restart.
# Distinguish by persistence: wait past the load window and probe once more.
# Real greedy DRIFT (the "!!!!" collapse) returns a NON-EMPTY wrong sha and
# is handled above/below without this delay.
if [ -z "$OUT2" ]; then
  sleep 45
  OUT3=$(curl -s -m 90 http://127.0.0.1:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"hura:latest","messages":[{"role":"user","content":"Write one sentence about the ocean."}],"max_tokens":48,"temperature":0}' \
    | python3 -c 'import json,sys,hashlib;c=json.load(sys.stdin)["choices"][0]["message"]["content"];print(hashlib.sha256(c.encode()).hexdigest()[:16])' 2>/dev/null)
  if [ "$OUT3" = "$GOOD" ]; then
    echo "canary ok after cold-start settle ($OUT3)"
    exit 0
  fi
  if [ -z "$OUT3" ]; then
    echo "canary WEDGE (empty for >45s past load window) — restarting aspida-hura"
    systemctl restart aspida-hura
    exit 1
  fi
  # non-empty but wrong → fall through to drift restart below
  OUT2="$OUT3"
fi
echo "canary DRIFT (got '$OUT' then '$OUT2', want $GOOD) — restarting aspida-hura"
systemctl restart aspida-hura
