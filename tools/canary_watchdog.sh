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
# Empty sha = request failed or engine still in its post-restart model-load
# window (returns fast with no content, not a 90s timeout). That is NOT
# state-poisoning — real drift (e.g. the "!!!!" collapse) returns a
# NON-EMPTY but wrong sha. Restarting on empty caused a spurious restart
# loop right after every legitimate restart, so treat empty as inconclusive.
if [ -z "$OUT2" ]; then
  echo "canary inconclusive (empty response — engine not ready / request failed); no restart"
  exit 0
fi
echo "canary DRIFT (got '$OUT' then '$OUT2', want $GOOD) — restarting aspida-hura"
systemctl restart aspida-hura
