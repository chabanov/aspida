#!/bin/bash
# Hunt v3: poison + full autopsy. Watchdog MUST be stopped beforehand.
cd /Users/ceo/Developer/agents
for cyc in 1 2 3 4 5 6 7 8 9 10; do
  PID0=$(ssh root@107.170.40.54 'systemctl show -p MainPID --value aspida-hura')
  echo "=== cycle $cyc (pid $PID0) ==="
  deno run --allow-net --allow-read --allow-write scripts/eval-hura.ts --endpoint http://127.0.0.1:18099/v1 --model hura:latest --concurrency 4 --out /tmp/eval-h3-$cyc.json 2>&1 | tail -1 >/dev/null
  STORM=$(ssh root@107.170.40.54 'python3 /root/rst_repro.py 12 2>&1 | tail -1'); echo "$STORM"
  CAN=$(ssh root@107.170.40.54 'curl -s -m 90 http://127.0.0.1:8099/v1/chat/completions -H Content-Type:application/json -d "{\"model\":\"hura:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the ocean.\"}],\"max_tokens\":48,\"temperature\":0}" | python3 -c "import json,sys,hashlib;c=json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"];print(hashlib.sha256(c.encode()).hexdigest()[:16])" 2>/dev/null || echo dead')
  PID1=$(ssh root@107.170.40.54 'systemctl show -p MainPID --value aspida-hura')
  echo "cycle $cyc: canary=$CAN pid=$PID0->$PID1"
  if [ "$PID1" != "$PID0" ]; then
    echo "(hard crash — no forensic state; continuing hunt on fresh pid)"
    continue
  fi
  if { [ "$CAN" != "8752e132c193abbe" ] && [ "$CAN" != "dead" ]; } || echo "$STORM" | grep -q POISONED-PERSISTENT; then
    echo "=== POISONED ALIVE at cycle $cyc — AUTOPSY ==="
    ssh root@107.170.40.54 '
      TS=$(date "+%Y-%m-%d %H:%M:%S")
      touch /tmp/aspida_phb
      curl -s -m 90 http://127.0.0.1:8099/v1/chat/completions -H Content-Type:application/json -d "{\"model\":\"hura:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the ocean.\"}],\"max_tokens\":1,\"temperature\":0}" -o /dev/null
      rm -f /tmp/aspida_phb
      journalctl -u aspida-hura --no-pager --since "$TS" | grep PHB | sed "s/.*\[PHB\]/[PHB]/" > /root/phb_poisoned.txt
      echo "--- per-layer diff (healthy vs poisoned, >2% flagged) ---"
      paste /root/phb_healthy.txt /root/phb_poisoned.txt | awk "{split(\$5,a,\"=\"); split(\$10,b,\"=\"); r=a[2]; g=b[2]; d=(g-r); ad=(d<0?-d:d); rel=(r>0?ad/r*100:0); m=(rel>2?\"  <== DIVERGES\":\"\"); printf \"li=%s h=%s p=%s %.1f%%%s\n\", substr(\$2,4), r, g, rel, m}" | head -42
      echo "--- forensics (GCHK/HSUM/WSUM in poisoned state) ---"
      for i in $(seq 1 21); do curl -s -m 30 http://127.0.0.1:8099/v1/chat/completions -H Content-Type:application/json -d "{\"model\":\"hura:latest\",\"messages\":[{\"role\":\"user\",\"content\":\"hi $i\"}],\"max_tokens\":2}" -o /dev/null; done
      journalctl -u aspida-hura --no-pager --since "$TS" | grep -E "GCHK|HSUM|WSUM" | tail -4 | sed "s/.*secure_server.[0-9]*.: //"'
    echo "AUTOPSY-DONE"
    exit 0
  fi
done
echo "HUNT-EXHAUSTED"
