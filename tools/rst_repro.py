#!/usr/bin/env python3
# RST-abort reproducer: parallel streamed generations killed with RST
# (SO_LINGER=0) after a short random read window + tiny SO_RCVBUF so the
# server-side send can also block. Loops until the engine pid changes.
import socket, json, time, threading, random, subprocess, sys, struct

HOST, PORT = "127.0.0.1", 8099
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def pid():
    return subprocess.run(["systemctl","show","-p","MainPID","--value","aspida-hura"],
                          capture_output=True,text=True).stdout.strip()

def aborted_stream(i, mode):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s.settimeout(15)
        s.connect((HOST, PORT))
        body = json.dumps({"model":"hura:latest",
            "messages":[{"role":"user","content":f"[{i}] Розкажи дуже детально про історію України та її регіонів."}],
            "max_tokens":400,"stream":True})
        s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
        deadline = time.time() + random.uniform(0.8, 2.5)
        try:
            while time.time() < deadline:
                s.recv(512)
        except socket.timeout:
            pass
        if mode == "rst":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()   # rst mode -> TCP RST; fin mode -> clean FIN
    except Exception:
        pass

def healthy():
    # greedy canary: sha drift = silent poisoning -> stop the storm IMMEDIATELY
    # so the state survives for the autopsy (a continued storm escalates to a
    # hard crash that destroys it).
    try:
        s = socket.create_connection((HOST, PORT), timeout=90)
        body = json.dumps({"model":"hura:latest",
            "messages":[{"role":"user","content":"Write one sentence about the ocean."}],
            "max_tokens":48,"temperature":0})
        s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {HOST}\r\n"
                   f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                   f"Connection: close\r\n\r\n{body}").encode())
        buf = b""
        s.settimeout(90)
        while True:
            d = s.recv(4096)
            if not d: break
            buf += d
        s.close()
        import hashlib, re
        m = re.search(rb"\r\n\r\n(.*)$", buf, re.S)
        body_r = m.group(1) if m else buf
        try:
            j = json.loads(body_r[body_r.index(b"{"):])
            c = j["choices"][0]["message"]["content"]
            h = hashlib.sha256(c.encode()).hexdigest()[:16]
            return "ok" if h == "8752e132c193abbe" else "POISONED:" + h
        except Exception:
            return "parse-err"
    except Exception:
        return "err"

p0 = pid()
print("pid", p0, flush=True)
for r in range(1, ROUNDS+1):
    ts = []
    for i in range(3):
        mode = "rst" if (r + i) % 2 == 0 else "fin"
        t = threading.Thread(target=aborted_stream, args=(f"{r}-{i}", mode)); t.start(); ts.append(t)
    for t in ts: t.join()
    h = healthy()
    pn = pid()
    print(f"round {r}: pid={pn} health={h}", flush=True)
    if pn != p0:
        print(f"=== CRASH at round {r} ===", flush=True)
        sys.exit(0)
    if h.startswith("POISONED"):
        # transient vs persistent: give in-flight abort processing 6s to
        # settle, then re-probe twice. Only a PERSISTENT drift is the state
        # we are hunting; a transient one self-heals (seen in autopsy #1).
        time.sleep(6)
        h2 = healthy(); h3 = healthy()
        if h2.startswith("POISONED") and h3.startswith("POISONED"):
            print(f"=== POISONED-PERSISTENT at round {r} ({h} then {h2}/{h3}) ===", flush=True)
            sys.exit(2)
        print(f"round {r}: transient drift ({h} -> {h2}/{h3}), storm continues", flush=True)
print("SURVIVED", ROUNDS, "rounds", flush=True)
