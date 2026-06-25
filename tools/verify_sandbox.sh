#!/usr/bin/env bash
# verify_sandbox.sh — hard sandbox for the Aspida verifier (set
# ASPIDA_VERIFY_SANDBOX to this script; Exec_Verifier then wraps every untrusted
# execution with it). Uses only preinstalled util-linux tools (no apt footprint):
#
#   * unshare -n   : private network namespace -> NO egress (blocks data
#                    exfiltration and the cloud-metadata IP 169.254.169.254)
#   * setpriv      : drop to nobody:nogroup (non-root, cleared supplementary groups)
#   * timeout -KILL: hard wall-clock bound (kills runaways / while-True)
#   * ulimit       : address space, CPU time, process count, file size caps
#
# Usage:  verify_sandbox.sh <program> [args...]
# Env:    ASPIDA_SBX_TIMEOUT (seconds, default 10), ASPIDA_SBX_MEM_KB (default 1048576)
set -u
TMO="${ASPIDA_SBX_TIMEOUT:-10}"
MEM="${ASPIDA_SBX_MEM_KB:-1048576}"     # 1 GiB address space
exec unshare -n -- \
  timeout -s KILL "$TMO" \
  setpriv --reuid 65534 --regid 65534 --clear-groups -- \
  bash --noprofile --norc -c 'ulimit -v '"$MEM"' -t '"$TMO"' -u 128 -f 65536 2>/dev/null; exec "$@"' _ "$@"
