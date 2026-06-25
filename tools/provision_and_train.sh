#!/usr/bin/env bash
# provision_and_train.sh — platform provisioning step: take a droplet count, spin
# GPU droplets, deploy the resident-Student shim + dp_node, run REAL distributed
# (data-parallel, TCP all-reduce) training, collect the result, and ALWAYS tear
# the droplets down. The "engineer picks N droplets -> we provision -> train ->
# teardown" step, built on the validated dp_node (Step 7).
#
#   ./provision_and_train.sh --dry-run            # print the plan, spend $0
#   ./provision_and_train.sh --run                # real: creates 2 GPU droplets ($$)
#
# Env: REGION (tor1), SIZE (gpu-l40sx1-48gb), SSH_KEY (54597878), ROUNDS (50),
#      PORT (5599). dp_node is a 2-rank prototype; N>2 needs ring all-reduce.
set -euo pipefail

MODE="${1:---dry-run}"
N=2                                   # dp_node supports 2 ranks (server + client)
REGION="${REGION:-tor1}"
SIZE="${SIZE:-gpu-l40sx1-48gb}"
SSH_KEY="${SSH_KEY:-54597878}"
ROUNDS="${ROUNDS:-50}"
PORT="${PORT:-5599}"
IMG="${IMG:-gpu-h100x1-base}"         # DO GPU image (overridable)
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAMES=(aspida-dp-0 aspida-dp-1)

plan() {
  echo "PLAN: distributed training on $N x $SIZE in $REGION"
  echo "  1. create droplets: ${NAMES[*]}  (ssh-key $SSH_KEY, --wait)"
  echo "  2. deploy gpu/student_shim.cu, student_kernels.cuh, dp_node.cu to each"
  echo "  3. build libaspidastudent.so + dp_node on each (nvcc)"
  echo "  4. run rank0 (server) on aspida-dp-0; rank1 (client) -> dp-0 PRIVATE ip; $ROUNDS rounds, port $PORT"
  echo "  5. collect both ranks' final_loss (must match -> all-reduce in sync)"
  echo "  6. ALWAYS destroy droplets (trap on exit)"
}

if [ "$MODE" = "--dry-run" ]; then
  plan
  echo "DRY-RUN: no droplets created, \$0 spent."
  exit 0
fi
[ "$MODE" = "--run" ] || { echo "usage: $0 --dry-run|--run"; exit 2; }

plan
DELETED=0
cleanup(){ if [ "$DELETED" = 0 ]; then echo ">> teardown"; doctl compute droplet delete "${NAMES[@]}" --force 2>/dev/null || true; DELETED=1; fi; }
trap cleanup EXIT INT TERM

echo ">> creating $N GPU droplets (capacity may fluctuate; retrying)"
for try in 1 2 3 4 5; do
  if doctl compute droplet create "${NAMES[@]}" --region "$REGION" --size "$SIZE" \
       --image "$IMG" --ssh-keys "$SSH_KEY" --wait \
       --format Name,PublicIPv4,PrivateIPv4 2>/dev/null; then break; fi
  echo "   create attempt $try failed (GPU capacity?); retrying"; sleep 20
done

read_ip(){ doctl compute droplet list --format Name,"$1" --no-header | awk -v n="$2" '$1==n{print $2}'; }
P0=$(read_ip PublicIPv4 aspida-dp-0);  PRIV0=$(read_ip PrivateIPv4 aspida-dp-0)
P1=$(read_ip PublicIPv4 aspida-dp-1)
echo ">> dp-0 pub=$P0 priv=$PRIV0 ; dp-1 pub=$P1"

deploy(){ local ip=$1
  for i in $(seq 1 30); do ssh $SSHO root@"$ip" "echo up" >/dev/null 2>&1 && break; sleep 5; done
  ssh $SSHO root@"$ip" "mkdir -p /root/dp"
  scp $SSHO "$HERE/gpu/student_shim.cu" "$HERE/gpu/student_kernels.cuh" "$HERE/gpu/dp_node.cu" root@"$ip":/root/dp/
  ssh $SSHO root@"$ip" 'export PATH=/usr/local/cuda/bin:$PATH; cd /root/dp;
    nvcc -O3 -arch=native -shared -Xcompiler -fPIC student_shim.cu -o libaspidastudent.so &&
    nvcc -O3 -arch=native dp_node.cu -L. -laspidastudent -o dp_node && echo BUILT_OK'
}
echo ">> deploying + building on both"; deploy "$P0"; deploy "$P1"

echo ">> launching distributed run (real TCP all-reduce over the VPC)"
ssh $SSHO root@"$P0" "cd /root/dp; LD_LIBRARY_PATH=. ./dp_node 0 0.0.0.0 $PORT $ROUNDS" > /tmp/dp0.txt 2>&1 &
sleep 3
ssh $SSHO root@"$P1" "cd /root/dp; LD_LIBRARY_PATH=. ./dp_node 1 $PRIV0 $PORT $ROUNDS" > /tmp/dp1.txt 2>&1
wait || true
echo "--- rank 0 ---"; cat /tmp/dp0.txt; echo "--- rank 1 ---"; cat /tmp/dp1.txt

cleanup
echo ">> done (droplets destroyed)"
