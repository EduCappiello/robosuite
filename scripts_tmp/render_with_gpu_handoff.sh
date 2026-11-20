#!/usr/bin/env bash
# Free the GPU from the local Strata server, render the replay, bring the server back.
#
# The Strata server on :8084 is the backend serving THIS agent, so this script kills its own
# caller's inference. It is therefore written to be run DETACHED (setsid + nohup) and to
# restore the server unconditionally: the EXIT trap restarts it no matter how the script
# ends. Restart uses the operator's own launcher ~/src/Strata/run-iq3_s.sh, not a
# reconstructed command line.
#
#   setsid nohup bash scripts_tmp/render_with_gpu_handoff.sh > /tmp/gpu_handoff.log 2>&1 &
set -u

LOG=/tmp/gpu_handoff.log
PORT=8084
RENDER_DIR=/home/ecappiell/Documents/dev/robosuite
OUT=/home/ecappiell/Documents/dev/lerobot-1-tg/outputs/next_free/replay_render.mp4
LAUNCHER=/home/ecappiell/src/Strata/run-iq3_s.sh
SERVER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

server_pids() {
  pgrep -f "serve/server.py --engine strata --config .*strata-iq3_s.json --port $PORT" 2>/dev/null
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

wait_for_port() {
  local deadline=$((SECONDS + ${1:-600}))
  while [ $SECONDS -lt $deadline ]; do
    if curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/v1/models"; then return 0; fi
    sleep 5
  done
  return 1
}

restart_server() {
  if [ -n "$(server_pids)" ]; then log "server already running again"; return 0; fi
  log "restarting Strata via $LAUNCHER"
  setsid nohup bash "$LAUNCHER" >> /tmp/strata_restart.log 2>&1 < /dev/null &
  disown 2>/dev/null || true
  if wait_for_port 900; then
    log "server back on :$PORT (GPU now $(gpu_used_mib) MiB used)"
  else
    log "ERROR: server did not answer on :$PORT within 900 s -- check /tmp/strata_restart.log"
  fi
}

cleanup() {
  local rc=$?
  log "script exiting (rc=$rc)"
  restart_server
  log "DONE -- render at $OUT ; handoff log $LOG"
}
trap cleanup EXIT

log "=== GPU handoff start ==="
SERVER_PID=$(server_pids | head -1)
if [ -z "$SERVER_PID" ]; then
  log "no Strata server found on :$PORT; rendering with whatever GPU state exists"
else
  log "server pids: $(server_pids | tr '\n' ' ')  GPU before: $(gpu_used_mib) MiB"
  log "sending TERM to the server supervisor (this drops the agent's own backend)"
  kill -TERM $SERVER_PID 2>/dev/null || true
  for _ in $(seq 1 20); do
    [ -z "$(server_pids)" ] && break
    sleep 1
  done
  if [ -n "$(server_pids)" ]; then
    log "still alive after 20 s; sending KILL"
    kill -KILL $(server_pids) 2>/dev/null || true
    sleep 3
  fi
  for _ in $(seq 1 30); do
    u=$(gpu_used_mib); [ "${u:-99999}" -lt 1500 ] && break
    sleep 2
  done
  log "GPU after kill: $(gpu_used_mib) MiB"
fi

log "rendering (this is the part that needs the GPU)"
cd "$RENDER_DIR" || { log "ERROR: no $RENDER_DIR"; exit 1; }
MUJOCO_GL=egl LEROBOT=/home/ecappiell/Documents/dev/lerobot-1-tg
PYTHONPATH="$RENDER_DIR:$LEROBOT/src:$LEROBOT/scripts" \
  timeout 1800 conda run -n lerobot --no-capture-output \
  python scripts_tmp/render_free_motion_replay.py --seconds 12 --start 30 \
  --out "$OUT" 2>&1 | tail -40 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "render exit=$rc"
ls -la "$OUT" 2>/dev/null | tee -a "$LOG"
exit "$rc"
