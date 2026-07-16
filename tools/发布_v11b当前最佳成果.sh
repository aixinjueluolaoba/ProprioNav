#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
ALIAS_JSON=/home/diana/screencap/file/target8-best/manifest.json
ARCHIVE_LOG=/tmp/v11b_publish_archive.log
SYNC_LOG=/tmp/v11b_publish_sync.log
DIAG_LOG=/tmp/v11b_publish_diag.log

cd "$ROOT"

run_and_capture() {
  local name="$1"
  local log="$2"
  shift 2
  if ! "$@" >"$log" 2>&1; then
    echo "$name failed; last log lines:"
    tail -n 80 "$log" || true
    exit 1
  fi
}

run_and_capture archive "$ARCHIVE_LOG" bash tools/同步并归档_v11b当前最佳_v1002.sh
run_and_capture sync "$SYNC_LOG" bash tools/同步_v11b当前最佳成果到本地分享.sh
run_and_capture diagnose "$DIAG_LOG" bash tools/诊断_v11b分享链路.sh

tail -n 6 "$SYNC_LOG"
tail -n 12 "$DIAG_LOG"

python3 - <<'PY'
import json
import urllib.request
from pathlib import Path

manifest = Path("/home/diana/screencap/file/target8-best/manifest.json")
data = json.loads(manifest.read_text(encoding="utf-8"))
print("ascii_preview=" + data["preview"])
print("ascii_links=" + data["links"])
print("normal_video=" + data["normal_video"])
print("pressure_video=" + data["pressure_video"])
print("diagonal_video=" + data["diagonal_video"])

for key in ("preview", "links"):
    req = urllib.request.Request(data[key], headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        print(f"{key}_status={resp.status}")
PY

echo "manifest=$ALIAS_JSON"
echo "archive_log=$ARCHIVE_LOG"
echo "sync_log=$SYNC_LOG"
echo "diag_log=$DIAG_LOG"
