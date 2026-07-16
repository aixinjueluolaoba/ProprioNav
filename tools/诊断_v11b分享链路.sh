#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME=目标8宏动作库v11b当前最佳成果
ALIAS_NAME=目标8当前最佳成果
ASCII_ALIAS_NAME=target8-best
PREVIEW_NAME=成果预览.html

URL_EXPORTS="$(
python3 - <<'PY'
import shlex
from urllib.parse import quote

remote = "目标8宏动作库v11b当前最佳成果"
alias = "目标8当前最佳成果"
ascii_alias = "target8-best"
preview = "成果预览.html"
ascii_preview = "preview.html"
bases = {
    "LOCAL_BASE": "http://127.0.0.1:6000/file",
    "PROXY_BASE": "http://ai.jiaran.icu:12346/file",
    "IP_BASE": "http://47.99.153.111:12346/file",
}

def make(base: str, folder: str) -> str:
    return f"{base}/{quote(folder + '/' + preview)}"

def make_ascii(base: str, folder: str) -> str:
    return f"{base}/{quote(folder + '/' + ascii_preview)}"

exports = {
    "LOCAL_PREVIEW": make(bases["LOCAL_BASE"], remote),
    "ALIAS_LOCAL_PREVIEW": make(bases["LOCAL_BASE"], alias),
    "ASCII_ALIAS_LOCAL_PREVIEW": make_ascii(bases["LOCAL_BASE"], ascii_alias),
    "PROXY_PREVIEW": make(bases["PROXY_BASE"], remote),
    "IP_PREVIEW": make(bases["IP_BASE"], remote),
    "ALIAS_IP_PREVIEW": make(bases["IP_BASE"], alias),
    "ASCII_ALIAS_IP_PREVIEW": make_ascii(bases["IP_BASE"], ascii_alias),
}
for key, value in exports.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$URL_EXPORTS"

PUBLIC_BASE="$(
  python3 - <<'PY'
import json
import urllib.request

url = "http://127.0.0.1:6000/api/files?limit=1"
with urllib.request.urlopen(url, timeout=5) as resp:
    data = json.load(resp)
print(data.get("public_base", ""))
PY
)"

SERVICE_PREVIEW=
if [ -n "$PUBLIC_BASE" ]; then
  SERVICE_PREVIEW="$(
    PUBLIC_BASE="$PUBLIC_BASE" python3 - <<'PY'
from urllib.parse import quote
import os
base = os.environ["PUBLIC_BASE"].rstrip("/")
print(f"{base}/{quote('目标8宏动作库v11b当前最佳成果/成果预览.html')}")
PY
  )"
fi

status() {
  local url="$1"
  if [ -z "$url" ]; then
    echo "empty"
    return
  fi
  curl -s -L -o /tmp/v11b_share_probe.bin -w '%{http_code}' "$url" || true
}

BEST_EXTERNAL_PREVIEW=
PROXY_STATUS="$(status "$PROXY_PREVIEW")"
IP_STATUS="$(status "$IP_PREVIEW")"
if [ "$PROXY_STATUS" = "200" ]; then
  BEST_EXTERNAL_PREVIEW="$PROXY_PREVIEW"
elif [ "$IP_STATUS" = "200" ]; then
  BEST_EXTERNAL_PREVIEW="$IP_PREVIEW"
fi

echo "local_preview=$LOCAL_PREVIEW"
echo "local_status=$(status "$LOCAL_PREVIEW")"
echo "alias_local_preview=$ALIAS_LOCAL_PREVIEW"
echo "alias_local_status=$(status "$ALIAS_LOCAL_PREVIEW")"
echo "ascii_alias_local_preview=$ASCII_ALIAS_LOCAL_PREVIEW"
echo "ascii_alias_local_status=$(status "$ASCII_ALIAS_LOCAL_PREVIEW")"
echo "proxy_preview=$PROXY_PREVIEW"
echo "proxy_status=$PROXY_STATUS"
echo "ip_preview=$IP_PREVIEW"
echo "ip_status=$IP_STATUS"
echo "alias_ip_preview=$ALIAS_IP_PREVIEW"
echo "alias_ip_status=$(status "$ALIAS_IP_PREVIEW")"
echo "ascii_alias_ip_preview=$ASCII_ALIAS_IP_PREVIEW"
echo "ascii_alias_ip_status=$(status "$ASCII_ALIAS_IP_PREVIEW")"
echo "service_public_base=$PUBLIC_BASE"
echo "service_public_preview=$SERVICE_PREVIEW"
echo "service_public_status=$(status "$SERVICE_PREVIEW")"
echo "best_external_preview=$BEST_EXTERNAL_PREVIEW"
