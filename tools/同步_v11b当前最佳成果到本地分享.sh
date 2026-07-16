#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME=目标8宏动作库v11b当前最佳成果
ALIAS_NAME=目标8当前最佳成果
ASCII_ALIAS_NAME=target8-best
LOCAL_ROOT=/home/diana/screencap/file
LOCAL_DIR=$LOCAL_ROOT/$REMOTE_NAME
ALIAS_DIR=$LOCAL_ROOT/$ALIAS_NAME
ASCII_ALIAS_DIR=$LOCAL_ROOT/$ASCII_ALIAS_NAME
REMOTE_DIR=/home/diana/screencap/file/$REMOTE_NAME
LOCAL_API=http://127.0.0.1:6000/api/files
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
    "DOMAIN_BASE": "http://ai.jiaran.icu:12346/file",
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
    "DOMAIN_PUBLIC_PREVIEW": make(bases["DOMAIN_BASE"], remote),
    "IP_PUBLIC_PREVIEW": make(bases["IP_BASE"], remote),
}
for key, value in exports.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$URL_EXPORTS"

mkdir -p "$LOCAL_ROOT"
rm -rf "$LOCAL_DIR"
rm -rf "$ALIAS_DIR"
rm -rf "$ASCII_ALIAS_DIR"

scp -r "v1002:$REMOTE_DIR" "$LOCAL_ROOT/"

test -f "$LOCAL_DIR/成果预览.html"
test -f "$LOCAL_DIR/最终选择说明.md"
find "$LOCAL_DIR" -maxdepth 1 -type f -name 'v11b_*.mp4' | grep -q .
find "$LOCAL_DIR" -maxdepth 1 -type f -name 'v11b_*.csv' | grep -q .

BEST_BASENAME="$(
  find "$LOCAL_DIR" -maxdepth 1 -type f -name 'v11b_*普通评估汇总.csv' \
    | head -n 1 \
    | xargs -r -n 1 basename \
    | sed 's/普通评估汇总\.csv$//'
)"
test -n "$BEST_BASENAME"

cp -r "$LOCAL_DIR" "$ALIAS_DIR"
cp -r "$LOCAL_DIR" "$ASCII_ALIAS_DIR"

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

if [ -n "$PUBLIC_BASE" ]; then
  SERVICE_PUBLIC_PREVIEW="$(
    PUBLIC_BASE="$PUBLIC_BASE" python3 - <<'PY'
from urllib.parse import quote
import os
base = os.environ["PUBLIC_BASE"].rstrip("/")
print(f"{base}/{quote('目标8宏动作库v11b当前最佳成果/成果预览.html')}")
PY
  )"
  EFFECTIVE_PUBLIC_BASE="$PUBLIC_BASE"
else
  SERVICE_PUBLIC_PREVIEW=
  EFFECTIVE_PUBLIC_BASE="http://47.99.153.111:12346/file"
fi

status() {
  local url="$1"
  if [ -z "$url" ]; then
    echo "empty"
    return
  fi
  curl -s -L -o /tmp/v11b_share_probe.bin -w '%{http_code}' "$url" || true
}

DOMAIN_STATUS="$(status "$DOMAIN_PUBLIC_PREVIEW")"
IP_STATUS="$(status "$IP_PUBLIC_PREVIEW")"
if [ "$DOMAIN_STATUS" = "200" ]; then
  VERIFIED_EXTERNAL_PREVIEW="$DOMAIN_PUBLIC_PREVIEW"
elif [ "$IP_STATUS" = "200" ]; then
  VERIFIED_EXTERNAL_PREVIEW="$IP_PUBLIC_PREVIEW"
else
  VERIFIED_EXTERNAL_PREVIEW=
fi

export EFFECTIVE_PUBLIC_BASE ALIAS_NAME ASCII_ALIAS_NAME BEST_BASENAME
URL_EXPORTS="$(
python3 - <<'PY'
import os
import shlex
from urllib.parse import quote

base = os.environ["EFFECTIVE_PUBLIC_BASE"].rstrip("/")
alias = os.environ["ALIAS_NAME"]
ascii_alias = os.environ["ASCII_ALIAS_NAME"]
basename = os.environ["BEST_BASENAME"]

def file_url(folder: str, name: str) -> str:
    return f"{base}/{quote(folder + '/' + name)}"

exports = {
    "ALIAS_PUBLIC_PREVIEW": file_url(alias, "成果预览.html"),
    "ALIAS_INFO_URL": file_url(alias, "链接清单.md"),
    "ALIAS_MANIFEST_URL": file_url(alias, "分享清单.json"),
    "ALIAS_NORMAL_VIDEO_URL": file_url(alias, f"{basename}普通评估20轮拼接.mp4"),
    "ALIAS_PRESSURE_VIDEO_URL": file_url(alias, f"{basename}压力评估20轮拼接.mp4"),
    "ALIAS_DIAGONAL_VIDEO_URL": file_url(alias, f"{basename}对角高障碍20轮拼接.mp4"),
    "ASCII_ALIAS_PUBLIC_PREVIEW": file_url(ascii_alias, "preview.html"),
    "ASCII_ALIAS_INFO_URL": file_url(ascii_alias, "links.md"),
    "ASCII_ALIAS_MANIFEST_URL": file_url(ascii_alias, "manifest.json"),
    "ASCII_ALIAS_README_URL": file_url(ascii_alias, "readme.md"),
    "ASCII_ALIAS_NORMAL_VIDEO_URL": file_url(ascii_alias, "normal.mp4"),
    "ASCII_ALIAS_PRESSURE_VIDEO_URL": file_url(ascii_alias, "pressure.mp4"),
    "ASCII_ALIAS_DIAGONAL_VIDEO_URL": file_url(ascii_alias, "diagonal.mp4"),
    "ASCII_ALIAS_NORMAL_CSV_URL": file_url(ascii_alias, "normal.csv"),
    "ASCII_ALIAS_PRESSURE_CSV_URL": file_url(ascii_alias, "pressure.csv"),
    "ASCII_ALIAS_DIAGONAL_CSV_URL": file_url(ascii_alias, "diagonal.csv"),
}
for key, value in exports.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$URL_EXPORTS"

cat > "$LOCAL_DIR/链接清单.md" <<EOF
# v11b 当前最佳分享链接

- 长目录预览页：$SERVICE_PUBLIC_PREVIEW
- 稳定短目录预览页：$ALIAS_PUBLIC_PREVIEW
- ASCII 预览页：$ASCII_ALIAS_PUBLIC_PREVIEW
- 稳定短目录普通评估视频：$ALIAS_NORMAL_VIDEO_URL
- 稳定短目录压力评估视频：$ALIAS_PRESSURE_VIDEO_URL
- 稳定短目录对角高障碍视频：$ALIAS_DIAGONAL_VIDEO_URL
- 稳定短目录 JSON 清单：$ALIAS_MANIFEST_URL
- ASCII JSON 清单：$ASCII_ALIAS_MANIFEST_URL
- 说明：
  - 当前可用公网基线优先使用上面的 IP 外链
  - 域名候选 $DOMAIN_PUBLIC_PREVIEW 当前仍返回 $DOMAIN_STATUS
EOF

cp "$LOCAL_DIR/链接清单.md" "$ALIAS_DIR/链接清单.md"
cp "$LOCAL_DIR/成果预览.html" "$ASCII_ALIAS_DIR/preview.html"
cp "$LOCAL_DIR/最终选择说明.md" "$ASCII_ALIAS_DIR/readme.md"
cp "$LOCAL_DIR/${BEST_BASENAME}普通评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/normal.mp4"
cp "$LOCAL_DIR/${BEST_BASENAME}压力评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/pressure.mp4"
cp "$LOCAL_DIR/${BEST_BASENAME}对角高障碍20轮拼接.mp4" "$ASCII_ALIAS_DIR/diagonal.mp4"
cp "$LOCAL_DIR/${BEST_BASENAME}普通评估汇总.csv" "$ASCII_ALIAS_DIR/normal.csv"
cp "$LOCAL_DIR/${BEST_BASENAME}压力评估汇总.csv" "$ASCII_ALIAS_DIR/pressure.csv"
cp "$LOCAL_DIR/${BEST_BASENAME}对角高障碍汇总.csv" "$ASCII_ALIAS_DIR/diagonal.csv"

cat > "$ALIAS_DIR/分享清单.json" <<EOF
{
  "remote_name": "$REMOTE_NAME",
  "alias_name": "$ALIAS_NAME",
  "service_public_base": "$EFFECTIVE_PUBLIC_BASE",
  "verified_external_preview": "$VERIFIED_EXTERNAL_PREVIEW",
  "stable_preview": "$ALIAS_PUBLIC_PREVIEW",
  "stable_info": "$ALIAS_INFO_URL",
  "normal_video": "$ALIAS_NORMAL_VIDEO_URL",
  "pressure_video": "$ALIAS_PRESSURE_VIDEO_URL",
  "diagonal_video": "$ALIAS_DIAGONAL_VIDEO_URL"
}
EOF

cat > "$ASCII_ALIAS_DIR/manifest.json" <<EOF
{
  "remote_name": "$REMOTE_NAME",
  "alias_name": "$ASCII_ALIAS_NAME",
  "service_public_base": "$EFFECTIVE_PUBLIC_BASE",
  "preview": "$ASCII_ALIAS_PUBLIC_PREVIEW",
  "links": "$ASCII_ALIAS_INFO_URL",
  "readme": "$ASCII_ALIAS_README_URL",
  "normal_video": "$ASCII_ALIAS_NORMAL_VIDEO_URL",
  "pressure_video": "$ASCII_ALIAS_PRESSURE_VIDEO_URL",
  "diagonal_video": "$ASCII_ALIAS_DIAGONAL_VIDEO_URL",
  "normal_csv": "$ASCII_ALIAS_NORMAL_CSV_URL",
  "pressure_csv": "$ASCII_ALIAS_PRESSURE_CSV_URL",
  "diagonal_csv": "$ASCII_ALIAS_DIAGONAL_CSV_URL"
}
EOF

cat > "$ASCII_ALIAS_DIR/links.md" <<EOF
# v11b current best links

- preview: $ASCII_ALIAS_PUBLIC_PREVIEW
- manifest: $ASCII_ALIAS_MANIFEST_URL
- readme: $ASCII_ALIAS_README_URL
- normal_video: $ASCII_ALIAS_NORMAL_VIDEO_URL
- pressure_video: $ASCII_ALIAS_PRESSURE_VIDEO_URL
- diagonal_video: $ASCII_ALIAS_DIAGONAL_VIDEO_URL
- normal_csv: $ASCII_ALIAS_NORMAL_CSV_URL
- pressure_csv: $ASCII_ALIAS_PRESSURE_CSV_URL
- diagonal_csv: $ASCII_ALIAS_DIAGONAL_CSV_URL
EOF

echo "local_dir=$LOCAL_DIR"
echo "alias_dir=$ALIAS_DIR"
echo "ascii_alias_dir=$ASCII_ALIAS_DIR"
echo "local_preview=$LOCAL_PREVIEW"
echo "alias_local_preview=$ALIAS_LOCAL_PREVIEW"
echo "ascii_alias_local_preview=$ASCII_ALIAS_LOCAL_PREVIEW"
echo "domain_preview=$DOMAIN_PUBLIC_PREVIEW"
echo "domain_status=$DOMAIN_STATUS"
echo "ip_preview=$IP_PUBLIC_PREVIEW"
echo "ip_status=$IP_STATUS"
echo "public_base=$PUBLIC_BASE"
echo "service_public_preview=$SERVICE_PUBLIC_PREVIEW"
echo "verified_external_preview=$VERIFIED_EXTERNAL_PREVIEW"
echo "alias_public_preview=$ALIAS_PUBLIC_PREVIEW"
echo "alias_info_url=$ALIAS_INFO_URL"
echo "ascii_alias_public_preview=$ASCII_ALIAS_PUBLIC_PREVIEW"
echo "ascii_alias_info_url=$ASCII_ALIAS_INFO_URL"
