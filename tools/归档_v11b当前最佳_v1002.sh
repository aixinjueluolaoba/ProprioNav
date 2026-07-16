#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
RUN_DIR=$ROOT/runs/目标8宏动作库v11b卡住探索短训_v1002
EVAL_DIR=$ROOT/runs/目标8宏动作库v11b_checkpoint评估_v1002
OUT_DIR=$ROOT/runs/目标8宏动作库v11b当前最佳归档_v1002
SHARE_DIR=/home/diana/screencap/file/目标8宏动作库v11b当前最佳成果
ALIAS_DIR=/home/diana/screencap/file/目标8当前最佳成果
ASCII_ALIAS_DIR=/home/diana/screencap/file/target8-best
OVERRIDES=$RUN_DIR/env_overrides.json
PRESSURE_DIR=$OUT_DIR/压力评估20
DIAGONAL_DIR=$OUT_DIR/对角高障碍20
PUBLIC_FILE_BASE="${PUBLIC_FILE_BASE:-http://47.99.153.111:12346/file}"

cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML

mkdir -p "$OUT_DIR" "$SHARE_DIR" "$ALIAS_DIR" "$ASCII_ALIAS_DIR"

export EVAL_DIR
BEST_EXPORTS="$(
python3 - <<'PY'
import csv
import os
import shlex
from pathlib import Path

summary = Path(os.environ["EVAL_DIR"]) / "checkpoint_eval_summary.csv"
stage_to_state = {
    "stage1": "observable12_target8_macro_library_v11_stage1",
    "stage2a": "observable12_target8_macro_library_v11_stage2a",
    "stage2": "observable12_target8_macro_library_v11_stage2",
    "stage3": "observable12_target8_macro_library_v11_stage3",
}

rows = list(csv.DictReader(summary.open(encoding="utf-8")))
if not rows:
    raise SystemExit(f"empty summary: {summary}")

def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0.0)

def choose_best(candidates: list[dict[str, str]]) -> dict[str, str]:
    complete = [
        row for row in candidates
        if f(row, "normal_success") >= 0.999999 and f(row, "pressure_success") >= 0.999999
    ]
    pool = complete or candidates
    return min(
        pool,
        key=lambda row: (
            -(1 if row in complete else 0),
            -(f(row, "pressure_success") + f(row, "normal_success")),
            f(row, "pressure_avg_steps") + f(row, "normal_avg_steps"),
            f(row, "pressure_avg_collisions"),
            int(row["checkpoint"]),
        ),
    )

best = choose_best(rows)
model = Path(best["model"])
stage = best["stage"]
checkpoint = best["checkpoint"]
experiment = model.parents[1].name
state_mode = stage_to_state[stage]
basename = f"v11b_{stage}_{checkpoint}"

exports = {
    "BEST_STAGE": stage,
    "BEST_CHECKPOINT": checkpoint,
    "BEST_EXPERIMENT": experiment,
    "BEST_STATE_MODE": state_mode,
    "BEST_MODEL": str(model),
    "BEST_BASENAME": basename,
    "BEST_NORMAL_SUCCESS": best["normal_success"],
    "BEST_NORMAL_AVG_STEPS": best["normal_avg_steps"],
    "BEST_PRESSURE_SUCCESS": best["pressure_success"],
    "BEST_PRESSURE_AVG_STEPS": best["pressure_avg_steps"],
    "BEST_PRESSURE_AVG_COLLISIONS": best["pressure_avg_collisions"],
}
for key, value in exports.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$BEST_EXPORTS"

NORMAL_DIR=$OUT_DIR/普通评估20/$BEST_EXPERIMENT

python eval_target8_sweep.py \
  --train-output-dir "$RUN_DIR" \
  --model-path "$BEST_MODEL" \
  --out-dir "$OUT_DIR/普通评估20" \
  --experiments "$BEST_EXPERIMENT" \
  --episodes 20 \
  --video-episodes 20 \
  --env-overrides-file "$OVERRIDES"

python eval_target8_pressure.py \
  --model "$BEST_MODEL" \
  --out-dir "$OUT_DIR/压力评估20" \
  --action-mode v11 \
  --state-mode "$BEST_STATE_MODE" \
  --episodes 20 \
  --video-episodes 20 \
  --env-overrides-file "$OVERRIDES"

python eval_target8_diagonal_dense_video.py \
  --model "$BEST_MODEL" \
  --out-dir "$OUT_DIR/对角高障碍20" \
  --episodes 20 \
  --video-episodes 20 \
  --env-overrides-file "$OVERRIDES"

rm -f "$OUT_DIR"/v11b_*
cp "$OUT_DIR/普通评估20/target8_sweep_summary.csv" "$OUT_DIR/${BEST_BASENAME}普通评估汇总.csv"
cp "$PRESSURE_DIR/pressure_eval_summary.csv" "$OUT_DIR/${BEST_BASENAME}压力评估汇总.csv"
cp "$DIAGONAL_DIR/diagonal_dense_eval_summary.csv" "$OUT_DIR/${BEST_BASENAME}对角高障碍汇总.csv"
cp "$NORMAL_DIR/target8_eval_concat.mp4" "$OUT_DIR/${BEST_BASENAME}普通评估20轮拼接.mp4"
cp "$PRESSURE_DIR/pressure_eval_concat.mp4" "$OUT_DIR/${BEST_BASENAME}压力评估20轮拼接.mp4"
cp "$DIAGONAL_DIR/diagonal_dense_eval_concat.mp4" "$OUT_DIR/${BEST_BASENAME}对角高障碍20轮拼接.mp4"

cat > "$OUT_DIR/最终选择说明.md" <<EOF
# v11b 当前最佳归档

- 模型：\`$BEST_STAGE checkpoint_ep_$BEST_CHECKPOINT.zip\`
- 模型路径：\`$BEST_MODEL\`
- 评估汇总：\`$EVAL_DIR/checkpoint_eval_summary.csv\`
- 自动选择规则：
  - 先优先 \`normal_success=1.0\` 且 \`pressure_success=1.0\`
  - 再按 \`normal_avg_steps + pressure_avg_steps\` 最小
  - 再按 \`pressure_avg_collisions\` 最小
  - 最后取更早的 checkpoint
- 本次选中结果：
  - \`normal_success=$BEST_NORMAL_SUCCESS\`
  - \`normal_avg_steps=$BEST_NORMAL_AVG_STEPS\`
  - \`pressure_success=$BEST_PRESSURE_SUCCESS\`
  - \`pressure_avg_steps=$BEST_PRESSURE_AVG_STEPS\`
  - \`pressure_avg_collisions=$BEST_PRESSURE_AVG_COLLISIONS\`
EOF

rm -f "$SHARE_DIR"/v11b_*
cp "$OUT_DIR"/v11b_* "$SHARE_DIR"/
cp "$OUT_DIR/最终选择说明.md" "$SHARE_DIR/"

cat > "$SHARE_DIR/成果预览.html" <<EOF
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v11b 当前最佳成果</title>
  <style>
    body { font-family: sans-serif; margin: 24px; line-height: 1.5; background: #f5f1e8; color: #1f2328; }
    h1 { margin-bottom: 8px; }
    .grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card { background: #fffdf8; border: 1px solid #d9d1c3; border-radius: 12px; padding: 16px; }
    video { width: 100%; border-radius: 8px; background: #000; }
    a { color: #0b57d0; }
    code { background: #f0eadf; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>目标8宏动作库 v11b 当前最佳成果</h1>
  <p>当前最佳 checkpoint：<code>$BEST_STAGE checkpoint_ep_$BEST_CHECKPOINT.zip</code></p>
  <p><a href="最终选择说明.md">最终选择说明.md</a></p>
  <div class="grid">
    <div class="card">
      <h2>普通评估 20 轮</h2>
      <video controls src="${BEST_BASENAME}普通评估20轮拼接.mp4"></video>
      <p><a href="${BEST_BASENAME}普通评估汇总.csv">汇总 CSV</a></p>
    </div>
    <div class="card">
      <h2>压力评估 20 轮</h2>
      <video controls src="${BEST_BASENAME}压力评估20轮拼接.mp4"></video>
      <p><a href="${BEST_BASENAME}压力评估汇总.csv">汇总 CSV</a></p>
    </div>
    <div class="card">
      <h2>对角高障碍 20 轮</h2>
      <video controls src="${BEST_BASENAME}对角高障碍20轮拼接.mp4"></video>
      <p><a href="${BEST_BASENAME}对角高障碍汇总.csv">汇总 CSV</a></p>
    </div>
  </div>
</body>
</html>
EOF

export PUBLIC_FILE_BASE BEST_BASENAME
URL_EXPORTS="$(
python3 - <<'PY'
import os
import shlex
from urllib.parse import quote

base = os.environ["PUBLIC_FILE_BASE"].rstrip("/")
basename = os.environ["BEST_BASENAME"]
alias = "目标8当前最佳成果"
ascii_alias = "target8-best"
long_name = "目标8宏动作库v11b当前最佳成果"

def file_url(folder: str, name: str) -> str:
    return f"{base}/{quote(folder + '/' + name)}"

exports = {
    "LONG_PUBLIC_PREVIEW": file_url(long_name, "成果预览.html"),
    "ALIAS_PUBLIC_PREVIEW": file_url(alias, "成果预览.html"),
    "ASCII_ALIAS_PUBLIC_PREVIEW": file_url(ascii_alias, "成果预览.html"),
    "ASCII_ALIAS_PUBLIC_PREVIEW_ASCII": file_url(ascii_alias, "preview.html"),
    "ALIAS_INFO_URL": file_url(alias, "链接清单.md"),
    "ASCII_ALIAS_INFO_URL": file_url(ascii_alias, "links.md"),
    "ALIAS_MANIFEST_URL": file_url(alias, "分享清单.json"),
    "ASCII_ALIAS_MANIFEST_URL": file_url(ascii_alias, "manifest.json"),
    "ALIAS_NORMAL_VIDEO_URL": file_url(alias, f"{basename}普通评估20轮拼接.mp4"),
    "ALIAS_PRESSURE_VIDEO_URL": file_url(alias, f"{basename}压力评估20轮拼接.mp4"),
    "ALIAS_DIAGONAL_VIDEO_URL": file_url(alias, f"{basename}对角高障碍20轮拼接.mp4"),
    "ASCII_ALIAS_NORMAL_VIDEO_URL": file_url(ascii_alias, f"{basename}普通评估20轮拼接.mp4"),
    "ASCII_ALIAS_PRESSURE_VIDEO_URL": file_url(ascii_alias, f"{basename}压力评估20轮拼接.mp4"),
    "ASCII_ALIAS_DIAGONAL_VIDEO_URL": file_url(ascii_alias, f"{basename}对角高障碍20轮拼接.mp4"),
    "ASCII_ALIAS_NORMAL_VIDEO_URL_ASCII": file_url(ascii_alias, "normal.mp4"),
    "ASCII_ALIAS_PRESSURE_VIDEO_URL_ASCII": file_url(ascii_alias, "pressure.mp4"),
    "ASCII_ALIAS_DIAGONAL_VIDEO_URL_ASCII": file_url(ascii_alias, "diagonal.mp4"),
    "ASCII_ALIAS_NORMAL_CSV_URL_ASCII": file_url(ascii_alias, "normal.csv"),
    "ASCII_ALIAS_PRESSURE_CSV_URL_ASCII": file_url(ascii_alias, "pressure.csv"),
    "ASCII_ALIAS_DIAGONAL_CSV_URL_ASCII": file_url(ascii_alias, "diagonal.csv"),
    "ASCII_ALIAS_README_URL_ASCII": file_url(ascii_alias, "readme.md"),
}
for key, value in exports.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"
eval "$URL_EXPORTS"

cat > "$SHARE_DIR/链接清单.md" <<EOF
# v11b 当前最佳分享链接

- 长目录预览页：$LONG_PUBLIC_PREVIEW
- 稳定短目录预览页：$ALIAS_PUBLIC_PREVIEW
- ASCII 预览页：$ASCII_ALIAS_PUBLIC_PREVIEW
- 稳定短目录普通评估视频：$ALIAS_NORMAL_VIDEO_URL
- 稳定短目录压力评估视频：$ALIAS_PRESSURE_VIDEO_URL
- 稳定短目录对角高障碍视频：$ALIAS_DIAGONAL_VIDEO_URL
- 稳定短目录 JSON 清单：$ALIAS_MANIFEST_URL
- ASCII JSON 清单：$ASCII_ALIAS_MANIFEST_URL
EOF

rm -rf "$ALIAS_DIR"
rm -rf "$ASCII_ALIAS_DIR"
cp -r "$SHARE_DIR" "$ALIAS_DIR"
cp -r "$SHARE_DIR" "$ASCII_ALIAS_DIR"
cp "$SHARE_DIR/成果预览.html" "$ASCII_ALIAS_DIR/preview.html"
cp "$SHARE_DIR/最终选择说明.md" "$ASCII_ALIAS_DIR/readme.md"
cp "$SHARE_DIR/${BEST_BASENAME}普通评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/normal.mp4"
cp "$SHARE_DIR/${BEST_BASENAME}压力评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/pressure.mp4"
cp "$SHARE_DIR/${BEST_BASENAME}对角高障碍20轮拼接.mp4" "$ASCII_ALIAS_DIR/diagonal.mp4"
cp "$SHARE_DIR/${BEST_BASENAME}普通评估汇总.csv" "$ASCII_ALIAS_DIR/normal.csv"
cp "$SHARE_DIR/${BEST_BASENAME}压力评估汇总.csv" "$ASCII_ALIAS_DIR/pressure.csv"
cp "$SHARE_DIR/${BEST_BASENAME}对角高障碍汇总.csv" "$ASCII_ALIAS_DIR/diagonal.csv"

cat > "$ALIAS_DIR/分享清单.json" <<EOF
{
  "source_share_dir": "$SHARE_DIR",
  "alias_dir": "$ALIAS_DIR",
  "public_base": "$PUBLIC_FILE_BASE",
  "stable_preview": "$ALIAS_PUBLIC_PREVIEW",
  "stable_info": "$ALIAS_INFO_URL",
  "normal_video": "$ALIAS_NORMAL_VIDEO_URL",
  "pressure_video": "$ALIAS_PRESSURE_VIDEO_URL",
  "diagonal_video": "$ALIAS_DIAGONAL_VIDEO_URL"
}
EOF

cp "$SHARE_DIR/链接清单.md" "$ALIAS_DIR/链接清单.md"

cat > "$ASCII_ALIAS_DIR/links.md" <<EOF
# v11b current best links

- preview: $ASCII_ALIAS_PUBLIC_PREVIEW
- preview_ascii: $ASCII_ALIAS_PUBLIC_PREVIEW_ASCII
- manifest: $ASCII_ALIAS_MANIFEST_URL
- readme: $ASCII_ALIAS_README_URL_ASCII
- normal_video: $ASCII_ALIAS_NORMAL_VIDEO_URL_ASCII
- pressure_video: $ASCII_ALIAS_PRESSURE_VIDEO_URL_ASCII
- diagonal_video: $ASCII_ALIAS_DIAGONAL_VIDEO_URL_ASCII
- normal_csv: $ASCII_ALIAS_NORMAL_CSV_URL_ASCII
- pressure_csv: $ASCII_ALIAS_PRESSURE_CSV_URL_ASCII
- diagonal_csv: $ASCII_ALIAS_DIAGONAL_CSV_URL_ASCII
EOF

cat > "$ASCII_ALIAS_DIR/manifest.json" <<EOF
{
  "source_share_dir": "$SHARE_DIR",
  "alias_dir": "$ASCII_ALIAS_DIR",
  "public_base": "$PUBLIC_FILE_BASE",
  "preview": "$ASCII_ALIAS_PUBLIC_PREVIEW_ASCII",
  "links": "$ASCII_ALIAS_INFO_URL",
  "readme": "$ASCII_ALIAS_README_URL_ASCII",
  "normal_video": "$ASCII_ALIAS_NORMAL_VIDEO_URL_ASCII",
  "pressure_video": "$ASCII_ALIAS_PRESSURE_VIDEO_URL_ASCII",
  "diagonal_video": "$ASCII_ALIAS_DIAGONAL_VIDEO_URL_ASCII",
  "normal_csv": "$ASCII_ALIAS_NORMAL_CSV_URL_ASCII",
  "pressure_csv": "$ASCII_ALIAS_PRESSURE_CSV_URL_ASCII",
  "diagonal_csv": "$ASCII_ALIAS_DIAGONAL_CSV_URL_ASCII"
}
EOF

echo "archive_dir=$OUT_DIR"
echo "share_dir=$SHARE_DIR"
echo "alias_dir=$ALIAS_DIR"
echo "ascii_alias_dir=$ASCII_ALIAS_DIR"
echo "alias_public_preview=$ALIAS_PUBLIC_PREVIEW"
echo "ascii_alias_public_preview=$ASCII_ALIAS_PUBLIC_PREVIEW_ASCII"
