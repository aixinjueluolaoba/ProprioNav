#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
MODEL_PATH=${MODEL_PATH:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
MODEL_BASE="${MODEL_PATH%.zip}"
STATE_MODE=${STATE_MODE:-observable12_target8_macro_library_v11rxc_stage2}
DIAGONAL_EPISODES=${DIAGONAL_EPISODES:-20}
VIDEO_EPISODES=${VIDEO_EPISODES:-20}
SEED_START=${SEED_START:-6300000}
REMOTE_OUT_ROOT=${REMOTE_OUT_ROOT:-$REMOTE_ROOT/runs/v11rxc_ep112各版本对角高障碍视频}
LOCAL_OUT_ROOT=${LOCAL_OUT_ROOT:-$ROOT/runs/v11rxc_ep112各版本对角高障碍视频}
SHARE_DIR=${SHARE_DIR:-/home/diana/screencap/file/v11rxc各版本对角高障碍视频}
ASCII_ALIAS_DIR=${ASCII_ALIAS_DIR:-/home/diana/screencap/file/v11rxc-ep112-diagonal-dense-videos}
PUBLIC_FILE_BASE=${PUBLIC_FILE_BASE:-http://47.99.153.111:12346/file}

VERSIONS=(
  "N19|中距抑制宏长链覆写|方案N19_v11rxc中距抑制宏长链覆写评估探针.json"
  "N20|中距抑制宏长链重定向|方案N20_v11rxc中距抑制宏长链重定向评估探针.json"
  "N21|中距抑制宏6长链放行|方案N21_v11rxc中距抑制宏6长链放行评估探针.json"
  "N22|中距抑制宏6后段放行|方案N22_v11rxc中距抑制宏6后段放行评估探针.json"
  "N23|中距覆写加后段放行|方案N23_v11rxc中距覆写加后段放行评估探针.json"
)

copy_file() {
  local rel="$1"
  ssh v1002 "mkdir -p '$REMOTE_ROOT/$(dirname "$rel")'"
  scp "$ROOT/$rel" "v1002:$REMOTE_ROOT/$rel"
}

copy_back() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  scp "v1002:$remote_path" "$local_path"
}

mkdir -p "$LOCAL_OUT_ROOT" "$SHARE_DIR"

copy_file benchmark_state_dims_10k.py
copy_file blind_nav_rl/env.py
copy_file eval_target8_diagonal_dense_video.py
copy_file eval_target8_sweep.py
for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r _ _ config_name <<<"$entry"
  copy_file "configs/v11/$config_name"
done

for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r version_id version_label config_name <<<"$entry"
  config_remote="$REMOTE_ROOT/configs/v11/$config_name"
  remote_dir="$REMOTE_OUT_ROOT/$version_id"
  local_dir="$LOCAL_OUT_ROOT/$version_id"

  ssh v1002 "mkdir -p '$remote_dir'"
  ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_diagonal_dense_video.py \
    --model '$MODEL_BASE' \
    --out-dir '$remote_dir/对角高障碍20' \
    --episodes '$DIAGONAL_EPISODES' \
    --video-episodes '$VIDEO_EPISODES' \
    --seed-start '$SEED_START' \
    --state-mode '$STATE_MODE' \
    --env-overrides-file '$config_remote' \
    --device cpu"

  mkdir -p "$local_dir"
  copy_back "$remote_dir/对角高障碍20/diagonal_dense_eval_summary.csv" "$local_dir/${version_id}_对角高障碍汇总.csv"
  copy_back "$remote_dir/对角高障碍20/diagonal_dense_eval_results.csv" "$local_dir/${version_id}_对角高障碍明细.csv"
  copy_back "$remote_dir/对角高障碍20/diagonal_dense_eval_concat.mp4" "$local_dir/${version_id}_对角高障碍20轮拼接.mp4"
  cp "$local_dir/${version_id}_对角高障碍汇总.csv" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_对角高障碍明细.csv" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_对角高障碍20轮拼接.mp4" "$SHARE_DIR/"
  echo "finished=$version_id label=$version_label"
done

export LOCAL_OUT_ROOT SHARE_DIR
python3 - <<'PY'
import csv
import html
import json
import os
from pathlib import Path

local_root = Path(os.environ["LOCAL_OUT_ROOT"])
share_dir = Path(os.environ["SHARE_DIR"])
versions = [
    ("N19", "中距抑制宏长链覆写"),
    ("N20", "中距抑制宏长链重定向"),
    ("N21", "中距抑制宏6长链放行"),
    ("N22", "中距抑制宏6后段放行"),
    ("N23", "中距覆写加后段放行"),
]
rows = []
for version_id, label in versions:
    base = local_root / version_id
    summary = next(csv.DictReader((base / f"{version_id}_对角高障碍汇总.csv").open(encoding="utf-8")))
    rows.append(
        {
            "version_id": version_id,
            "label": label,
            "success_rate": float(summary["success_rate"]),
            "avg_steps": float(summary["avg_steps"]),
            "avg_collisions": float(summary["avg_collision_count"]),
            "avg_angle": float(summary["avg_abs_angle_error_deg"]),
            "avg_still_rate": float(summary["avg_still_step_rate"]),
            "avg_max_still_run": float(summary["avg_max_still_run"]),
            "video": f"{version_id}_对角高障碍20轮拼接.mp4",
            "summary_csv": f"{version_id}_对角高障碍汇总.csv",
            "detail_csv": f"{version_id}_对角高障碍明细.csv",
        }
    )

rows.sort(key=lambda row: (-row["success_rate"], row["avg_collisions"], row["avg_steps"], row["version_id"]))
manifest = {"versions": rows}
(share_dir / "分享清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

cards = []
for row in rows:
    cards.append(
        f"""
    <section class="card">
      <div class="title-row">
        <h2>{html.escape(row["version_id"])} <span>{html.escape(row["label"])}</span></h2>
      </div>
      <div class="metrics">
        <div><strong>成功率</strong> {row["success_rate"]:.2f}</div>
        <div><strong>平均步数</strong> {row["avg_steps"]:.2f}</div>
        <div><strong>平均碰撞</strong> {row["avg_collisions"]:.2f}</div>
        <div><strong>平均角度误差</strong> {row["avg_angle"]:.2f} deg</div>
        <div><strong>静止率</strong> {row["avg_still_rate"]:.4f}</div>
        <div><strong>最长连续静止</strong> {row["avg_max_still_run"]:.2f}</div>
      </div>
      <div class="panel">
        <h3>对角高障碍 20 轮</h3>
        <video controls preload="metadata" src="{html.escape(row["video"])}"></video>
        <div class="links">
          <a href="{html.escape(row["summary_csv"])}">汇总 CSV</a>
          <a href="{html.escape(row["detail_csv"])}">明细 CSV</a>
        </div>
      </div>
    </section>
"""
    )

html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v11rxc 各版本对角高障碍视频</title>
  <style>
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      background: #ece8df;
      color: #1f2328;
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .subtitle {{
      margin: 0 0 20px;
      color: #5e625d;
    }}
    .card {{
      background: #fffdf9;
      border: 1px solid #d7d1c6;
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(66, 50, 24, 0.08);
    }}
    .title-row {{
      margin-bottom: 10px;
    }}
    .title-row h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .title-row span {{
      font-size: 14px;
      color: #7b6f5e;
      margin-left: 8px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px 14px;
      margin-bottom: 14px;
    }}
    .panel {{
      background: #f6f1e8;
      border: 1px solid #e4dbc9;
      border-radius: 12px;
      padding: 14px;
    }}
    .panel h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    video {{
      width: 100%;
      border-radius: 10px;
      background: #000;
      margin-bottom: 10px;
    }}
    .links {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    a {{
      color: #0b57d0;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <h1>v11rxc 各版本对角高障碍视频</h1>
    <p class="subtitle">版本范围：N19 / N20 / N21 / N22 / N23。固定远起点与远目标，中间高密度树木和山体，每个版本 20 轮拼接视频。</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""
(share_dir / "成果预览.html").write_text(html_text, encoding="utf-8")
PY

rm -rf "$ASCII_ALIAS_DIR"
cp -r "$SHARE_DIR" "$ASCII_ALIAS_DIR"
cp "$SHARE_DIR/成果预览.html" "$ASCII_ALIAS_DIR/preview.html"
cp "$SHARE_DIR/分享清单.json" "$ASCII_ALIAS_DIR/manifest.json"

for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r version_id _ _ <<<"$entry"
  cp "$SHARE_DIR/${version_id}_对角高障碍20轮拼接.mp4" "$ASCII_ALIAS_DIR/${version_id}_diagonal.mp4"
  cp "$SHARE_DIR/${version_id}_对角高障碍汇总.csv" "$ASCII_ALIAS_DIR/${version_id}_diagonal_summary.csv"
  cp "$SHARE_DIR/${version_id}_对角高障碍明细.csv" "$ASCII_ALIAS_DIR/${version_id}_diagonal_detail.csv"
done

export PUBLIC_FILE_BASE
python3 - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import quote

share_dir = Path("/home/diana/screencap/file/v11rxc各版本对角高障碍视频")
ascii_dir = Path("/home/diana/screencap/file/v11rxc-ep112-diagonal-dense-videos")
base = os.environ["PUBLIC_FILE_BASE"].rstrip("/")

def file_url(folder: str, name: str) -> str:
    return f"{base}/{quote(folder + '/' + name)}"

manifest = json.loads((share_dir / "分享清单.json").read_text(encoding="utf-8"))
lines = [
    "# v11rxc 各版本对角高障碍视频链接",
    "",
    f"- 预览页：{file_url(share_dir.name, '成果预览.html')}",
    f"- ASCII 预览页：{file_url(ascii_dir.name, 'preview.html')}",
    "",
]
for row in manifest["versions"]:
    version_id = row["version_id"]
    lines.append(f"## {version_id} {row['label']}")
    lines.append(f"- 对角高障碍视频：{file_url(share_dir.name, row['video'])}")
    lines.append(f"- 汇总 CSV：{file_url(share_dir.name, row['summary_csv'])}")
    lines.append(f"- 明细 CSV：{file_url(share_dir.name, row['detail_csv'])}")
    lines.append("")

(share_dir / "链接清单.md").write_text("\n".join(lines), encoding="utf-8")
(ascii_dir / "links.md").write_text("\n".join(lines), encoding="utf-8")
PY

echo "share_dir=$SHARE_DIR"
echo "preview_url=$PUBLIC_FILE_BASE/$(python3 - <<'PY'
from urllib.parse import quote
print(quote('v11rxc各版本对角高障碍视频/成果预览.html'))
PY
)"
