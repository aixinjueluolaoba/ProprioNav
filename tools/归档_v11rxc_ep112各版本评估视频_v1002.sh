#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/diana/fishing/blind_nav_rl
REMOTE_ROOT=/home/diana/fishing/blind_nav_rl
MODEL_PATH=${MODEL_PATH:-$REMOTE_ROOT/runs/v11rxc_nlite125_fix6300031_targeted_far6_stuck10_search112/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/models/checkpoint_ep_00112.zip}
MODEL_BASE="${MODEL_PATH%.zip}"
STATE_MODE=${STATE_MODE:-observable12_target8_macro_library_v11rxc_stage2}
VIDEO_EPISODES=${VIDEO_EPISODES:-20}
PRESSURE_EPISODES=${PRESSURE_EPISODES:-20}
NORMAL_EPISODES=${NORMAL_EPISODES:-20}
REMOTE_OUT_ROOT=${REMOTE_OUT_ROOT:-$REMOTE_ROOT/runs/v11rxc_ep112各版本评估视频}
LOCAL_OUT_ROOT=${LOCAL_OUT_ROOT:-$ROOT/runs/v11rxc_ep112各版本评估视频}
SHARE_DIR=${SHARE_DIR:-/home/diana/screencap/file/v11rxc_ep112各版本评估视频}
ALIAS_DIR=${ALIAS_DIR:-/home/diana/screencap/file/v11rxc各版本评估视频}
ASCII_ALIAS_DIR=${ASCII_ALIAS_DIR:-/home/diana/screencap/file/v11rxc-ep112-version-videos}
PUBLIC_FILE_BASE=${PUBLIC_FILE_BASE:-http://47.99.153.111:12346/file}

VERSIONS=(
  "N19|中距抑制宏长链覆写|方案N19_v11rxc中距抑制宏长链覆写评估探针.json|mid_suppressed_override"
  "N20|中距抑制宏长链重定向|方案N20_v11rxc中距抑制宏长链重定向评估探针.json|mid_suppressed_redirect"
  "N21|中距抑制宏6长链放行|方案N21_v11rxc中距抑制宏6长链放行评估探针.json|mid_macro6_passthrough"
  "N22|中距抑制宏6后段放行|方案N22_v11rxc中距抑制宏6后段放行评估探针.json|mid_macro6_late_passthrough"
  "N23|中距覆写加后段放行|方案N23_v11rxc中距覆写加后段放行评估探针.json|mid_override_plus_late_passthrough"
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
copy_file eval_target8_bad_seeds.py
copy_file eval_target8_pressure.py
copy_file eval_target8_sweep.py
for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r _ _ config_name _ <<<"$entry"
  copy_file "configs/v11/$config_name"
done

for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r version_id version_label config_name out_slug <<<"$entry"
  config_remote="$REMOTE_ROOT/configs/v11/$config_name"
  remote_dir="$REMOTE_OUT_ROOT/$version_id"
  local_dir="$LOCAL_OUT_ROOT/$version_id"

  ssh v1002 "mkdir -p '$remote_dir'"
  ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_bad_seeds.py \
    --model '$MODEL_BASE' \
    --out-dir '$remote_dir/坏种子评估' \
    --state-mode '$STATE_MODE' \
    --env-overrides-file '$config_remote' \
    --device cpu \
    --seed-list 6300031,6300058,6300069,6300006"

  ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_pressure.py \
    --model '$MODEL_BASE' \
    --out-dir '$remote_dir/压力评估20' \
    --action-mode v11 \
    --state-mode '$STATE_MODE' \
    --episodes '$PRESSURE_EPISODES' \
    --video-episodes '$VIDEO_EPISODES' \
    --env-overrides-file '$config_remote' \
    --device cpu"

  ssh v1002 "cd '$REMOTE_ROOT' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ML && python eval_target8_sweep.py \
    --train-output-dir '${MODEL_PATH%/models/*}' \
    --model-path '$MODEL_PATH' \
    --out-dir '$remote_dir/普通评估20' \
    --experiments rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2 \
    --episodes '$NORMAL_EPISODES' \
    --video-episodes '$VIDEO_EPISODES' \
    --env-overrides-file '$config_remote' \
    --device cpu"

  mkdir -p "$local_dir"
  copy_back "$remote_dir/坏种子评估/bad_seed_eval_summary.csv" "$local_dir/${version_id}_坏种子汇总.csv"
  copy_back "$remote_dir/压力评估20/pressure_eval_summary.csv" "$local_dir/${version_id}_压力评估汇总.csv"
  copy_back "$remote_dir/普通评估20/target8_sweep_summary.csv" "$local_dir/${version_id}_普通评估汇总.csv"
  copy_back "$remote_dir/压力评估20/pressure_eval_concat.mp4" "$local_dir/${version_id}_压力评估20轮拼接.mp4"
  copy_back "$remote_dir/普通评估20/rppo_medium_lstm128x2_observable12_target8_macro_library_v11rxc_stage2/target8_eval_concat.mp4" "$local_dir/${version_id}_普通评估20轮拼接.mp4"
  cp "$local_dir/${version_id}_坏种子汇总.csv" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_压力评估汇总.csv" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_普通评估汇总.csv" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_压力评估20轮拼接.mp4" "$SHARE_DIR/"
  cp "$local_dir/${version_id}_普通评估20轮拼接.mp4" "$SHARE_DIR/"
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
    normal = next(csv.DictReader((base / f"{version_id}_普通评估汇总.csv").open(encoding="utf-8")))
    pressure = next(csv.DictReader((base / f"{version_id}_压力评估汇总.csv").open(encoding="utf-8")))
    badseed = next(csv.DictReader((base / f"{version_id}_坏种子汇总.csv").open(encoding="utf-8")))
    rows.append(
        {
            "version_id": version_id,
            "label": label,
            "normal_success": float(normal["success_rate"]),
            "normal_score": float(normal["score"]),
            "normal_steps": float(normal["avg_steps"]),
            "normal_collisions": float(normal["avg_collision_count"]),
            "pressure_success": float(pressure["success_rate"]),
            "pressure_steps": float(pressure["avg_steps"]),
            "pressure_collisions": float(pressure["avg_collision_count"]),
            "pressure_angle": float(pressure["avg_abs_angle_error_deg"]),
            "pressure_still_rate": float(pressure["avg_still_step_rate"]),
            "badseed_success": float(badseed["success_rate"]),
            "badseed_steps": float(badseed["avg_steps"]),
            "badseed_collisions": float(badseed["avg_collision_count"]),
            "normal_video": f"{version_id}_普通评估20轮拼接.mp4",
            "pressure_video": f"{version_id}_压力评估20轮拼接.mp4",
            "normal_csv": f"{version_id}_普通评估汇总.csv",
            "pressure_csv": f"{version_id}_压力评估汇总.csv",
            "badseed_csv": f"{version_id}_坏种子汇总.csv",
        }
    )

rows.sort(key=lambda row: (row["pressure_steps"], row["pressure_collisions"], row["version_id"]))

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
        <div><strong>pressure20</strong> {row["pressure_success"]:.2f} / {row["pressure_steps"]:.2f} steps / {row["pressure_collisions"]:.2f} coll / {row["pressure_angle"]:.2f} deg</div>
        <div><strong>normal20</strong> {row["normal_success"]:.2f} / score {row["normal_score"]:.3f} / {row["normal_steps"]:.2f} steps</div>
        <div><strong>badseed4</strong> {row["badseed_success"]:.3f} / {row["badseed_steps"]:.2f} steps / {row["badseed_collisions"]:.2f} coll</div>
      </div>
      <div class="grid">
        <div class="panel">
          <h3>普通评估 20 轮</h3>
          <video controls preload="metadata" src="{html.escape(row["normal_video"])}"></video>
          <a href="{html.escape(row["normal_csv"])}">普通评估汇总 CSV</a>
        </div>
        <div class="panel">
          <h3>压力评估 20 轮</h3>
          <video controls preload="metadata" src="{html.escape(row["pressure_video"])}"></video>
          <a href="{html.escape(row["pressure_csv"])}">压力评估汇总 CSV</a>
        </div>
      </div>
      <p class="links"><a href="{html.escape(row["badseed_csv"])}">坏种子汇总 CSV</a></p>
    </section>
"""
    )

html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v11rxc ep112 各版本评估视频</title>
  <style>
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      background: #ebe7df;
      color: #1f2328;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .subtitle {{
      margin: 0 0 20px;
      color: #5a5f66;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #d8d0c2;
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 10px 28px rgba(73, 56, 28, 0.08);
    }}
    .title-row {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .title-row h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .title-row span {{
      font-size: 14px;
      color: #7b6d54;
    }}
    .metrics {{
      display: grid;
      gap: 6px;
      margin-bottom: 14px;
      color: #33383f;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: #f6f1e7;
      border-radius: 12px;
      padding: 14px;
      border: 1px solid #e5dccb;
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
    a {{
      color: #0b57d0;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .links {{
      margin: 12px 0 0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>v11rxc ep112 各版本评估视频</h1>
    <p class="subtitle">版本范围：N19 / N20 / N21 / N22 / N23。每个版本提供普通评估 20 轮与压力评估 20 轮拼接视频，以及对应汇总 CSV。</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""
(share_dir / "成果预览.html").write_text(html_text, encoding="utf-8")
PY

rm -rf "$ALIAS_DIR" "$ASCII_ALIAS_DIR"
cp -r "$SHARE_DIR" "$ALIAS_DIR"
cp -r "$SHARE_DIR" "$ASCII_ALIAS_DIR"
cp "$SHARE_DIR/成果预览.html" "$ASCII_ALIAS_DIR/preview.html"
cp "$SHARE_DIR/分享清单.json" "$ASCII_ALIAS_DIR/manifest.json"

for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r version_id _ _ _ <<<"$entry"
  cp "$SHARE_DIR/${version_id}_普通评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/${version_id}_normal.mp4"
  cp "$SHARE_DIR/${version_id}_压力评估20轮拼接.mp4" "$ASCII_ALIAS_DIR/${version_id}_pressure.mp4"
  cp "$SHARE_DIR/${version_id}_普通评估汇总.csv" "$ASCII_ALIAS_DIR/${version_id}_normal.csv"
  cp "$SHARE_DIR/${version_id}_压力评估汇总.csv" "$ASCII_ALIAS_DIR/${version_id}_pressure.csv"
  cp "$SHARE_DIR/${version_id}_坏种子汇总.csv" "$ASCII_ALIAS_DIR/${version_id}_badseed.csv"
done

export PUBLIC_FILE_BASE
python3 - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import quote

share_dir = Path("/home/diana/screencap/file/v11rxc_ep112各版本评估视频")
alias_dir = Path("/home/diana/screencap/file/v11rxc各版本评估视频")
ascii_dir = Path("/home/diana/screencap/file/v11rxc-ep112-version-videos")
base = os.environ["PUBLIC_FILE_BASE"].rstrip("/")

def file_url(folder: str, name: str) -> str:
    return f"{base}/{quote(folder + '/' + name)}"

manifest = json.loads((share_dir / "分享清单.json").read_text(encoding="utf-8"))
lines = [
    "# v11rxc ep112 各版本评估视频链接",
    "",
    f"- 长目录预览页：{file_url(share_dir.name, '成果预览.html')}",
    f"- 稳定短目录预览页：{file_url(alias_dir.name, '成果预览.html')}",
    f"- ASCII 预览页：{file_url(ascii_dir.name, 'preview.html')}",
    f"- 长目录 JSON 清单：{file_url(share_dir.name, '分享清单.json')}",
    f"- 稳定短目录 JSON 清单：{file_url(alias_dir.name, '分享清单.json')}",
    f"- ASCII JSON 清单：{file_url(ascii_dir.name, 'manifest.json')}",
    "",
]
for row in manifest["versions"]:
    version_id = row["version_id"]
    lines.append(f"## {version_id} {row['label']}")
    lines.append(f"- 普通评估视频：{file_url(alias_dir.name, row['normal_video'])}")
    lines.append(f"- 压力评估视频：{file_url(alias_dir.name, row['pressure_video'])}")
    lines.append(f"- 坏种子汇总：{file_url(alias_dir.name, row['badseed_csv'])}")
    lines.append("")

(alias_dir / "链接清单.md").write_text("\n".join(lines), encoding="utf-8")
(ascii_dir / "links.md").write_text("\n".join(lines), encoding="utf-8")
PY

echo "share_dir=$SHARE_DIR"
echo "preview_url=$PUBLIC_FILE_BASE/$(python3 - <<'PY'
from urllib.parse import quote
print(quote('v11rxc各版本评估视频/成果预览.html'))
PY
)"
