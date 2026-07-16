from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_EXPERIMENT = "rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2"
DEFAULT_STATE_MODE = "observable12_target8_macro_library_v11_stage2"
CURRICULUM_STAGES = [
    (
        "stage1",
        "rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage1",
        "observable12_target8_macro_library_v11_stage1",
        20_000,
    ),
    (
        "stage2a",
        "rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2a",
        "observable12_target8_macro_library_v11_stage2a",
        10_000,
    ),
    (
        "stage2",
        "rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage2",
        "observable12_target8_macro_library_v11_stage2",
        30_000,
    ),
    (
        "stage3",
        "rppo_medium_lstm128x2_observable12_target8_macro_library_v11_stage3",
        "observable12_target8_macro_library_v11_stage3",
        50_000,
    ),
]


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "stage",
        "checkpoint",
        "normal_success",
        "normal_avg_steps",
        "normal_path_eff",
        "normal_avg_angle",
        "normal_final_angle",
        "normal_last5_angle",
        "normal_recovery_rate",
        "pressure_success",
        "pressure_avg_steps",
        "pressure_path_eff",
        "pressure_avg_angle",
        "pressure_final_angle",
        "pressure_last5_angle",
        "pressure_recovery_rate",
        "pressure_avg_recovery_steps_to_trigger",
        "pressure_recovery_escape_rate",
        "pressure_avg_collisions",
        "normal_summary",
        "pressure_summary",
        "model",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_seen(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stage = (row.get("stage") or "").strip()
            checkpoint = (row.get("checkpoint") or "").strip()
            if stage and checkpoint:
                seen.add((stage, checkpoint))
    return seen


def evaluate_checkpoint(
    model_path: Path,
    out_dir: Path,
    checkpoint: str,
    python: str,
    log_path: Path,
    experiment: str,
    state_mode: str,
) -> dict[str, object]:
    run_dir = model_path.parents[2]
    env_overrides = run_dir / "env_overrides.json"
    normal_dir = out_dir / f"{checkpoint}_普通评估20"
    pressure_dir = out_dir / f"{checkpoint}_压力评估20"
    normal_cmd = [
        python,
        "eval_target8_sweep.py",
        "--train-output-dir",
        str(run_dir),
        "--model-path",
        str(model_path),
        "--out-dir",
        str(normal_dir),
        "--experiments",
        experiment,
        "--episodes",
        "20",
        "--video-episodes",
        "0",
    ]
    pressure_cmd = [
        python,
        "eval_target8_pressure.py",
        "--model",
        str(model_path),
        "--out-dir",
        str(pressure_dir),
        "--action-mode",
        "v11",
        "--state-mode",
        state_mode,
        "--episodes",
        "20",
        "--video-episodes",
        "0",
    ]
    if env_overrides.exists():
        normal_cmd.extend(["--env-overrides-file", str(env_overrides)])
        pressure_cmd.extend(["--env-overrides-file", str(env_overrides)])
    run(normal_cmd, log_path)
    run(pressure_cmd, log_path)
    normal = read_summary(normal_dir / "target8_sweep_summary.csv")
    pressure = read_summary(pressure_dir / "pressure_eval_summary.csv")
    return {
        "stage": "",
        "checkpoint": checkpoint,
        "normal_success": normal.get("success_rate"),
        "normal_avg_steps": normal.get("avg_steps"),
        "normal_path_eff": normal.get("avg_path_efficiency"),
        "normal_avg_angle": normal.get("avg_abs_angle_error_deg"),
        "normal_final_angle": normal.get("avg_final_abs_angle_error_deg"),
        "normal_last5_angle": normal.get("avg_last5_abs_angle_error_deg"),
        "normal_recovery_rate": normal.get("avg_recovery_action_rate"),
        "pressure_success": pressure.get("success_rate"),
        "pressure_avg_steps": pressure.get("avg_steps"),
        "pressure_path_eff": pressure.get("avg_path_efficiency"),
        "pressure_avg_angle": pressure.get("avg_abs_angle_error_deg"),
        "pressure_final_angle": pressure.get("avg_final_abs_angle_error_deg"),
        "pressure_last5_angle": pressure.get("avg_last5_abs_angle_error_deg"),
        "pressure_recovery_rate": pressure.get("avg_recovery_action_rate"),
        "pressure_avg_recovery_steps_to_trigger": pressure.get("avg_recovery_steps_to_trigger"),
        "pressure_recovery_escape_rate": pressure.get("recovery_escape_rate"),
        "pressure_avg_collisions": pressure.get("avg_collision_count"),
        "normal_summary": str(normal_dir / "target8_sweep_summary.csv"),
        "pressure_summary": str(pressure_dir / "pressure_eval_summary.csv"),
        "model": str(model_path),
    }


def curriculum_entries(args: argparse.Namespace) -> list[tuple[str, str, str, int]]:
    if not args.curriculum:
        return [("", args.experiment, args.state_mode, args.max_checkpoint)]
    stage_limits = [
        int(args.stage1_max_checkpoint or CURRICULUM_STAGES[0][3]),
        int(args.stage2a_max_checkpoint or CURRICULUM_STAGES[1][3]),
        int(args.stage2_max_checkpoint or CURRICULUM_STAGES[2][3]),
        int(args.stage3_max_checkpoint or CURRICULUM_STAGES[3][3]),
    ]
    entries = [
        (stage_name, experiment, state_mode, max_checkpoint)
        for (stage_name, experiment, state_mode, _), max_checkpoint in zip(CURRICULUM_STAGES, stage_limits)
    ]
    stage2_model_dir = Path(args.run_dir) / DEFAULT_EXPERIMENT / "models"
    if stage2_model_dir.exists():
        has_any = any(stage2_model_dir.glob("checkpoint_ep_*.zip"))
        has_stage1 = (Path(args.run_dir) / CURRICULUM_STAGES[0][1] / "models").exists()
        if has_any and not has_stage1:
            return [entries[2]]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-checkpoint", type=int, default=50000)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--state-mode", default=DEFAULT_STATE_MODE)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--stage1-max-checkpoint", type=int, default=None)
    parser.add_argument("--stage2a-max-checkpoint", type=int, default=None)
    parser.add_argument("--stage2-max-checkpoint", type=int, default=None)
    parser.add_argument("--stage3-max-checkpoint", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    result_csv = out_dir / "checkpoint_eval_summary.csv"
    log_path = out_dir / "watcher.log"
    seen = load_seen(result_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        stage_done = 0
        entries = curriculum_entries(args)
        for stage_name, experiment, state_mode, max_checkpoint in entries:
            model_dir = run_dir / experiment / "models"
            checkpoints = sorted(model_dir.glob("checkpoint_ep_*.zip"))
            for model_path in checkpoints:
                checkpoint = model_path.stem.replace("checkpoint_ep_", "")
                checkpoint_value = int(checkpoint)
                if checkpoint_value % args.checkpoint_step != 0:
                    continue
                seen_key = (stage_name or experiment, checkpoint)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                row = evaluate_checkpoint(
                    model_path,
                    out_dir / (stage_name or experiment),
                    checkpoint,
                    args.python,
                    log_path,
                    experiment,
                    state_mode,
                )
                row["stage"] = stage_name or experiment
                append_result(result_csv, row)
                print(f"evaluated stage={stage_name or experiment} checkpoint={checkpoint}", flush=True)
            if any(key[0] == (stage_name or experiment) and int(key[1]) >= max_checkpoint for key in seen):
                stage_done += 1
        if stage_done >= len(entries):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
