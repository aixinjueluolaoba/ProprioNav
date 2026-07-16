from __future__ import annotations

import argparse
import time
from pathlib import Path

from watch_v11_checkpoints import append_result, evaluate_checkpoint, load_seen


DEFAULT_EXPERIMENT = "rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2"
DEFAULT_STATE_MODE = "observable12_target8_macro_library_v11re_stage2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--python", default="python")
    parser.add_argument("--max-checkpoint", type=int, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--state-mode", default=DEFAULT_STATE_MODE)
    args = parser.parse_args()

    if args.tiny:
        max_checkpoint = int(args.max_checkpoint or 64)
        checkpoint_step = int(args.checkpoint_step or 32)
    elif args.micro:
        max_checkpoint = int(args.max_checkpoint or 250)
        checkpoint_step = int(args.checkpoint_step or 125)
    else:
        max_checkpoint = int(args.max_checkpoint or (500 if args.quick else 1_000))
        checkpoint_step = int(args.checkpoint_step or (250 if args.quick else 500))

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    result_csv = out_dir / "checkpoint_eval_summary.csv"
    log_path = out_dir / "watcher.log"
    seen = load_seen(result_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        model_dir = run_dir / args.experiment / "models"
        checkpoints = sorted(model_dir.glob("checkpoint_ep_*.zip"))
        for model_path in checkpoints:
            checkpoint = model_path.stem.replace("checkpoint_ep_", "")
            checkpoint_value = int(checkpoint)
            if checkpoint_value % checkpoint_step != 0:
                continue
            seen_key = ("stage2", checkpoint)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            row = evaluate_checkpoint(
                model_path,
                out_dir / "stage2",
                checkpoint,
                args.python,
                log_path,
                args.experiment,
                args.state_mode,
            )
            row["stage"] = "stage2"
            append_result(result_csv, row)
            print(f"evaluated stage=stage2 checkpoint={checkpoint}", flush=True)
        if any(key[0] == "stage2" and int(key[1]) >= max_checkpoint for key in seen):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
