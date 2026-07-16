from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--specialist-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")

    base_model = RecurrentPPO.load(args.base_model, device=args.device)
    specialist_model = RecurrentPPO.load(args.specialist_model, device=args.device)

    base_state = base_model.policy.state_dict()
    specialist_state = specialist_model.policy.state_dict()
    merged_state: dict[str, torch.Tensor] = {}
    matched = 0
    skipped: list[str] = []

    for key, base_value in base_state.items():
        specialist_value = specialist_state.get(key)
        if specialist_value is None or specialist_value.shape != base_value.shape:
            merged_state[key] = base_value.detach().clone()
            skipped.append(key)
            continue
        base_float = base_value.detach().float()
        specialist_float = specialist_value.detach().float()
        merged = (1.0 - alpha) * base_float + alpha * specialist_float
        merged_state[key] = merged.to(device=base_value.device, dtype=base_value.dtype)
        matched += 1

    base_model.policy.load_state_dict(merged_state, strict=False)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    base_model.save(str(output_model.with_suffix("")))

    print(f"alpha={alpha}", flush=True)
    print(f"matched_tensors={matched}", flush=True)
    print(f"skipped_tensors={len(skipped)}", flush=True)
    if skipped:
        print("skipped_keys=" + ",".join(skipped), flush=True)
    print(f"saved_model={output_model}", flush=True)


if __name__ == "__main__":
    main()
