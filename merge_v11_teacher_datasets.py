from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_mapping(values: list[str], flag_name: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag_name} entry must be label=path, got: {value}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"{flag_name} entry has empty label: {value}")
        mapping[label] = Path(raw_path.strip())
    return mapping


def best_row_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    success = float(row.get("success", 0.0) or 0.0)
    steps = float(row.get("steps", 1e9) or 1e9)
    collisions = float(row.get("collisions", 1e9) or 1e9)
    final_distance = float(row.get("final_distance", 1e9) or 1e9)
    return (success, -steps, -collisions, -final_distance)


def load_result_rows(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["seed"]): row for row in csv.DictReader(path.open(encoding="utf-8"))}


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def take_seed_rows(dataset: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    seed_mask = np.asarray(dataset["seeds"], dtype=np.int64) == int(seed)
    if not np.any(seed_mask):
        raise KeyError(f"seed {seed} not found in dataset")
    return {key: np.asarray(value)[seed_mask] for key, value in dataset.items()}


def concat_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("no parts to concatenate")
    keys = list(parts[0].keys())
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="label=dataset.npz")
    parser.add_argument("--results", action="append", required=True, help="label=bad_seed_eval_results.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-out", default=None)
    args = parser.parse_args()

    dataset_paths = parse_mapping(args.dataset, "--dataset")
    results_paths = parse_mapping(args.results, "--results")
    if set(dataset_paths) != set(results_paths):
        raise ValueError("dataset labels and result labels must match exactly")

    datasets = {label: load_dataset(path) for label, path in dataset_paths.items()}
    result_rows = {label: load_result_rows(path) for label, path in results_paths.items()}
    seed_set = sorted({seed for rows in result_rows.values() for seed in rows})
    selections: list[dict[str, object]] = []
    merged_parts: list[dict[str, np.ndarray]] = []

    for seed in seed_set:
        candidates: list[tuple[tuple[float, float, float, float], str, dict[str, str]]] = []
        for label, rows in result_rows.items():
            row = rows.get(seed)
            if row is None:
                continue
            candidates.append((best_row_key(row), label, row))
        if not candidates:
            raise ValueError(f"no candidates found for seed {seed}")
        _, best_label, best_row = max(candidates, key=lambda item: item[0])
        merged_parts.append(take_seed_rows(datasets[best_label], seed))
        selections.append(
            {
                "seed": seed,
                "source": best_label,
                "success": best_row.get("success", ""),
                "steps": best_row.get("steps", ""),
                "collisions": best_row.get("collisions", ""),
                "final_distance": best_row.get("final_distance", ""),
            }
        )

    merged = concat_parts(merged_parts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)

    selection_out = Path(args.selection_out) if args.selection_out else out_path.with_suffix(".selection.csv")
    selection_out.parent.mkdir(parents=True, exist_ok=True)
    with selection_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "source", "success", "steps", "collisions", "final_distance"])
        writer.writeheader()
        writer.writerows(selections)

    source_counts: dict[str, int] = {}
    for row in selections:
        source = str(row["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"saved_dataset={out_path}", flush=True)
    print(f"saved_selection={selection_out}", flush=True)
    print(f"episodes={len(selections)}", flush=True)
    print("source_counts=" + ",".join(f"{label}:{count}" for label, count in sorted(source_counts.items())), flush=True)


if __name__ == "__main__":
    main()
