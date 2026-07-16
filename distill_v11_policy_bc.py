from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from benchmark_state_dims_10k import Experiment, N_ENVS, make_env, make_model


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11re_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11re_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def make_zero_states(model: RecurrentPPO, batch_size: int, device: torch.device) -> RNNStates:
    shape = model.policy.lstm_hidden_state_shape
    zeros = torch.zeros(shape[0], batch_size, shape[2], device=device)
    return RNNStates((zeros.clone(), zeros.clone()), (zeros.clone(), zeros.clone()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dataset", default=None)
    parser.add_argument("--teacher-datasets", nargs="*", default=None)
    parser.add_argument("--resume-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--recovery-weight", type=float, default=4.0)
    parser.add_argument("--stuck-weight", type=float, default=2.0)
    parser.add_argument("--collision-weight", type=float, default=2.0)
    parser.add_argument("--stuck-no-progress-threshold", type=float, default=0.8)
    parser.add_argument("--stuck-time-threshold", type=float, default=0.5)
    parser.add_argument("--dataset-weights", default=None)
    parser.add_argument("--dataset-roles", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_paths = list(args.teacher_datasets or [])
    if args.teacher_dataset:
        dataset_paths.insert(0, args.teacher_dataset)
    if not dataset_paths:
        raise ValueError("Provide --teacher-dataset or --teacher-datasets")
    dataset_weights = [1.0] * len(dataset_paths)
    if args.dataset_weights:
        parsed = [float(value.strip()) for value in args.dataset_weights.split(",") if value.strip()]
        if len(parsed) != len(dataset_paths):
            raise ValueError("dataset_weights count must match teacher datasets")
        dataset_weights = parsed
    dataset_roles = ["generic"] * len(dataset_paths)
    if args.dataset_roles:
        parsed = [value.strip() for value in args.dataset_roles.split(",") if value.strip()]
        if len(parsed) != len(dataset_paths):
            raise ValueError("dataset_roles count must match teacher datasets")
        dataset_roles = parsed

    obs_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    episode_start_parts: list[np.ndarray] = []
    no_progress_parts: list[np.ndarray] = []
    stuck_time_parts: list[np.ndarray] = []
    collided_parts: list[np.ndarray] = []
    recovery_mode_parts: list[np.ndarray] = []
    dataset_weight_parts: list[np.ndarray] = []
    for dataset_path, dataset_weight, dataset_role in zip(dataset_paths, dataset_weights, dataset_roles, strict=True):
        data = np.load(dataset_path)
        mask = np.ones(len(data["obs"]), dtype=bool)
        if args.success_only and "teacher_success" in data:
            mask &= np.asarray(data["teacher_success"], dtype=bool)
        no_progress_raw = np.asarray(data["no_progress_time"], dtype=np.float32)
        stuck_time_raw = np.asarray(data["stuck_time"], dtype=np.float32)
        collided_raw = np.asarray(data["collided"], dtype=bool)
        recovery_mode_raw = np.asarray(data["recovery_mode"], dtype=np.int64)
        stuck_like = (no_progress_raw >= float(args.stuck_no_progress_threshold)) | (
            stuck_time_raw >= float(args.stuck_time_threshold)
        )
        recovery_like = recovery_mode_raw != 0
        if dataset_role == "specialist":
            mask &= stuck_like | recovery_like
        elif dataset_role == "base":
            mask &= ~(stuck_like | recovery_like | collided_raw)
        obs_parts.append(np.asarray(data["obs"], dtype=np.float32)[mask])
        actions_parts.append(np.asarray(data["actions"], dtype=np.int64)[mask])
        episode_start_parts.append(np.asarray(data["episode_starts"], dtype=np.float32)[mask])
        no_progress_parts.append(no_progress_raw[mask])
        stuck_time_parts.append(stuck_time_raw[mask])
        collided_parts.append(collided_raw[mask])
        recovery_mode_parts.append(recovery_mode_raw[mask])
        dataset_weight_parts.append(np.full(int(mask.sum()), float(dataset_weight), dtype=np.float32))

    obs_np = np.concatenate(obs_parts, axis=0)
    actions_np = np.concatenate(actions_parts, axis=0)
    episode_starts_np = np.concatenate(episode_start_parts, axis=0)
    no_progress_np = np.concatenate(no_progress_parts, axis=0)
    stuck_time_np = np.concatenate(stuck_time_parts, axis=0)
    collided_np = np.concatenate(collided_parts, axis=0)
    recovery_mode_np = np.concatenate(recovery_mode_parts, axis=0)
    dataset_weight_np = np.concatenate(dataset_weight_parts, axis=0)

    if args.max_samples is not None:
        limit = min(int(args.max_samples), int(obs_np.shape[0]))
        obs_np = obs_np[:limit]
        actions_np = actions_np[:limit]
        episode_starts_np = episode_starts_np[:limit]
        no_progress_np = no_progress_np[:limit]
        stuck_time_np = stuck_time_np[:limit]
        collided_np = collided_np[:limit]
        recovery_mode_np = recovery_mode_np[:limit]
        dataset_weight_np = dataset_weight_np[:limit]

    weights_np = dataset_weight_np.copy()
    recovery_active = recovery_mode_np != 0
    stuck_active = (no_progress_np >= float(args.stuck_no_progress_threshold)) | (
        stuck_time_np >= float(args.stuck_time_threshold)
    )
    weights_np[recovery_active] *= float(args.recovery_weight)
    weights_np[stuck_active] *= float(args.stuck_weight)
    weights_np[collided_np] *= float(args.collision_weight)

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)
    episode_starts = torch.as_tensor(episode_starts_np, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    env = __import__("stable_baselines3.common.vec_env", fromlist=["SubprocVecEnv"]).SubprocVecEnv(
        [make_env(12_345_000, rank, STAGE.state_mode) for rank in range(1)],
        start_method="fork",
    )
    model = make_model(STAGE, env, 12_345_000, args.device)
    source = RecurrentPPO.load(args.resume_model, device=args.device)
    model.policy.load_state_dict(source.policy.state_dict(), strict=False)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(args.learning_rate))
    batch_states = make_zero_states(model, 1, device)
    total_samples = int(obs.shape[0])
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        values, log_prob, entropy = model.policy.evaluate_actions(
            obs,
            actions,
            batch_states,
            episode_starts,
        )
        del values
        weighted_nll = -(log_prob * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
        loss = weighted_nll - 1e-3 * entropy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        print(
            "epoch={epoch} samples={samples} loss={loss:.6f} avg_weight={avg_weight:.4f} recovery_frac={recovery_frac:.4f} stuck_frac={stuck_frac:.4f}".format(
                epoch=epoch + 1,
                samples=total_samples,
                loss=float(loss.detach().cpu()),
                avg_weight=float(sample_weights.mean().detach().cpu()),
                recovery_frac=float(torch.as_tensor(recovery_active, dtype=torch.float32).mean().cpu()),
                stuck_frac=float(torch.as_tensor(stuck_active, dtype=torch.float32).mean().cpu()),
            ),
            flush=True,
        )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model.with_suffix("")))
    env.close()
    print(f"saved_model={output_model}", flush=True)


if __name__ == "__main__":
    main()
