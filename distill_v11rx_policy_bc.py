from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from benchmark_state_dims_10k import Experiment, make_env, make_model


STAGE = Experiment(
    name="rppo_medium_lstm128x2_observable12_target8_macro_library_v11rx_stage2",
    algo="RecurrentPPO",
    policy="MlpLstmPolicy",
    net_arch={"pi": [128, 128], "vf": [128, 128]},
    state_mode="observable12_target8_macro_library_v11rx_stage2",
    lstm_hidden_size=128,
    n_lstm_layers=2,
)


def make_zero_states(model: RecurrentPPO, batch_size: int, device: torch.device) -> RNNStates:
    shape = model.policy.lstm_hidden_state_shape
    zeros = torch.zeros(shape[0], batch_size, shape[2], device=device)
    return RNNStates((zeros.clone(), zeros.clone()), (zeros.clone(), zeros.clone()))


def warm_start_policy(model: RecurrentPPO, resume_model: str, device: str) -> int:
    source = RecurrentPPO.load(resume_model, device=device)
    target_state = model.policy.state_dict()
    source_state = source.policy.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value.detach().clone()
    target_state.update(matched)
    model.policy.load_state_dict(target_state, strict=False)
    return len(matched)


def map_actions_to_v11rx(actions: np.ndarray, action_format: str, remap_macro4_to_7: bool) -> np.ndarray:
    actions_np = np.asarray(actions, dtype=np.int64)
    if actions_np.ndim != 2 or actions_np.shape[1] != 3:
        raise ValueError(f"expected actions with shape [N,3], got {actions_np.shape}")
    mapped = actions_np.copy()
    mapped[:, 0] = np.clip(mapped[:, 0], 0, 4)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, 1)
    if action_format == "v11rx":
        mapped[:, 2] = np.clip(mapped[:, 2], 0, 7)
    elif action_format == "v11re":
        mapped[:, 2] = np.clip(mapped[:, 2], 0, 6)
        if remap_macro4_to_7:
            mapped[:, 2] = np.where(mapped[:, 2] == 4, 7, mapped[:, 2])
    else:
        raise ValueError(f"unsupported action format: {action_format}")
    return mapped


def parse_loss_head_mode(mode: str) -> np.ndarray:
    normalized = mode.strip().lower()
    mapping = {
        "all": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "macro": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "macro_only": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "speed_macro": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        "direction_macro": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "dir_speed": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "direction_speed": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported loss head mode: {mode}")
    return mapping[normalized]


def macro_head_slice(model: RecurrentPPO) -> tuple[int, int]:
    action_dims = getattr(model.policy.action_dist, "action_dims", None)
    if action_dims is None or len(action_dims) != 3:
        raise ValueError(f"expected 3-head MultiCategorical action dims, got {action_dims}")
    start = int(action_dims[0]) + int(action_dims[1])
    end = start + int(action_dims[2])
    return start, end


def configure_trainable_params(model: RecurrentPPO, mode: str) -> tuple[str, tuple[int, int] | None, int]:
    normalized = mode.strip().lower()
    for param in model.policy.parameters():
        param.requires_grad_(False)
    macro_slice: tuple[int, int] | None = None
    if normalized == "all":
        for param in model.policy.parameters():
            param.requires_grad_(True)
    elif normalized == "action_net_only":
        model.policy.action_net.weight.requires_grad_(True)
        if model.policy.action_net.bias is not None:
            model.policy.action_net.bias.requires_grad_(True)
    elif normalized == "macro_slice_only":
        model.policy.action_net.weight.requires_grad_(True)
        if model.policy.action_net.bias is not None:
            model.policy.action_net.bias.requires_grad_(True)
        macro_slice = macro_head_slice(model)
    else:
        raise ValueError(f"unsupported trainable param mode: {mode}")
    trainable = sum(param.numel() for param in model.policy.parameters() if param.requires_grad)
    return normalized, macro_slice, int(trainable)


def apply_gradient_mask_for_mode(
    model: RecurrentPPO,
    trainable_param_mode: str,
    macro_slice: tuple[int, int] | None,
) -> None:
    if trainable_param_mode != "macro_slice_only":
        return
    if macro_slice is None:
        raise RuntimeError("macro_slice is required for macro_slice_only mode")
    start, end = macro_slice
    weight_grad = model.policy.action_net.weight.grad
    if weight_grad is not None:
        weight_grad[:start].zero_()
        weight_grad[end:].zero_()
    bias_grad = model.policy.action_net.bias.grad
    if bias_grad is not None:
        bias_grad[:start].zero_()
        bias_grad[end:].zero_()


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
    parser.add_argument("--dataset-remap-macro4-to-7", default=None)
    parser.add_argument("--dataset-action-formats", default=None)
    parser.add_argument("--dataset-loss-heads", default=None)
    parser.add_argument("--trainable-param-mode", default="all")
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

    dataset_remap_flags = [False] * len(dataset_paths)
    if args.dataset_remap_macro4_to_7:
        parsed = [value.strip() for value in args.dataset_remap_macro4_to_7.split(",") if value.strip()]
        if len(parsed) != len(dataset_paths):
            raise ValueError("dataset_remap_macro4_to_7 count must match teacher datasets")
        dataset_remap_flags = [value in {"1", "true", "True", "yes", "y"} for value in parsed]

    dataset_action_formats = ["v11re"] * len(dataset_paths)
    if args.dataset_action_formats:
        parsed = [value.strip() for value in args.dataset_action_formats.split(",") if value.strip()]
        if len(parsed) != len(dataset_paths):
            raise ValueError("dataset_action_formats count must match teacher datasets")
        dataset_action_formats = parsed

    dataset_loss_heads = ["all"] * len(dataset_paths)
    if args.dataset_loss_heads:
        parsed = [value.strip() for value in args.dataset_loss_heads.split(",") if value.strip()]
        if len(parsed) != len(dataset_paths):
            raise ValueError("dataset_loss_heads count must match teacher datasets")
        dataset_loss_heads = parsed

    obs_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    episode_start_parts: list[np.ndarray] = []
    no_progress_parts: list[np.ndarray] = []
    stuck_time_parts: list[np.ndarray] = []
    collided_parts: list[np.ndarray] = []
    recovery_mode_parts: list[np.ndarray] = []
    dataset_weight_parts: list[np.ndarray] = []
    loss_head_parts: list[np.ndarray] = []

    for dataset_path, dataset_weight, dataset_role, remap_macro4, action_format, loss_head_mode in zip(
        dataset_paths,
        dataset_weights,
        dataset_roles,
        dataset_remap_flags,
        dataset_action_formats,
        dataset_loss_heads,
        strict=True,
    ):
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
        actions_parts.append(
            map_actions_to_v11rx(
                np.asarray(data["actions"], dtype=np.int64)[mask],
                action_format=action_format,
                remap_macro4_to_7=remap_macro4,
            )
        )
        episode_start_parts.append(np.asarray(data["episode_starts"], dtype=np.float32)[mask])
        no_progress_parts.append(no_progress_raw[mask])
        stuck_time_parts.append(stuck_time_raw[mask])
        collided_parts.append(collided_raw[mask])
        recovery_mode_parts.append(recovery_mode_raw[mask])
        dataset_weight_parts.append(np.full(int(mask.sum()), float(dataset_weight), dtype=np.float32))
        head_mask = parse_loss_head_mode(loss_head_mode)
        loss_head_parts.append(np.repeat(head_mask[None, :], int(mask.sum()), axis=0))

    obs_np = np.concatenate(obs_parts, axis=0)
    actions_np = np.concatenate(actions_parts, axis=0)
    episode_starts_np = np.concatenate(episode_start_parts, axis=0)
    no_progress_np = np.concatenate(no_progress_parts, axis=0)
    stuck_time_np = np.concatenate(stuck_time_parts, axis=0)
    collided_np = np.concatenate(collided_parts, axis=0)
    recovery_mode_np = np.concatenate(recovery_mode_parts, axis=0)
    dataset_weight_np = np.concatenate(dataset_weight_parts, axis=0)
    loss_head_mask_np = np.concatenate(loss_head_parts, axis=0)

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
        loss_head_mask_np = loss_head_mask_np[:limit]

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
    loss_head_mask = torch.as_tensor(loss_head_mask_np, dtype=torch.float32, device=device)

    env = __import__("stable_baselines3.common.vec_env", fromlist=["SubprocVecEnv"]).SubprocVecEnv(
        [make_env(12_345_000, rank, STAGE.state_mode) for rank in range(1)],
        start_method="fork",
    )
    model = make_model(STAGE, env, 12_345_000, args.device)
    matched = warm_start_policy(model, args.resume_model, args.device)
    print(f"warm_start matched_policy_tensors={matched}", flush=True)
    trainable_param_mode, macro_slice, trainable_params = configure_trainable_params(model, args.trainable_param_mode)
    print(
        f"trainable_param_mode={trainable_param_mode} trainable_params={trainable_params} macro_slice={macro_slice}",
        flush=True,
    )

    trainable_tensors = [param for param in model.policy.parameters() if param.requires_grad]
    if not trainable_tensors:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.Adam(trainable_tensors, lr=float(args.learning_rate))
    batch_states = make_zero_states(model, 1, device)
    total_samples = int(obs.shape[0])
    for epoch in range(int(args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        dist, _ = model.policy.get_distribution(obs, batch_states.pi, episode_starts)
        per_head_log_prob: list[torch.Tensor] = []
        per_head_entropy: list[torch.Tensor] = []
        for head_index, categorical in enumerate(dist.distribution):
            logits = categorical.logits
            per_head_log_prob.append(-F.cross_entropy(logits, actions[:, head_index], reduction="none"))
            per_head_entropy.append(categorical.entropy())
        log_prob_heads = torch.stack(per_head_log_prob, dim=1)
        entropy_heads = torch.stack(per_head_entropy, dim=1)
        weighted_head_mask = loss_head_mask * sample_weights[:, None]
        normalizer = weighted_head_mask.sum().clamp_min(1e-6)
        loss = -(log_prob_heads * weighted_head_mask).sum() / normalizer - 1e-3 * (
            entropy_heads * weighted_head_mask
        ).sum() / normalizer
        loss.backward()
        apply_gradient_mask_for_mode(model, trainable_param_mode, macro_slice)
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        head_usage = loss_head_mask.mean(dim=0).detach().cpu().tolist()
        print(
            "epoch={epoch} samples={samples} loss={loss:.6f} avg_weight={avg_weight:.4f} recovery_frac={recovery_frac:.4f} stuck_frac={stuck_frac:.4f} head_usage={head_usage}".format(
                epoch=epoch + 1,
                samples=total_samples,
                loss=float(loss.detach().cpu()),
                avg_weight=float(sample_weights.mean().detach().cpu()),
                recovery_frac=float(torch.as_tensor(recovery_active, dtype=torch.float32).mean().cpu()),
                stuck_frac=float(torch.as_tensor(stuck_active, dtype=torch.float32).mean().cpu()),
                head_usage="[{:.3f},{:.3f},{:.3f}]".format(*head_usage),
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
