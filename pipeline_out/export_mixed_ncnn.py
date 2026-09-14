"""Export the maze / mixed (coordinate-only) policy for PNNX/NCNN deployment.

Same single-step LSTM inference graph as ``export_v4_ncnn.py`` but supports the
larger hidden size (192) and any speed-bin count, so the general maze+open-world
policy can be shipped through the existing ``nav_step`` C ABI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from run_pipeline import OBS_DIM, RecurrentActorCritic  # noqa: E402


class RecurrentInference(nn.Module):
    """Single-step LSTM inference; jump stays in the SO probe controller."""

    def __init__(self, model: RecurrentActorCritic):
        super().__init__()
        self.W_ih_t = nn.Parameter(model.lstm.weight_ih_l0.detach().clone().t())
        self.W_hh_t = nn.Parameter(model.lstm.weight_hh_l0.detach().clone().t())
        self.b_ih = nn.Parameter(model.lstm.bias_ih_l0.detach().clone())
        self.b_hh = nn.Parameter(model.lstm.bias_hh_l0.detach().clone())
        self.actor_fc = model.actor_fc
        self.steer_head = model.steer_head
        self.speed_head = model.speed_head

    def forward(self, x, h, c):
        gates = (
            torch.matmul(x, self.W_ih_t)
            + self.b_ih
            + torch.matmul(h, self.W_hh_t)
            + self.b_hh
        )
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        features = self.actor_fc(h_next)
        return self.steer_head(features), self.speed_head(features), h_next, c_next


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(BASE / "pipeline_out" / "policy_weights_mixed.pth"),
    )
    parser.add_argument(
        "--output",
        default=str(BASE / "pipeline_out" / "policy_mixed.pt"),
    )
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    hidden_dim = state["lstm.weight_hh_l0"].shape[1]
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    actor_width = state["actor_fc.0.weight"].shape[0]
    speed_bins = state["speed_head.weight"].shape[0]
    if input_dim != OBS_DIM:
        raise ValueError(f"exporter expects state_dim={OBS_DIM}, got {input_dim}")

    model = RecurrentActorCritic(
        state_dim=OBS_DIM,
        hidden_dim=hidden_dim,
        actor_width=actor_width,
        macro=False,
        speed_bins=speed_bins,
    )
    model.load_state_dict(state)
    model.eval()
    inference = RecurrentInference(model).eval()

    x = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    h = torch.zeros(1, hidden_dim, dtype=torch.float32)
    c = torch.zeros(1, hidden_dim, dtype=torch.float32)
    traced = torch.jit.trace(inference, (x, h, c))
    output = Path(args.output)
    traced.save(str(output))
    print(
        f"{output.resolve()} hidden={hidden_dim} actor={actor_width} "
        f"speed_bins={speed_bins} inputshape=[1,{OBS_DIM}],[1,{hidden_dim}],[1,{hidden_dim}]"
    )


if __name__ == "__main__":
    main()
