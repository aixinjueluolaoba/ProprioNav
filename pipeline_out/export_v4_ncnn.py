"""Export the V4 unknown-heading policy for PNNX/NCNN deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from run_pipeline import OBS_DIM, RecurrentActorCritic  # noqa: E402


class RecurrentInferenceV4(nn.Module):
    """Single-step LSTM inference; jump remains in the SO probe controller."""

    def __init__(self, model: RecurrentActorCritic):
        super().__init__()
        self.W_ih_t = nn.Parameter(model.lstm.weight_ih_l0.detach().clone().t())
        self.W_hh_t = nn.Parameter(model.lstm.weight_hh_l0.detach().clone().t())
        self.b_ih = nn.Parameter(model.lstm.bias_ih_l0.detach().clone())
        self.b_hh = nn.Parameter(model.lstm.bias_hh_l0.detach().clone())
        self.actor_fc = model.actor_fc
        self.steer_head = model.steer_head
        self.speed_head = model.speed_head
        self.macro_head = model.macro_head

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
        outputs = [self.steer_head(features), self.speed_head(features), h_next, c_next]
        if self.macro_head is not None:
            outputs.append(self.macro_head(features))
        return tuple(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(BASE / "pipeline_out" / "policy_weights_v4_unknown_heading.pth"),
    )
    parser.add_argument(
        "--output",
        default=str(BASE / "pipeline_out" / "policy_v4.pt"),
    )
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    hidden_dim = state["lstm.weight_hh_l0"].shape[1]
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    actor_width = state["actor_fc.0.weight"].shape[0]
    if input_dim != OBS_DIM:
        raise ValueError(f"V4 exporter expects state_dim={OBS_DIM}, got {input_dim}")
    if hidden_dim != 96:
        raise ValueError(f"V4 exporter expects hidden_dim=96, got {hidden_dim}")

    macro = "macro_head.weight" in state
    model = RecurrentActorCritic(
        state_dim=OBS_DIM,
        hidden_dim=hidden_dim,
        actor_width=actor_width,
        macro=macro,
    )
    model.load_state_dict(state)
    model.eval()
    inference = RecurrentInferenceV4(model).eval()

    x = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    h = torch.zeros(1, hidden_dim, dtype=torch.float32)
    c = torch.zeros(1, hidden_dim, dtype=torch.float32)
    traced = torch.jit.trace(inference, (x, h, c))
    output = Path(args.output)
    traced.save(str(output))
    print(f"{output.resolve()} macro={macro}")


if __name__ == "__main__":
    main()
