"""Compare V4 TorchScript math against the Rust/NCNN low-level FFI."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

from export_v4_ncnn import RecurrentInferenceV4

BASE = Path(__file__).resolve().parent


def ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def main():
    from run_pipeline import OBS_DIM, RecurrentActorCritic

    state = torch.load(BASE / "policy_weights_v4_unknown_heading.pth", map_location="cpu")
    model = RecurrentActorCritic(state_dim=OBS_DIM, hidden_dim=96, actor_width=48, macro=False)
    model.load_state_dict(state)
    reference = RecurrentInferenceV4(model).eval()

    library = ctypes.CDLL(str(BASE / "libncnn_rust.so"))
    library.init_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.init_net.restype = ctypes.c_void_p
    library.free_net.argtypes = [ctypes.c_void_p]
    library.run_inference.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.run_inference.restype = ctypes.c_int

    handle = library.init_net(
        str(BASE / "policy_v4.param").encode(),
        str(BASE / "policy_v4.bin").encode(),
    )
    if not handle:
        raise RuntimeError("init_net failed")

    rng = np.random.default_rng(20260825)
    max_diffs = np.zeros(4, dtype=np.float64)
    for _ in range(32):
        x = rng.normal(size=OBS_DIM).astype(np.float32)
        h = rng.normal(scale=0.2, size=96).astype(np.float32)
        c = rng.normal(scale=0.2, size=96).astype(np.float32)
        with torch.no_grad():
            expected = reference(
                torch.from_numpy(x).view(1, OBS_DIM),
                torch.from_numpy(h).view(1, 96),
                torch.from_numpy(c).view(1, 96),
            )
        expected_np = [value.numpy().reshape(-1) for value in expected]
        actual = [
            np.zeros(7, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(96, dtype=np.float32),
            np.zeros(96, dtype=np.float32),
        ]
        status = library.run_inference(
            handle,
            ptr(x),
            ptr(h),
            ptr(c),
            ptr(actual[0]),
            ptr(actual[1]),
            ptr(actual[2]),
            ptr(actual[3]),
        )
        if status != 0:
            raise RuntimeError(f"run_inference failed: {status}")
        max_diffs = np.maximum(
            max_diffs,
            [float(np.max(np.abs(a - b))) for a, b in zip(actual, expected_np)],
        )

    library.free_net(handle)
    print(
        "max_abs_diff "
        f"steer={max_diffs[0]:.6f} speed={max_diffs[1]:.6f} "
        f"h={max_diffs[2]:.6f} c={max_diffs[3]:.6f}"
    )
    if np.max(max_diffs) >= 0.005:
        raise SystemExit("NCNN deviation exceeds FP32 tolerance")


if __name__ == "__main__":
    main()
