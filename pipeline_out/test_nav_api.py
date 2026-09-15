"""高层导航 ABI 冒烟测试（当前迷宫/通用策略 policy_mixed, hidden 192）。

校验 `nav_init` / `nav_configure` / `nav_step` / `nav_step_feedback` / `nav_free`：

- 每步返回 0，`turn_delta` ∈ [-π/4, π/4]，`speed` ∈ [0, 100]，`jump` ∈ {0, 1}；
- `abs_angle`（库内部维护的绝对摇杆角）首步精确等于「当前点→目标」的方位角，
  其后每步等于「上一步 abs_angle + 本步 turn_delta」（归一化后）；
- `nav_step_feedback` 在 `macro_out = NULL` 时同样可用（权威碰撞 + 库内置脱困）。

用法：
    python pipeline_out/test_nav_api.py

先把库和模型准备好（二者都可由本仓库生成）：
    RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build --manifest-path ncnn_rust/Cargo.toml --release
    cp ncnn_rust/target/release/libncnn_rust.so pipeline_out/libncnn_rust.so
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

SO_CANDIDATES = [
    BASE / "libncnn_rust.so",
    ROOT / "ncnn_rust/target/release/libncnn_rust.so",
]
PARAM = BASE / "policy_mixed.ncnn.param"
BIN = BASE / "policy_mixed.ncnn.bin"

# 与 policy_mixed 训练环境一致的尺度 (迷宫: 世界 512, 年龄 800/1200, 速度 30)。
WORLD_SIZE = 512.0
AGE_MAX_MS = 800.0
STALE_MS = 1200.0
MAX_SPEED = 30.0

DT_MS = 300.0
STEPS = 24
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
INT_PTR = ctypes.POINTER(ctypes.c_int)


def load_library() -> ctypes.CDLL:
    for path in SO_CANDIDATES:
        if path.exists():
            return ctypes.CDLL(str(path))
    raise RuntimeError(
        "找不到 libncnn_rust.so；先执行：\n"
        '  RUSTFLAGS="-C linker=/usr/bin/gcc" cargo build '
        "--manifest-path ncnn_rust/Cargo.toml --release\n"
        "  cp ncnn_rust/target/release/libncnn_rust.so pipeline_out/libncnn_rust.so"
    )


def declare(lib: ctypes.CDLL) -> None:
    lib.nav_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.nav_init.restype = ctypes.c_void_p
    lib.nav_configure.argtypes = [
        ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.nav_configure.restype = ctypes.c_int
    lib.nav_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        FLOAT_PTR, FLOAT_PTR, INT_PTR, FLOAT_PTR,
    ]
    lib.nav_step.restype = ctypes.c_int
    lib.nav_step_feedback.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int,
        FLOAT_PTR, FLOAT_PTR, INT_PTR, INT_PTR, FLOAT_PTR,
    ]
    lib.nav_step_feedback.restype = ctypes.c_int
    lib.nav_free.argtypes = [ctypes.c_void_p]
    lib.nav_free.restype = None


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def run(lib: ctypes.CDLL, feedback: bool) -> float:
    label = "nav_step_feedback(NULL macro)" if feedback else "nav_step"
    handle = lib.nav_init(str(PARAM).encode(), str(BIN).encode())
    if not handle:
        raise RuntimeError("nav_init failed")
    try:
        status = lib.nav_configure(handle, WORLD_SIZE, AGE_MAX_MS, STALE_MS, MAX_SPEED)
        if status != 0:
            raise RuntimeError(f"nav_configure failed: {status}")

        pos = [-100.0, -50.0]
        target = [100.0, 100.0]
        seed = math.atan2(target[1] - pos[1], target[0] - pos[0])
        prev_abs = None
        max_turn = 0.0

        for step in range(STEPS):
            turn = ctypes.c_float()
            speed = ctypes.c_float()
            jump = ctypes.c_int()
            macro = ctypes.c_int()
            abs_angle = ctypes.c_float()
            if feedback:
                # macro_out = NULL：权威碰撞 + 库内置脱困，通用模型也能走。
                result = lib.nav_step_feedback(
                    handle, pos[0], pos[1], target[0], target[1], DT_MS, 0,
                    ctypes.byref(turn), ctypes.byref(speed), ctypes.byref(jump),
                    None, ctypes.byref(abs_angle),
                )
            else:
                result = lib.nav_step(
                    handle, pos[0], pos[1], target[0], target[1], DT_MS,
                    ctypes.byref(turn), ctypes.byref(speed), ctypes.byref(jump),
                    ctypes.byref(abs_angle),
                )
            if result != 0:
                raise AssertionError(f"{label}: step {step} 返回 {result}")
            if not -math.pi / 4.0 - 1e-5 <= turn.value <= math.pi / 4.0 + 1e-5:
                raise AssertionError(f"{label}: turn_delta 越界 {turn.value}")
            if not 0.0 <= speed.value <= 100.0:
                raise AssertionError(f"{label}: speed 越界 {speed.value}")
            if jump.value not in (0, 1):
                raise AssertionError(f"{label}: jump 非法 {jump.value}")
            if not -math.pi - 1e-4 <= abs_angle.value <= math.pi + 1e-4:
                raise AssertionError(f"{label}: abs_angle 越界 {abs_angle.value}")

            if step == 0:
                if abs(normalize_angle(abs_angle.value - seed)) >= 1e-3:
                    raise AssertionError(
                        f"{label}: 首步 abs_angle {abs_angle.value} != 目标方位角 {seed}"
                    )
            else:
                expected = normalize_angle(prev_abs + turn.value)
                if abs(normalize_angle(abs_angle.value - expected)) >= 1e-4:
                    raise AssertionError(
                        f"{label}: step {step} abs_angle {abs_angle.value} != 上一步+turn {expected}"
                    )
            prev_abs = abs_angle.value
            max_turn = max(max_turn, abs(turn.value))
            pos[0] += math.cos(abs_angle.value) * 5.0
            pos[1] += math.sin(abs_angle.value) * 5.0

        print(
            f"  {label:28s} OK  {STEPS} 步  seed={math.degrees(seed):+7.2f}deg  "
            f"末态 abs={math.degrees(prev_abs):+8.2f}deg  max|turn|={math.degrees(max_turn):.2f}deg"
        )
        return seed
    finally:
        lib.nav_free(handle)


def main() -> None:
    if not PARAM.exists() or not BIN.exists():
        raise RuntimeError(f"缺模型：{PARAM} / {BIN}")

    lib = load_library()
    declare(lib)
    print(f"lib   : {next(p for p in SO_CANDIDATES if p.exists())}")
    print(f"model : {PARAM.name} (hidden 自动识别)")
    run(lib, feedback=False)
    run(lib, feedback=True)
    print("test_nav_api: PASS")


if __name__ == "__main__":
    main()
