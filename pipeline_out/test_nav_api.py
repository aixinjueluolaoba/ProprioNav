"""高层导航 ABI 冒烟测试（迷宫/通用 policy_mixed + V5 宏模型 policy_v5_macro_fast）。

校验 `nav_init` / `nav_configure` / `nav_step` / `nav_step_feedback` / `nav_free`：

- 每步返回 0，`turn_delta` ∈ [-π/4, π/4]，`speed` ∈ [0, 100]，`jump` ∈ {0, 1}；
- `abs_angle`（库内部维护的绝对摇杆角）首步精确等于「当前点→目标」的方位角，
  其后每步等于「上一步 abs_angle + 本步 turn_delta」（归一化后）；
- `nav_step` 与 `nav_step_feedback` 使用**同一套库内碰撞推断**（后者只是多一个
  `macro` 输出）；无宏头的模型把 `macro` 传 NULL 也能调后者；
- `nav_step_feedback` 传非 NULL `macro` 时（需要 V5 宏模型），宏 id ∈ [0, 7]。

用法：
    python pipeline_out/test_nav_api.py

先把库准备好（模型已在仓库内）：
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
MIXED_PARAM = BASE / "policy_mixed.ncnn.param"
MIXED_BIN = BASE / "policy_mixed.ncnn.bin"
V5_PARAM = BASE / "policy_v5_macro_fast.param"
V5_BIN = BASE / "policy_v5_macro_fast.bin"

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
        FLOAT_PTR, FLOAT_PTR, INT_PTR, INT_PTR, FLOAT_PTR,
    ]
    lib.nav_step_feedback.restype = ctypes.c_int
    lib.nav_free.argtypes = [ctypes.c_void_p]
    lib.nav_free.restype = None


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def run(
    lib: ctypes.CDLL,
    param: Path,
    model_bin: Path,
    api: str,
    macro: bool,
    configure: bool,
) -> None:
    label = api + ("(macro)" if macro else "")
    handle = lib.nav_init(str(param).encode(), str(model_bin).encode())
    if not handle:
        raise RuntimeError(f"nav_init failed for {param.name}")
    try:
        if configure:
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
            macro_id = ctypes.c_int()
            abs_angle = ctypes.c_float()
            if api == "nav_step_feedback":
                result = lib.nav_step_feedback(
                    handle, pos[0], pos[1], target[0], target[1], DT_MS,
                    ctypes.byref(turn), ctypes.byref(speed), ctypes.byref(jump),
                    ctypes.byref(macro_id) if macro else None,
                    ctypes.byref(abs_angle),
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
            if macro and not 0 <= macro_id.value <= 7:
                raise AssertionError(f"{label}: macro id 越界 {macro_id.value}")

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
            # 模拟按库给的绝对角前进，好让位移/碰撞推断有输入。
            pos[0] += math.cos(abs_angle.value) * 5.0
            pos[1] += math.sin(abs_angle.value) * 5.0

        print(
            f"  {label:26s} OK  {STEPS} 步  seed={math.degrees(seed):+7.2f}deg  "
            f"末态 abs={math.degrees(prev_abs):+8.2f}deg  max|turn|={math.degrees(max_turn):.2f}deg"
        )
    finally:
        lib.nav_free(handle)


def main() -> None:
    if not MIXED_PARAM.exists() or not MIXED_BIN.exists():
        raise RuntimeError(f"缺模型：{MIXED_PARAM} / {MIXED_BIN}")

    lib = load_library()
    declare(lib)
    print(f"lib   : {next(p for p in SO_CANDIDATES if p.exists())}")

    print(f"model : {MIXED_PARAM.name} (hidden 自动识别, 无宏头)")
    run(lib, MIXED_PARAM, MIXED_BIN, "nav_step", macro=False, configure=True)
    run(lib, MIXED_PARAM, MIXED_BIN, "nav_step_feedback", macro=False, configure=True)

    if V5_PARAM.exists() and V5_BIN.exists():
        print(f"model : {V5_PARAM.name} (hidden 自动识别, 有宏头, 默认 V5 尺度)")
        run(lib, V5_PARAM, V5_BIN, "nav_step_feedback", macro=True, configure=False)
    else:
        print(f"skip  : 缺 {V5_PARAM.name}，跳过宏输出检查")

    print("test_nav_api: PASS")


if __name__ == "__main__":
    main()
