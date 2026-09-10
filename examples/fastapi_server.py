"""
ProprioNav 极简 FastAPI 导航服务端
================================

封装 pipeline_out/libncnn_rust.so 的高层 V4 nav API
(nav_init/nav_step/nav_free)，通过 HTTP 暴露。调用者每步传「当前坐标、
目标坐标、定位观测年龄」，库内部维护 LSTM、13 维观测、运动方向估计、
碰撞推断、延迟补偿与跳跃探测，输出相对转向 turn_delta、速度 speed 和 jump。

导航是有状态的（每步依赖上一步的 LSTM h/c 与时序累积量），因此按会话隔离：
  1) POST /nav/init    {}                           -> {session_id}
  2) POST /nav/step    {session_id, pos_x, pos_y, target_x, target_y, position_age_ms}
                                                   -> {turn_delta, speed, jump}
  3) DELETE /nav/{session_id}                       -> {freed}

启动：
    pip install fastapi uvicorn
    cd /path/to/ProprioNav
    uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import ctypes
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── 定位 pipeline_out 下的 .so 与模型文件 ──
ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline_out"
SO_PATH = PIPELINE / "libncnn_rust.so"
PARAM_PATH = PIPELINE / "policy.param"
BIN_PATH = PIPELINE / "policy.bin"

if not SO_PATH.exists():
    raise RuntimeError(
        f"找不到 {SO_PATH}，请先编译 ncnn_rust 并将 libncnn_rust.so 拷贝至 pipeline_out/"
    )

try:
    lib = ctypes.CDLL(str(SO_PATH))
except OSError as exc:
    raise RuntimeError(f"加载 {SO_PATH} 失败：{exc}") from exc

# ── FFI 签名（与 pipeline_out/test_nav_api.py 一致）──
lib.nav_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.nav_init.restype = ctypes.c_void_p

lib.nav_step.argtypes = [
    ctypes.c_void_p,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
]
lib.nav_step.restype = ctypes.c_int

lib.nav_free.argtypes = [ctypes.c_void_p]
lib.nav_free.restype = None

app = FastAPI(title="ProprioNav Nav API", version="3.0")

# ── 会话存储：session_id -> nav 句柄（c_void_p）──
_sessions: dict[str, object] = {}


class StepReq(BaseModel):
    session_id: str
    pos_x: float = Field(..., description="当前 x 坐标")
    pos_y: float = Field(..., description="当前 y 坐标")
    target_x: float = Field(..., description="目标 x 坐标")
    target_y: float = Field(..., description="目标 y 坐标")
    position_age_ms: float = Field(..., ge=0.0, description="定位观测年龄（毫秒）")


class StepResp(BaseModel):
    turn_delta: float = Field(..., description="相对上一运动方向的转角（弧度）")
    speed: float = Field(..., ge=0.0, le=100.0, description="定位新鲜度限速后的速度")
    jump: int = Field(..., ge=0, le=1, description="是否跳跃 0/1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/nav/init")
def nav_init() -> dict:
    """初始化导航会话并返回 session_id。"""
    handle = lib.nav_init(
        str(PARAM_PATH).encode("utf-8"),
        str(BIN_PATH).encode("utf-8"),
    )
    if not handle:
        raise HTTPException(status_code=500, detail="nav_init 失败：模型加载错误")
    sid = uuid.uuid4().hex
    _sessions[sid] = handle
    return {"session_id": sid}


@app.post("/nav/step", response_model=StepResp)
def nav_step(req: StepReq) -> StepResp:
    """单步导航：输入坐标与定位年龄，输出相对转向、速度和跳跃。"""
    handle = _sessions.get(req.session_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="无效或已过期的 session_id")
    turn_delta = ctypes.c_float(0.0)
    sp = ctypes.c_float(0.0)
    jump = ctypes.c_int(0)
    ret = lib.nav_step(
        handle,
        ctypes.c_float(req.pos_x), ctypes.c_float(req.pos_y),
        ctypes.c_float(req.target_x), ctypes.c_float(req.target_y),
        ctypes.c_float(req.position_age_ms),
        ctypes.byref(turn_delta), ctypes.byref(sp), ctypes.byref(jump),
    )
    if ret != 0:
        raise HTTPException(status_code=500, detail=f"nav_step 失败：错误码 {ret}")
    return StepResp(turn_delta=turn_delta.value, speed=sp.value, jump=jump.value)


@app.delete("/nav/{session_id}")
def nav_free(session_id: str) -> dict:
    """释放导航会话及其底层 NCNN 网络。"""
    handle = _sessions.pop(session_id, None)
    if handle is None:
        raise HTTPException(status_code=404, detail="session_id 不存在")
    lib.nav_free(handle)
    return {"freed": session_id}
