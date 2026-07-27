# ProprioNav 范例：极简 FastAPI 导航服务端

把 `pipeline_out/libncnn_rust.so` 的高层 `nav_init/nav_step/nav_free` 封装成 HTTP 服务。调用者每步只传**当前坐标、目标坐标、朝向**，库内部维护 LSTM、10 维观测、碰撞推断、hybrid 恢复与跳跃探测，输出 `direction`、`speed`、`jump`。

## 依赖

```bash
pip install fastapi uvicorn
```

## 启动

```bash
cd /path/to/ProprioNav
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000/docs` 可看交互式 API 文档。

## API

| 方法 | 路径 | 输入 | 输出 |
|---|---|---|---|
| POST | `/nav/init` | — | `{session_id}` |
| POST | `/nav/step` | `{session_id, pos_x, pos_y, target_x, target_y, heading}` | `{direction, speed, jump}` |
| DELETE | `/nav/{session_id}` | — | `{freed}` |

导航是有状态的，因此按 `session_id` 隔离会话。碰撞不再由调用方传入：SO 根据上一帧指令与本帧实际坐标位移自动推断。

## curl 调用示例

```bash
# 1) 初始化会话
SID=$(curl -s -XPOST localhost:8000/nav/init |
  python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "session=$SID"

# 2) 单步：当前位置 (500, −300)，目标 (−200, 800)，朝向 2.1 rad
curl -s -XPOST localhost:8000/nav/step \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"pos_x\":500.0,\"pos_y\":-300.0,\"target_x\":-200.0,\"target_y\":800.0,\"heading\":2.1}"
# -> {"direction":...,"speed":100.0,"jump":0}

# 3) 结束会话
curl -s -XDELETE localhost:8000/nav/$SID
```

## Python 客户端片段（完整导航循环）

```python
import math, requests

BASE = "http://localhost:8000"
target = (-200.0, 800.0)
pos = [500.0, -300.0]
heading = 2.1
sid = requests.post(f"{BASE}/nav/init").json()["session_id"]

for step in range(240):
    r = requests.post(f"{BASE}/nav/step", json={
        "session_id": sid, "pos_x": pos[0], "pos_y": pos[1],
        "target_x": target[0], "target_y": target[1], "heading": heading,
    }).json()
    heading, spd, jump = r["direction"], r["speed"], r["jump"]
    # direction 已在 SO 内应用 45°/步的转向限制。
    pos[0] += math.cos(heading) * spd * 0.3
    pos[1] += math.sin(heading) * spd * 0.3
    if jump:
        pass  # 在游戏侧触发一次跳跃

requests.delete(f"{BASE}/nav/{sid}")
```

## 字段说明

- `direction`：本步移动方向（弧度，`[-π, π]`），SO 内已应用最大 45° 转向。
- `speed`：建议速度（50 或 100 单位/步，`dt=0.3`）。
- `jump`：是否执行跳跃（0/1）。

LSTM、碰撞推断、hybrid 脱困和跳跃试探/冷却全部由 SO 内部维护。
