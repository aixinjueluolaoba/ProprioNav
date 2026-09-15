import json, pathlib

els = []
seed = 100000
def nid(): 
    global seed; seed += 1; return seed

def text_only(s, x, y, size, color="#334155", w=None, h=None):
    i = f"t{nid()}"
    w = w or max(60, int(len(s) * (size * 0.62)))
    h = h or int(size * 1.4)
    els.append({"id": i, "type": "text", "x": x, "y": y, "width": w, "height": h,
                "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
                "fillStyle": "solid", "strokeWidth": 1, "roughness": 0, "opacity": 100,
                "seed": nid(), "text": s, "fontSize": size, "fontFamily": 2,
                "textAlign": "left", "verticalAlign": "top", "boundElements": None,
                "updated": 1})
    return i

def box(bid, x, y, w, h, label, fill, stroke, size=16, tcolor="#1e293b"):
    els.append({"id": bid, "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
                "angle": 0, "strokeColor": stroke, "backgroundColor": fill,
                "fillStyle": "solid", "strokeWidth": 2, "roughness": 0, "opacity": 100,
                "seed": nid(), "roundness": {"type": 3},
                "boundElements": [{"id": f"{bid}_t", "type": "text"}], "updated": 1})
    lines = label.count("\n") + 1
    els.append({"id": f"{bid}_t", "type": "text", "x": x + 8, "y": y + h / 2 - lines * 11,
                "width": w - 16, "height": lines * 22, "angle": 0,
                "strokeColor": tcolor, "backgroundColor": "transparent", "fillStyle": "solid",
                "strokeWidth": 1, "roughness": 0, "opacity": 100, "seed": nid(),
                "text": label, "fontSize": size, "fontFamily": 2, "textAlign": "center",
                "verticalAlign": "middle", "containerId": bid, "originalContainerId": bid,
                "boundElements": None, "lineHeight": 1.25, "updated": 1})

def arrow(aid, x1, y1, x2, y2, label=None, dashed=False, stroke="#475569"):
    el = {"id": aid, "type": "arrow", "x": x1, "y": y1, "width": abs(x2 - x1), "height": abs(y2 - y1),
          "angle": 0, "strokeColor": stroke, "backgroundColor": "transparent", "fillStyle": "solid",
          "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid", "roughness": 0,
          "opacity": 100, "seed": nid(), "points": [[0, 0], [x2 - x1, y2 - y1]],
          "endArrowhead": "arrow", "startArrowhead": None, "updated": 1}
    if dashed:
        el["endArrowhead"] = "arrow"
    els.append(el)

# ---- layout ----
COLW = 250
X = [40, 440, 840, 1240]
H = 70
STEP = 108

text_only("ProprioNav 迷宫 / 通用盲导航 — 训练与部署架构", 40, 16, 28, "#1e293b")

heads = ["① 训练", "② 导出", "③ 推理库 (Rust/NCNN)", "④ 部署"]
for x, htxt in zip(X, heads):
    text_only(htxt, x, 78, 22, "#0f172a")

# Col1 训练
box("b_run", X[0], 130, COLW, H, "run_pipeline.py\nPPO + LSTM(192) 向量化训练", "#dbeafe", "#1e40af")
box("b_env", X[0], 130 + STEP, COLW, H, "map_env\n迷宫 / 混合(迷宫+开放世界)", "#dbeafe", "#1e40af")
box("b_shape", X[0], 130 + 2*STEP, COLW, H, "课程 · 定位延迟 200-800ms\n减速/过冲 · 探索", "#fef9c3", "#854d0e")
box("b_w", X[0], 130 + 3*STEP, COLW, H, "权重 .pth\nmixed / overshoot", "#dcfce7", "#166534")

# Col2 导出
box("b_exp", X[1], 130, COLW, H, "export_mixed_ncnn.py", "#e0f2fe", "#0369a1")
box("b_ts", X[1], 130 + STEP, COLW, H, "TorchScript (.pt)", "#e0f2fe", "#0369a1")
box("b_ncnn", X[1], 130 + 2*STEP, COLW, H, "pnnx → NCNN\npolicy_mixed.param/bin", "#e0f2fe", "#0369a1")

# Col3 库
box("b_init", X[2], 130, COLW, H, "nav_init\nhidden 自动识别 96 / 192", "#f3e8ff", "#6b21a8")
box("b_cfg", X[2], 130 + STEP, COLW, H, "nav_configure\nworld / age / speed", "#f3e8ff", "#6b21a8")
box("b_step", X[2], 130 + 2*STEP, COLW, H, "nav_step / nav_step_feedback\n(C ABI)", "#f3e8ff", "#6b21a8")
box("b_belief", X[2], 130 + 3*STEP, COLW, H, "内部状态\n速度/朝向估计 · freshness 限速\n相对阈值 · 转向冷却(防转圈)", "#fef9c3", "#854d0e")

# Col4 部署
box("b_so", X[3], 130, COLW, H, "Linux .so", "#dcfce7", "#166534")
box("b_aar", X[3], 130 + STEP, COLW, H, "Android AAR\nJNI + Kotlin ProprioNav", "#dcfce7", "#166534")
box("b_game", X[3], 130 + 2*STEP, COLW, H, "真实游戏 (寻路)\npos.txt→inject_test→bridge→摇杆", "#dcfce7", "#166534")
box("b_other", X[3], 130 + 3*STEP, COLW, H, "其它游戏\n同一 pos/target/age ABI", "#dcfce7", "#166534")

# vertical internal arrows (dashed)
def vchain(ids):
    for a, b in zip(ids, ids[1:]):
        arrow(f"a_{a}_{b}", X[0] + COLW/2, 0, 0, 0)  # placeholder replaced below
# build vertical arrows properly
def vlink(a_id, ay, b_id, by, col):
    arrow(f"v_{a_id}_{b_id}", col + COLW/2, ay, col + COLW/2, by, dashed=True, stroke="#94a3b8")

for col, ids, ys in [
    (X[0], ["b_run","b_env","b_shape","b_w"], [130,130+STEP,130+2*STEP,130+3*STEP]),
    (X[1], ["b_exp","b_ts","b_ncnn"], [130,130+STEP,130+2*STEP]),
    (X[2], ["b_init","b_cfg","b_step","b_belief"], [130,130+STEP,130+2*STEP,130+3*STEP]),
    (X[3], ["b_so","b_aar","b_game"], [130,130+STEP,130+2*STEP]),
]:
    for i in range(len(ids)-1):
        vlink(ids[i], ys[i]+H, ids[i+1], ys[i+1], col)

# horizontal cross-column arrows
arrow("h_train_exp", X[0]+COLW, 130+3*STEP+H/2, X[1], 130+H/2, "权重")
arrow("h_ncnn_lib", X[1]+COLW, 130+2*STEP+H/2, X[2], 130+H/2, "NCNN 模型")
arrow("h_lib_deploy", X[2]+COLW, 130+2*STEP+H/2, X[3], 130+H/2, "C ABI")

# bottom I/O legend (free text, no boxes)
text_only("调用方输入", 40, 130+4*STEP+30, 18, "#0f172a")
text_only("自身坐标 (x,y)  ·  目标/waypoint (x,y)  ·  position_age_ms  ·  (collided)", 40, 130+4*STEP+58, 16, "#334155")
text_only("库输出", 40, 130+4*STEP+96, 18, "#0f172a")
text_only("turn_delta (±45°/步)  ·  speed (0-100, 已按新鲜度限速)  ·  jump (0/1)", 40, 130+4*STEP+124, 16, "#334155")

doc = {"type": "excalidraw", "version": 2, "source": "claude-code", "elements": els,
       "appState": {"viewBackgroundColor": "#ffffff"}}
out = pathlib.Path("/home/diana/盲人寻路/docs/architecture_maze_nav.excalidraw")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
print("wrote", out)
