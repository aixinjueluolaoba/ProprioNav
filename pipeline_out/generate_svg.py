import re

def main():
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1250 820" width="1250" height="820">
  <defs>
    <style>
      .title { font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; font-weight: 800; font-size: 16px; fill: #1E293B; text-anchor: middle; }
      .section-title { font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; font-weight: 700; font-size: 14px; fill: #0F172A; text-anchor: middle; }
      .box-title { font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; font-weight: 700; font-size: 12px; fill: #1E293B; text-anchor: middle; }
      .box-text { font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }
      .code-text { font-family: 'Consolas', 'Fira Code', monospace; font-size: 11px; fill: #0F172A; }
      
      .box-blue { fill: #EFF6FF; stroke: #3B82F6; stroke-width: 1.5; rx: 8px; }
      .box-green { fill: #F0FDF4; stroke: #22C55E; stroke-width: 1.5; rx: 8px; }
      .box-yellow { fill: #FEFCE8; stroke: #EAB308; stroke-width: 1.5; rx: 8px; }
      .box-rose { fill: #FFF1F2; stroke: #F43F5E; stroke-width: 1.5; rx: 8px; }
      .box-slate { fill: #F8FAFC; stroke: #64748B; stroke-width: 1.5; rx: 8px; }
      .box-gray { fill: #F1F5F9; stroke: #CBD5E1; stroke-width: 1; rx: 6px; }
      
      .boundary-ffi { fill: none; stroke: #D97706; stroke-width: 2; stroke-dasharray: 6,4; rx: 12px; }
      .boundary-py { fill: none; stroke: #0284C7; stroke-width: 2; stroke-dasharray: 8,5; rx: 16px; }
      
      .arrow { fill: none; stroke: #475569; stroke-width: 1.5; marker-end: url(#arrowhead); }
      .arrow-dashed { fill: none; stroke: #94A3B8; stroke-width: 1.2; stroke-dasharray: 4,3; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 8 3.5, 0 7" fill="#475569" />
    </marker>
  </defs>

  <!-- Background Canvas -->
  <rect width="100%" height="100%" fill="#FFFFFF" />

  <!-- 1. Column 1: 物理输入与原始信号 (x: 20 -> 220) -->
  <rect x="20" y="80" width="200" height="710" fill="none" stroke="#E2E8F0" stroke-width="1.5" rx="8" />
  <text x="120" y="110" class="section-title">物理输入与原始信号</text>
  
  <g transform="translate(30, 130)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">字符位置 (pos)</text>
    <text x="90" y="42" class="box-text">坐标向量 (x, y) 像素</text>
  </g>
  
  <g transform="translate(30, 200)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">目标位置 (target)</text>
    <text x="90" y="42" class="box-text">坐标向量 (tx, ty) 像素</text>
  </g>
  
  <g transform="translate(30, 270)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">朝向角度 (heading)</text>
    <text x="90" y="42" class="box-text">罗盘指针弧度 θ</text>
  </g>
  
  <g transform="translate(30, 340)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">坐标年龄 (coord_age)</text>
    <text x="90" y="42" class="box-text">延迟时长 (模拟真实延迟)</text>
  </g>

  <g transform="translate(30, 410)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">卡住时间 (stuck_time)</text>
    <text x="90" y="42" class="box-text">无距离进展的累计时长</text>
  </g>

  <g transform="translate(30, 480)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">碰撞记录 (collision)</text>
    <text x="90" y="42" class="box-text">距离上一次撞墙的累计时长</text>
  </g>

  <g transform="translate(30, 550)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">脱困模板状态 (mode)</text>
    <text x="90" y="42" class="box-text">当前活跃的 macro 模板 ID</text>
  </g>

  <g transform="translate(30, 620)">
    <rect width="180" height="55" class="box-blue" />
    <text x="90" y="24" class="box-title">上一步动作 (last_act)</text>
    <text x="90" y="42" class="box-text">包含 [角差, 速度, 宏ID]</text>
  </g>

  <g transform="translate(30, 690)">
    <rect width="180" height="85" class="box-slate" />
    <text x="90" y="20" class="box-title">物理环境层</text>
    <text x="90" y="38" class="box-text">2D Obstacle Env</text>
    <text x="90" y="54" class="box-text">含有 concave 内凹陷障碍</text>
    <text x="90" y="70" class="box-text">目标距离判定限 &lt;= 8px</text>
  </g>


  <!-- 2. Column 2: 特征预处理与几何映射 (x: 250 -> 550) -->
  <rect x="250" y="80" width="300" height="710" fill="none" stroke="#E2E8F0" stroke-width="1.5" rx="8" />
  <text x="400" y="110" class="section-title">特征预处理与几何映射</text>

  <!-- 几何转换块 -->
  <g transform="translate(265, 130)">
    <rect width="270" height="110" class="box-green" />
    <text x="135" y="20" class="box-title">1. 极坐标几何计算</text>
    <text x="25" y="45" class="code-text">dx = tx - x,  dy = ty - y</text>
    <text x="25" y="65" class="code-text">distance = sqrt(dx² + dy²)</text>
    <text x="25" y="85" class="code-text">target_angle = atan2(dy, dx)</text>
    <text x="25" y="100" class="code-text">angle_err = wrap(target_angle - θ)</text>
  </g>

  <!-- 12D 特征构成列表 -->
  <g transform="translate(265, 260)">
    <rect width="270" height="515" class="box-green" />
    <text x="135" y="22" class="box-title">2. 策略输入向量构造 x [1, 12]</text>
    
    <!-- 12 维的具体子项 -->
    <g transform="translate(15, 38)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[0]  dx_norm = clamp(dx / 2400)</text>
    </g>
    <g transform="translate(15, 73)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[1]  dy_norm = clamp(dy / 2400)</text>
    </g>
    <g transform="translate(15, 108)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[2]  sin(heading) = sin(θ)</text>
    </g>
    <g transform="translate(15, 143)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[3]  cos(heading) = cos(θ)</text>
    </g>
    <g transform="translate(15, 178)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[4]  sin(angle_error) = sin(err)</text>
    </g>
    <g transform="translate(15, 213)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[5]  cos(angle_error) = cos(err)</text>
    </g>
    <g transform="translate(15, 248)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[6]  coord_age_scaled = clamp(age / 1.5)</text>
    </g>
    <g transform="translate(15, 283)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[7]  no_progress_norm = clamp(stuck / 3.0)</text>
    </g>
    <g transform="translate(15, 318)">
      <rect width="240" height="40" class="box-gray" />
      <text x="10" y="17" class="code-text">[8]  stuck_context = </text>
      <text x="10" y="32" class="code-text">     (stuck &gt;= 1.65 || recent_coll &gt;= 0.82)</text>
    </g>
    <g transform="translate(15, 365)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[9]  dist_norm = clamp(distance / 2400)</text>
    </g>
    <g transform="translate(15, 400)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[10] macro_mode_norm = macro_mode / 7.0</text>
    </g>
    <g transform="translate(15, 435)">
      <rect width="240" height="28" class="box-gray" />
      <text x="10" y="18" class="code-text">[11] dist_pressure = clamp(distance / 60)</text>
    </g>
    
    <text x="135" y="495" class="box-text" style="font-weight: 700; fill: #16A34A;">特征经归一化限制于 [-1.0, 1.0]</text>
  </g>


  <!-- 3. Column 3: NCNN核心推理层 (x: 580 -> 900) -->
  <rect x="580" y="80" width="310" height="710" fill="none" stroke="#E2E8F0" stroke-width="1.5" rx="8" />
  <text x="735" y="110" class="section-title">NCNN核心计算图 (policy.param)</text>

  <!-- LSTM Cell Box -->
  <g transform="translate(595, 130)">
    <rect width="280" height="230" class="box-rose" />
    <text x="140" y="22" class="box-title" style="fill: #BE123C;">1. LSTM 门控与隐状态更新单元</text>
    <text x="20" y="50" class="code-text">输入: x [1, 12], h_in [1, 64], c_in [1, 64]</text>
    <text x="20" y="75" class="code-text">gates = x * W_ih_t + h_in * W_hh_t + bias</text>
    <text x="20" y="95" class="code-text">i, f, g, o = chunk(gates, 4)</text>
    
    <!-- 门运算细节 -->
    <text x="20" y="125" class="code-text">i_gate = sigmoid(i)  |  f_gate = sigmoid(f)</text>
    <text x="20" y="145" class="code-text">g_gate = tanh(g)     |  o_gate = sigmoid(o)</text>
    
    <!-- 隐状态更新 -->
    <rect x="15" y="165" width="250" height="48" fill="#FFF1F2" stroke="#FDA4AF" stroke-width="1" rx="4" />
    <text x="140" y="183" class="code-text" style="font-weight:700; fill:#BE123C; text-anchor:middle;">c_out = f_gate * c_in + i_gate * g_gate</text>
    <text x="140" y="201" class="code-text" style="font-weight:700; fill:#BE123C; text-anchor:middle;">h_out = o_gate * tanh(c_out)</text>
  </g>

  <!-- Shared Actor FC -->
  <g transform="translate(595, 380)">
    <rect width="280" height="75" class="box-rose" />
    <text x="140" y="22" class="box-title" style="fill: #BE123C;">2. 共享特征映射层 (actor_fc)</text>
    <text x="140" y="45" class="box-text">全连接线性降维: 64 dims. ➔ 32 dims.</text>
    <text x="140" y="62" class="code-text" style="text-anchor:middle;">actor_features = Tanh(FC_32(h_out))</text>
  </g>

  <!-- Multi-Task Heads -->
  <g transform="translate(595, 475)">
    <rect width="280" height="300" class="box-rose" />
    <text x="140" y="22" class="box-title" style="fill: #BE123C;">3. 多任务输出分支 (Three Action Heads)</text>
    
    <!-- 舵角头 -->
    <g transform="translate(15, 35)">
      <rect width="250" height="70" fill="#FFF1F2" stroke="#FDA4AF" stroke-width="1" rx="6" />
      <text x="125" y="22" class="box-title">舵角决策头 (steer_head)</text>
      <text x="125" y="42" class="box-text">输入 [32] ➔ 线性输出 logits [5]</text>
      <text x="125" y="58" class="code-text" style="text-anchor:middle;">steer_logits = FC_5(actor_features)</text>
    </g>

    <!-- 速度头 -->
    <g transform="translate(15, 120)">
      <rect width="250" height="70" fill="#FFF1F2" stroke="#FDA4AF" stroke-width="1" rx="6" />
      <text x="125" y="22" class="box-title">速度决策头 (speed_head)</text>
      <text x="125" y="42" class="box-text">输入 [32] ➔ 线性输出 logits [2]</text>
      <text x="125" y="58" class="code-text" style="text-anchor:middle;">speed_logits = FC_2(actor_features)</text>
    </g>

    <!-- 宏动作头 -->
    <g transform="translate(15, 205)">
      <rect width="250" height="80" fill="#FFF1F2" stroke="#FDA4AF" stroke-width="1" rx="6" />
      <text x="125" y="22" class="box-title">脱困宏模板决策头 (macro_head)</text>
      <text x="125" y="42" class="box-text">输入 [32] ➔ 线性输出 logits [8]</text>
      <text x="125" y="58" class="code-text" style="text-anchor:middle;">macro_logits = FC_8(actor_features)</text>
      <text x="125" y="72" class="box-text">对应 0:正常跟踪, 1~7:七类预设脱困微动作</text>
    </g>
  </g>


  <!-- 4. Column 4: 物理决策与执行 (x: 930 -> 1230) -->
  <rect x="930" y="80" width="300" height="710" fill="none" stroke="#E2E8F0" stroke-width="1.5" rx="8" />
  <text x="1080" y="110" class="section-title">物理动作决策与摇杆执行</text>

  <!-- 舵角解算 -->
  <g transform="translate(945, 130)">
    <rect width="270" height="135" class="box-yellow" />
    <text x="135" y="22" class="box-title" style="fill: #A16207;">1. 舵角决策与叠加</text>
    <text x="20" y="46" class="code-text">steer_idx = argmax(steer_logits)</text>
    <text x="20" y="66" class="code-text">bias = ANGLE_OFFSETS[steer_idx]</text>
    <text x="20" y="86" class="code-text"># 偏移角: [-45°, -15°, 0°, 15°, 45°]</text>
    <rect x="15" y="98" width="240" height="28" fill="#FEFCE8" stroke="#FEF08A" stroke-width="1" rx="4" />
    <text x="135" y="116" class="code-text" style="font-weight:700; fill:#A16207; text-anchor:middle;">desired_angle = target_angle + bias</text>
  </g>

  <!-- 速度解算 -->
  <g transform="translate(945, 280)">
    <rect width="270" height="95" class="box-yellow" />
    <text x="135" y="22" class="box-title" style="fill: #A16207;">2. 速度映射决策</text>
    <text x="20" y="46" class="code-text">speed_idx = argmax(speed_logits)</text>
    <rect x="15" y="58" width="240" height="28" fill="#FEFCE8" stroke="#FEF08A" stroke-width="1" rx="4" />
    <text x="135" y="76" class="code-text" style="font-weight:700; fill:#A16207; text-anchor:middle;">speed = 100.0 if speed_idx == 1 else 50.0</text>
  </g>

  <!-- 脱困模板状态机 -->
  <g transform="translate(945, 390)">
    <rect width="270" height="200" class="box-yellow" />
    <text x="135" y="22" class="box-title" style="fill: #A16207;">3. 脱困宏模板状态机 (Override FSM)</text>
    <text x="20" y="45" class="code-text">macro_idx = argmax(macro_logits)</text>
    <text x="20" y="65" class="code-text">if macro_idx != 0 and stuck_context:</text>
    <text x="35" y="85" class="code-text">启动特定时长 N 步的逃脱模板:</text>
    
    <!-- 模板细节 -->
    <text x="35" y="110" class="box-text" style="text-anchor:start; font-weight:700;">• 模式1：直接朝反方向倒退逃逸</text>
    <text x="35" y="128" class="box-text" style="text-anchor:start; font-weight:700;">• 模式2~5：后退 + 锁定左/右平移</text>
    <text x="35" y="146" class="box-text" style="text-anchor:start; font-weight:700;">• 模式6/7：后退 + 沿边缘滑移重捕获</text>
    
    <text x="20" y="185" class="code-text" style="fill: #BE123C;">覆盖正常舵角与速度，强锁高速后退</text>
  </g>

  <!-- 最终控制物理摇杆量 -->
  <g transform="translate(945, 605)">
    <rect width="270" height="120" class="box-rose" style="stroke-width: 2;" />
    <text x="135" y="24" class="box-title" style="fill: #9F1239; font-size:13px;">4. 最终摇杆物理输出控制向量</text>
    <text x="20" y="50" class="code-text">计算沿最终转向角的运动分量:</text>
    <rect x="15" y="65" width="240" height="42" fill="#FFF1F2" stroke="#FDA4AF" stroke-width="1.5" rx="6" />
    <text x="135" y="82" class="code-text" style="font-weight:800; font-size:12px; fill:#9F1239; text-anchor:middle;">vx = cos(desired_angle) * speed</text>
    <text x="135" y="99" class="code-text" style="font-weight:800; font-size:12px; fill:#9F1239; text-anchor:middle;">vy = sin(desired_angle) * speed</text>
  </g>


  <!-- Boundary Dashed Rectangles (FFI, Python) -->
  <!-- Rust FFI Boundary (Columns 3 & Param files) -->
  <rect x="568" y="35" width="335" height="765" class="boundary-ffi" />
  <text x="735" y="55" class="box-title" style="fill: #D97706; font-size:13px;">Rust FFI 零拷贝共享库 (libncnn_rust.so)</text>

  <!-- Python ctypes Boundary (Columns 2, 3, 4) -->
  <rect x="238" y="15" width="1002" height="795" class="boundary-py" />
  <text x="739" y="30" class="box-title" style="fill: #0284C7; font-size:14px;">Python ctypes 客户端调用 (test_inference.py)</text>


  <!-- Connecting Arrows -->
  <!-- Col 1 -> Col 2 -->
  <path d="M 210,235 L 230,235 L 230,185 L 257,185" class="arrow" />
  <path d="M 210,160 L 257,160" class="arrow" />
  <path d="M 210,300 L 230,300 L 230,220 L 257,220" class="arrow" />
  <path d="M 210,440 L 240,440 L 240,545 L 272,545" class="arrow-dashed" />
  <path d="M 210,510 L 242,510 L 242,580 L 272,580" class="arrow-dashed" />
  <path d="M 210,580 L 245,580 L 245,665 L 272,665" class="arrow-dashed" />

  <!-- Col 2 -> Col 3 -->
  <path d="M 535,185 L 565,185 L 565,150 L 587,150" class="arrow" />
  <path d="M 535,510 L 570,510 L 570,170 L 587,170" class="arrow" />
  
  <!-- LSTM Cell -> Actor FC -->
  <path d="M 735,360 L 735,372" class="arrow" />
  <!-- Actor FC -> heads -->
  <path d="M 735,455 L 735,467" class="arrow" />
  
  <!-- Col 3 Output -> Col 4 -->
  <path d="M 875,540 L 910,540 L 910,195 L 937,195" class="arrow" />
  <path d="M 875,625 L 915,625 L 915,325 L 937,325" class="arrow" />
  <path d="M 875,710 L 920,710 L 920,490 L 937,490" class="arrow" />

  <!-- Col 4 FSM / steer / speed -> joystick -->
  <path d="M 1080,265 L 1080,272" class="arrow" />
  <path d="M 1080,375 L 1080,382" class="arrow" />
  <path d="M 1080,590 L 1080,597" class="arrow" />
</svg>"""

    # 保存为 SVG 文件
    output_path = "/home/diana/盲人寻路/pipeline_out/proprio_nav_architecture.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"矢量架构图成功保存至: {output_path}")

if __name__ == "__main__":
    main()
