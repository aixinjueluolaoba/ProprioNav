import ctypes
import numpy as np
import torch
import sys
from pathlib import Path

# 添加导出脚本所在路径，方便导入网络定义
sys.path.append(str(Path(__file__).parent))
from export_onnx import RecurrentActorCritic, RecurrentInference

def main():
    print("================== 🌲 PyTorch VS NCNN 推理一致性验证 🌲 ==================")
    
    # 1. 实例化 PyTorch 模型并运行推理
    weights_path = Path('/home/diana/盲人寻路/pipeline_out/policy_weights.pth')
    if not weights_path.exists():
        print(f"Error: 找不到权重文件 {weights_path}")
        return
        
    model = RecurrentActorCritic(state_dim=12, hidden_dim=64)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    inference_model = RecurrentInference(model)
    inference_model.eval()
    
    # 构造随机测试输入
    np.random.seed(42)
    x_np = np.random.randn(1, 12).astype(np.float32)
    h_np = np.random.randn(1, 64).astype(np.float32)
    c_np = np.random.randn(1, 64).astype(np.float32)
    
    # PyTorch 推理
    with torch.no_grad():
        x_torch = torch.from_numpy(x_np)
        h_torch = torch.from_numpy(h_np)
        c_torch = torch.from_numpy(c_np)
        py_steer, py_speed, py_macro, py_h, py_c = inference_model(x_torch, h_torch, c_torch)
        
    py_steer = py_steer.numpy().flatten()
    py_speed = py_speed.numpy().flatten()
    py_macro = py_macro.numpy().flatten()
    py_h = py_h.numpy().flatten()
    py_c = py_c.numpy().flatten()
    
    # 2. 调用 Rust NCNN SO 推理
    so_path = "/home/diana/盲人寻路/pipeline_out/libncnn_rust.so"
    if not Path(so_path).exists():
        print(f"Error: 找不到动态链接库 {so_path}")
        return
        
    lib = ctypes.CDLL(so_path)
    
    # 声明 FFI 接口类型
    lib.init_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.init_net.restype = ctypes.c_void_p
    
    lib.free_net.argtypes = [ctypes.c_void_p]
    lib.free_net.restype = None
    
    lib.run_inference.argtypes = [
        ctypes.c_void_p,                  # net
        ctypes.POINTER(ctypes.c_float),   # x
        ctypes.POINTER(ctypes.c_float),   # h_in
        ctypes.POINTER(ctypes.c_float),   # c_in
        ctypes.POINTER(ctypes.c_float),   # steer_logits
        ctypes.POINTER(ctypes.c_float),   # speed_logits
        ctypes.POINTER(ctypes.c_float),   # macro_logits
        ctypes.POINTER(ctypes.c_float),   # h_out
        ctypes.POINTER(ctypes.c_float),   # c_out
    ]
    lib.run_inference.restype = ctypes.c_int
    
    # 初始化 NCNN
    param_path = "/home/diana/盲人寻路/pipeline_out/policy.param".encode('utf-8')
    bin_path = "/home/diana/盲人寻路/pipeline_out/policy.bin".encode('utf-8')
    
    net_handle = lib.init_net(param_path, bin_path)
    if not net_handle:
        print("Error: NCNN 载入模型参数与权重失败！")
        return
    
    # 准备 NCNN 的输出缓冲区
    ncnn_steer = np.zeros(5, dtype=np.float32)
    ncnn_speed = np.zeros(2, dtype=np.float32)
    ncnn_macro = np.zeros(8, dtype=np.float32)
    ncnn_h = np.zeros(64, dtype=np.float32)
    ncnn_c = np.zeros(64, dtype=np.float32)
    
    # 获取指针
    x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_ptr = h_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    steer_ptr = ncnn_steer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    speed_ptr = ncnn_speed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    macro_ptr = ncnn_macro.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_out_ptr = ncnn_h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_out_ptr = ncnn_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # 执行 Rust 推理
    status = lib.run_inference(
        net_handle,
        x_ptr, h_ptr, c_ptr,
        steer_ptr, speed_ptr, macro_ptr,
        h_out_ptr, c_out_ptr
    )
    
    if status != 0:
        print(f"Error: Rust NCNN 推理失败，错误码为: {status}")
        lib.free_net(net_handle)
        return
        
    # 3. 输出比对结果
    print("\n--- 1. steer_logits 比对 ---")
    print(f"PyTorch: {py_steer}")
    print(f"NCNN   : {ncnn_steer}")
    print(f"Max Abs Diff: {np.max(np.abs(py_steer - ncnn_steer)):.6f}")
    
    print("\n--- 2. speed_logits 比对 ---")
    print(f"PyTorch: {py_speed}")
    print(f"NCNN   : {ncnn_speed}")
    print(f"Max Abs Diff: {np.max(np.abs(py_speed - ncnn_speed)):.6f}")
    
    print("\n--- 3. macro_logits 比对 ---")
    print(f"PyTorch: {py_macro}")
    print(f"NCNN   : {ncnn_macro}")
    print(f"Max Abs Diff: {np.max(np.abs(py_macro - ncnn_macro)):.6f}")
    
    print("\n--- 4. h_next 比对 (部分前 5 维数据展示) ---")
    print(f"PyTorch: {py_h[:5]}")
    print(f"NCNN   : {ncnn_h[:5]}")
    print(f"Max Abs Diff: {np.max(np.abs(py_h - ncnn_h)):.6f}")
    
    print("\n--- 5. c_next 比对 (部分前 5 维数据展示) ---")
    print(f"PyTorch: {py_c[:5]}")
    print(f"NCNN   : {ncnn_c[:5]}")
    print(f"Max Abs Diff: {np.max(np.abs(py_c - ncnn_c)):.6f}")
    
    # 释放网络内存
    lib.free_net(net_handle)
    
    # 校验最终精度差（因为 PNNX 默认启用了 FP16 浮点优化，所以在 1e-2 以内均属超高精度范围）
    max_diff = max(
        np.max(np.abs(py_steer - ncnn_steer)),
        np.max(np.abs(py_speed - ncnn_speed)),
        np.max(np.abs(py_macro - ncnn_macro)),
        np.max(np.abs(py_h - ncnn_h)),
        np.max(np.abs(py_c - ncnn_c))
    )
    
    print("\n" + "=" * 62)
    if max_diff < 1e-2:
        print(f"🎉 验证通过！PyTorch 与 Rust-NCNN 推理结果高度一致！最大偏差: {max_diff:.6f}")
    else:
        print(f"⚠️ 校验偏差超出阈值，最大偏差: {max_diff:.6f}")
    print("=" * 62 + "\n")

if __name__ == '__main__':
    main()
