import torch
import torch.nn as nn
from pathlib import Path

class RecurrentActorCritic(nn.Module):
    def __init__(self, state_dim=12, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=1)
        
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh()
        )
        self.steer_head = nn.Linear(32, 5)
        self.speed_head = nn.Linear(32, 2)
        self.macro_head = nn.Linear(32, 8)
        
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

class RecurrentInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 在构造时提前进行转置，消除 forward 里的动态 Transpose 节点
        self.W_ih_t = nn.Parameter(model.lstm.weight_ih_l0.clone().t())
        self.W_hh_t = nn.Parameter(model.lstm.weight_hh_l0.clone().t())
        self.b_ih = nn.Parameter(model.lstm.bias_ih_l0.clone())
        self.b_hh = nn.Parameter(model.lstm.bias_hh_l0.clone())
        
        self.actor_fc = model.actor_fc
        self.steer_head = model.steer_head
        self.speed_head = model.speed_head
        self.macro_head = model.macro_head
        
    def forward(self, x, h, c):
        # x: [1, 12]
        # h: [1, 64]
        # c: [1, 64]
        
        # 直接乘以提前转置好的权重矩阵 W_ih_t 和 W_hh_t
        gates = torch.matmul(x, self.W_ih_t) + self.b_ih + torch.matmul(h, self.W_hh_t) + self.b_hh
        
        # 拆分为 input, forget, cell, output 四个门 (PyTorch 默认顺序)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        # 经过 MLP 部分
        actor_features = self.actor_fc(h_next)
        
        steer_logits = self.steer_head(actor_features)
        speed_logits = self.speed_head(actor_features)
        macro_logits = self.macro_head(actor_features)
        
        return steer_logits, speed_logits, macro_logits, h_next, c_next

def main():
    current_dir = Path(__file__).parent
    weights_path = current_dir / 'policy_weights.pth'
    if not weights_path.exists():
        print(f"Error: weights file not found at {weights_path}")
        return
        
    model = RecurrentActorCritic(state_dim=12, hidden_dim=64)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    inference_model = RecurrentInference(model)
    inference_model.eval()
    
    # 构造 dummy inputs
    x = torch.zeros(1, 12, dtype=torch.float32)
    h = torch.zeros(1, 64, dtype=torch.float32)
    c = torch.zeros(1, 64, dtype=torch.float32)
    
    # 导出为 ONNX (指定输入输出名，采用标准 opset 11)
    onnx_path = current_dir / 'policy.onnx'
    torch.onnx.export(
        inference_model,
        (x, h, c),
        str(onnx_path),
        input_names=['x', 'h_in', 'c_in'],
        output_names=['steer_logits', 'speed_logits', 'macro_logits', 'h_out', 'c_out'],
        opset_version=11
    )
    print(f"ONNX model successfully exported to {onnx_path}")
    
    # 导出为 TorchScript (.pt) 以便 pnnx 转换
    pt_path = current_dir / 'policy.pt'
    traced_model = torch.jit.trace(inference_model, (x, h, c))
    traced_model.save(str(pt_path))
    print(f"TorchScript model successfully exported to {pt_path}")

if __name__ == '__main__':
    main()
