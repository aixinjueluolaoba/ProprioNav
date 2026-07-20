import onnx
from pathlib import Path

def main():
    current_dir = Path(__file__).parent
    onnx_path = str(current_dir / "policy.onnx")
    output_path = str(current_dir / "policy_cleaned.onnx")
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 找到所有的 Constant 节点
    constant_nodes = [node for node in graph.node if node.op_type == "Constant"]
    
    print(f"Found {len(constant_nodes)} Constant nodes to convert.")
    
    for node in constant_nodes:
        out_name = node.output[0]
        tensor_value = None
        for attr in node.attribute:
            if attr.name == "value":
                tensor_value = attr.t
                break
        
        if tensor_value is not None:
            # 命名为对应输出的名称
            tensor_value.name = out_name
            # 追加到初始化器中
            graph.initializer.append(tensor_value)
            # 从计算图节点中移除此 Constant 节点
            graph.node.remove(node)
            print(f"Converted constant '{out_name}' to initializer.")
            
    # 保存优化后的模型
    onnx.save(model, output_path)
    print(f"Saved optimized ONNX to {output_path}")

if __name__ == "__main__":
    main()
