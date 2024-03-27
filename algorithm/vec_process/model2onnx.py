import torch
import torch.onnx
import onnxruntime as rt
import numpy as np
#
# # 假设你已经加载了PyTorch模型
# model = ...
#
# # 设置模型为评估模式
# model.eval()
#
# # 定义输入示例
# dummy_input = torch.randn(1, 3, 224, 224)
#
# # 导出ONNX模型
# output_path = "model_features.onnx"
# input_names = ["input"]
# output_names = ["output"]
#
# # 导出 model.features 部分
# torch.onnx.export(model.features, dummy_input, output_path, input_names=input_names, output_names=output_names)


# ==================================


# 加载ONNX模型
onnx_session = rt.InferenceSession(r"C:\Code\ML\Model\onnx\model_features_card03.onnx")

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 运行ONNX模型
onnx_outputs = onnx_session.run(None, {"input": input_data})

# 获取输出
output = onnx_outputs[0]
print(f"Output shape: {output.shape}")