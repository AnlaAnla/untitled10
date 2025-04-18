import torch
import torch.onnx
from torchvision import models
import onnxruntime as rt
import numpy as np


# 假设你已经加载了PyTorch模型


# 直接放要转换的模型
def model2onnx(model, output_path):
    # 设置模型为评估模式
    model.eval()

    # 定义输入示例
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 导出ONNX模型
    # output_path = "model_features.onnx"
    input_names = ["input"]
    output_names = ["output"]

    # 导出 model.features 部分
    torch.onnx.export(model, dummy_input, output_path, input_names=input_names, output_names=output_names)


def onnx_run(onnx_file_path):
    # 加载ONNX模型
    onnx_session = rt.InferenceSession(onnx_file_path)

    # 准备输入数据
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # 运行ONNX模型
    onnx_outputs = onnx_session.run(None, {"input": input_data})

    # 获取输出
    output = onnx_outputs[0]
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # resnet50
    # model = models.resnet50(pretrained=False)
    #
    # num_in_feature = model.fc.in_features
    # model.fc = torch.nn.Linear(num_in_feature, 17355)
    # model.load_state_dict(torch.load(r"C:\Code\ML\Model\resent_out17355_AllCard08.pth", map_location=torch.device('cpu')))
    #
    # features = list(model.children())[:-1]  # 去掉全连接层和池化层, 池化层操作在numpy处理 [:-1]为去掉全连接,-2为去掉全连接和池化层
    # model = torch.nn.Sequential(*features)
    #
    # model2onnx(model, r"C:\Code\ML\Model\resent_out17355_AllCard08.onnx")

    # resnest50
    # torch.hub.list('zhanghang1989/ResNeSt')
    # model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=False)
    # num_features = model.fc.in_features
    # model.fc = torch.nn.Linear(num_features, 175, bias=True)
    # model.load_state_dict(torch.load(r"D:\Code\ML\Model\Card_cls2\resnest50_series01.pth"))

    # 直接加载模型
    torch.hub.list('zhanghang1989/ResNeSt')
    model = torch.load(r"D:\Code\ML\Model\Card_cls\resnest50_AllCard08.pth", map_location=device)
    features = list(model.children())[:-1]  # 去掉全连接层和池化层, 池化层操作在numpy处理 [:-1]为去掉全连接,-2为去掉全连接和池化层
    model = torch.nn.Sequential(*features)

    model2onnx(model, r"D:\Code\ML\Model\onnx\resnest50_AllCard08.onnx")
    print('end')
