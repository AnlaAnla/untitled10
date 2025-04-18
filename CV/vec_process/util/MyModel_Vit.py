import torch
from transformers import ViTForImageClassification, ViTConfig, ViTModel
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class MyViTModel:
    def __init__(self, model_path: str, use_onnx=False) -> None:

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_onnx = use_onnx

        if not use_onnx:  # 使用PyTorch模型
            self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224')  # 或者你训练时用的其他config
            self.config.num_labels = 17355
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                  config=self.config,
                                                  ignore_mismatched_sizes=True)  # 改为 ViTModel，用于特征提取
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.classifier = torch.nn.Identity()  # 移除分类器，直接提取特征

            self.model.to(self.device)
            self.model.eval()
        else:
            import onnxruntime as rt
            self.model = rt.InferenceSession(model_path)

    def inference_transform(self):
        inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 确保与你的训练尺寸一致
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        return inference_transform

    # 输入图片, 获取图片特征向量
    def run(self, img):
        if type(img) == str:
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            img = img.convert('RGB')
        else:
            # 假设是PIL Image对象
            pass  # 不做转换
        transform = self.inference_transform()

        img_tensor = transform(img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

        if not self.use_onnx:
            with torch.no_grad():
                outputs = self.model(img_tensor)
                output_features = outputs.logits

            return output_features.cpu().numpy()
        else:  # 使用ONNX模型
            img_tensor = img_tensor.detach().cpu().numpy()
            output_features = self.model.run(None, {"pixel_values": img_tensor})[0]
            return output_features  # 假设ONNX模型直接输出特征向量


# 使用例子
# if __name__ == '__main__':
#     # 模型路径，替换成你自己的
#     model_path = r"D:\Code\ML\Project\untitled10\CV\vit\best_model.pth"  # 注意这里应该是pytorch模型pth的路径，而不是onnx
#     image_path = r"D:\Code\ML\Image\_CLASSIFY\card_cls\train_test\train\#1 kyle kuzma\IMG_5742.JPG"
#
#     # 初始化模型
#     my_vit_model = MyViTModel(model_path=model_path, use_onnx=False)
#
#     # 读取图像
#     img = Image.open(image_path).convert('RGB')
#
#     # 获取特征向量
#     features = my_vit_model.run(img)
#
#     print(f"Extracted features shape: {features.shape}")  # 应该是 (1, 768)
#     print(f"Extracted features: {features}")
