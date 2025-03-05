from sentence_transformers import SentenceTransformer
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")

if __name__ == '__main__':
    model_path = r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag5"  # 替换为你的模型路径
    model = SentenceTransformer(model_path)

    print(model)
    print()