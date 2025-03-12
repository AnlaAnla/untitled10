import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_empty_settings')


if __name__ == '__main__':

    model = SentenceTransformer(r"D:\Code\ML\Model\huggingface\all-MiniLM-L6-v2_fine_tag7")

    output = model.encode(["111"], normalize_embeddings=True)

    print(output)
    