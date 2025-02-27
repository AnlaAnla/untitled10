from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
sentences = ["2023-24 Panini Mosaic Mosaic Reactive Blue #289 Tyrese Haliburton City Edition!", "Base Mosaic Reactive Blue"]

embedding1 = model.encode(sentences[0], prompt_name="passage", normalize_embeddings=True)
embedding2 = model.encode(sentences[1], prompt_name="query", normalize_embeddings=True)

print(embedding1.shape)
# (2, 768)

similarity = model.similarity(embedding1, embedding2)
print(similarity)
# tensor([[0.9118]])
