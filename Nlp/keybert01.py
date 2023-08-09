from keybert import KeyBERT
import jieba

# 中文测试
# doc = ""

with open('11.txt', encoding='utf-8') as f:
    doc = f.read()


doc = " ".join(jieba.cut(doc))
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

print("naive ...")
keywords = kw_model.extract_keywords(doc)
print(keywords)

print("\nkeyphrase_ngram_range ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
print(keywords)

print("\nhighlight ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), highlight=True)
print(keywords)

# 为了使结果多样化，我们将 2 x top_n 与文档最相似的词/短语。
# 然后，我们从 2 x top_n 单词中取出所有 top_n 组合，并通过余弦相似度提取彼此最不相似的组合。
print("\nuse_maxsum ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                              use_maxsum=True, nr_candidates=20, top_n=5)
print(keywords)

# 为了使结果多样化，我们可以使用最大边界相关算法(MMR)
# 来创建同样基于余弦相似度的关键字/关键短语。 具有高度多样性的结果：
print("\nhight diversity ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                        use_mmr=True, diversity=0.7)
print(keywords)

# 低多样性的结果
print("\nlow diversity ...")
keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                        use_mmr=True, diversity=0.2)
print(keywords)
