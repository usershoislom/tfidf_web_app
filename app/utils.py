import math
import re
import pandas as pd


def tokenize(text):
    return re.findall(r"\b[а-яА-ЯёЁa-zA-Z]+\b", text.lower())


def compute_tf(words):
    total_words = len(words)
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] /= total_words
    return tf


def compute_idf(docs):
    N = len(docs)
    df = {}

    for doc in docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1

    idf = {}
    for word, count in df.items():
        idf[word] = math.log10((N + 1) / (count + 1)) + 1

    return idf


def calculate_tfidf(target_text: str, corpus_texts: list[str]):
    all_docs = [tokenize(text) for text in [target_text] + corpus_texts]

    tf = compute_tf(all_docs[0])
    idf = compute_idf(all_docs)
    data = []
    for word in tf:
        if word in idf:
            data.append(
                {
                    "word": word,
                    "tf": round(tf[word], 5),
                    "idf": round(idf[word], 5),
                    "tfidf": round(tf[word] * idf[word], 5),
                }
            )

    df = pd.DataFrame(data)
    return df.sort_values(by="idf", ascending=False).head(50)
