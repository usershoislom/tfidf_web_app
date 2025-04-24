from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def calcutate_tfidf(text: str, top_n: int = 50):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    data = [
        {"word": word, "tf": text.lower().split().count(word), "idf": idf}
        for word, idf in zip(feature_names, scores)
    ]

    df = pd.DataFrame(data)
    df = df.sort_values(by="idf", ascending=False).head(top_n)

    return df