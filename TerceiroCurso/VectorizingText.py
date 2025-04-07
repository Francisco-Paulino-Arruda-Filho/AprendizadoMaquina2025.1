from sklearn.feature_extraction.text import TfidfVectorizer

Tfidf_vec = TfidfVectorizer()
text_tfidf = Tfidf_vec.fit_transform(documents)
print(text_tfidf.shape)