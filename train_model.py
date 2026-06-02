
import pandas as pd, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("augmented_dataset.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["disease"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=9000)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

pickle.dump(model, open("final_disease_model.pkl","wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl","wb"))

print("Initial model trained successfully")
