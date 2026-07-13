import nltk
import re
import pandas as pd
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df = pd.read_csv("Reviews.csv")

df = df[['Text','Score']].dropna()

df['sentiment'] = df['Score'].apply(lambda x:1 if x>=4 else 0)

df_pos = df[df['sentiment']==1].sample(1000,random_state=42)
df_neg = df[df['sentiment']==0].sample(1000,random_state=42)

df = pd.concat([df_pos,df_neg])

df['clean'] = df['Text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['clean'])

y = df['sentiment']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model Saved Successfully!")