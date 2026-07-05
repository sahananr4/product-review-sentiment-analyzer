import streamlit as st
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

@st.cache_data
def load_data():
    df = pd.read_csv('Reviews.csv')
    df = df[['Text', 'Score']].dropna()
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    df_pos = df[df['sentiment'] == 1].sample(1000, random_state=42)
    df_neg = df[df['sentiment'] == 0].sample(1000, random_state=42)
    return pd.concat([df_pos, df_neg])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

@st.cache_resource
def train_model():
    df = load_data()
    df['clean'] = df['Text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model, vectorizer, acc, cm

st.set_page_config(page_title="Product Review Analyzer", page_icon="🛍️")
st.title(" Product Review Sentiment Analyzer")
st.write("Analyzing **real product reviews** using Machine Learning!")

with st.spinner("Training model... please wait"):
    model, vectorizer, acc, cm = train_model()

st.success(f"✅ Model trained! Accuracy: {acc * 100:.1f}%")


st.subheader("🎯 Model Accuracy")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Accuracy", value=f"{acc * 100:.1f}%")
with col2:
    st.metric(label="Training Data", value="1,600 reviews")
with col3:
    st.metric(label="Testing Data", value="400 reviews")
st.progress(int(acc * 100))


st.subheader("📊 Dataset Overview")
fig2, ax2 = plt.subplots()
colors = ['#2ecc71', '#e74c3c']
ax2.bar(['Positive Reviews', 'Negative Reviews'], [1000, 1000], color=colors)
ax2.set_title('Dataset Distribution')
ax2.set_ylabel('Number of Reviews')
ax2.set_xlabel('Sentiment')
for i, v in enumerate([1000, 1000]):
    ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
st.pyplot(fig2)


st.subheader("📈 Model Performance")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)


if 'history' not in st.session_state:
    st.session_state.history = []


st.subheader("✍️ Try It Yourself")
user_input = st.text_area("Enter a product review:", height=150,
    placeholder="e.g. The product quality is amazing and delivery was fast!")

if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0][prediction] * 100
        if prediction == 1:
            st.success(f" Positive Review! (Confidence: {confidence:.1f}%)")
            st.balloons()
        else:
            st.error(f" Negative Review! (Confidence: {confidence:.1f}%)")
        st.session_state.history.append({
            "Review": user_input[:50] + "...",
            "Result": " Positive" if prediction == 1 else " Negative",
            "Confidence": f"{confidence:.1f}%"
        })

st.subheader("📋 Review History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
else:
    st.info("No reviews analyzed yet. Try analyzing a review above!")

st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.success("✅ *'Amazing product, great quality and fast shipping!'*")
with col2:
    st.error("❌ *'Terrible quality, broke after two days of use.'*")

st.caption("Built with Python, NLTK, Scikit-learn & Streamlit | Product Reviews Dataset")