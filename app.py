import streamlit as st
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer



st.set_page_config(page_title="Product Review Analyzer", page_icon="🛍️")
st.title(" Product Review Sentiment Analyzer")
st.write("Analyzing **real product reviews** using Machine Learning!")

with st.spinner("Loading model..."):
    model, vectorizer = load_model()

st.success(f"✅ Model loaded Successfully!")





st.subheader(" Dataset Overview")
fig2, ax2 = plt.subplots()
colors = ['#2ecc71', '#e74c3c']
ax2.bar(['Positive Reviews', 'Negative Reviews'], [1000, 1000], color=colors)
ax2.set_title('Dataset Distribution')
ax2.set_ylabel('Number of Reviews')
ax2.set_xlabel('Sentiment')
for i, v in enumerate([1000, 1000]):
    ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
st.pyplot(fig2)



if 'history' not in st.session_state:
    st.session_state.history = []


st.subheader(" Try It Yourself")
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
        else:
            st.error(f" Negative Review! (Confidence: {confidence:.1f}%)")
        st.session_state.history.append({
            "Review": user_input[:50] + "...",
            "Result": " Positive" if prediction == 1 else " Negative",
            "Confidence": f"{confidence:.1f}%"
        })

st.subheader(" Review History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
else:
    st.info("No reviews analyzed yet. Try analyzing a review above!")

st.markdown("---")
st.markdown("###  Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.success("✅ *'Amazing product, great quality and fast shipping!'*")
with col2:
    st.error("❌ *'Terrible quality, broke after two days of use.'*")

st.caption("Built with Python, NLTK, Scikit-learn & Streamlit | Product Reviews Dataset")