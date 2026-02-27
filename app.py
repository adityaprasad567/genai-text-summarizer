import streamlit as st
import nltk
import heapq
import re
import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTextArea textarea {
        background-color: #1E1E1E;
        color: white;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Text Summarizer")
st.caption("Lightweight Extractive Summarization using NLTK")

def summarize_text(text, num_sentences=3):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    word_frequencies = {}

    for word in words:
        if word.isalnum() and word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if len(word_frequencies) == 0:
        return "Not enough meaningful content to summarize."

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    sentence_scores = {}

    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return ' '.join(summary_sentences)

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Enter your text here:", height=300)

with col2:
    st.markdown("### Settings")
    num_sentences = st.slider("Summary Length (sentences)", 1, 5, 3)
    word_count = len(text_input.split())
    st.metric("Word Count", word_count)

if st.button("Generate Summary"):
    if text_input.strip() != "":
        summary = summarize_text(text_input, num_sentences)

        st.markdown("## 📄 Generated Summary")
        st.success(summary)

        st.download_button(
            label="⬇ Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please enter text to summarize.")