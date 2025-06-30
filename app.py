
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from textstat import flesch_reading_ease
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

nltk.download('punkt')
nltk.download('stopwords')

DANDELION_TOKEN = st.secrets.get("DANDELION_API_KEY", "")

stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")

AMBIGUOUS_PHRASES = {
    "click here": "Use descriptive anchor text like 'Read our SEO guide'",
    "learn more": "Specify what the user will learn, e.g., 'Learn SEO basics'",
    "read more": "Use clear CTA like 'Read the full keyword research case study'",
    "overview": "Use topic-specific phrasing like 'SEO strategy breakdown'",
    "details": "Clarify scope like 'technical audit details'",
    "more info": "Replace with specific queries like 'internal linking examples'",
    "start now": "Add context like 'Start optimizing your on-page SEO now'",
    "this article": "Clarify with topic, e.g., 'This guide on local SEO'",
    "the following": "Specify the subject, e.g., 'The following optimization tips'",
    "some believe": "Avoid weasel phrases — cite actual expert opinion or source",
    "it could be argued": "Make assertions clearer or cite a source"
}

def count_tokens(text):
    return len(tokenizer.encode(text))

def analyze_readability(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0.0

def fetch_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if 'text/html' not in response.headers.get('Content-Type', ''):
            st.warning("Unsupported content type")
            return None
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def extract_chunks(soup):
    for tag in soup(['header', 'footer', 'nav', 'aside']):
        tag.decompose()

    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            hits = [p for p in AMBIGUOUS_PHRASES if p in text.lower()]
            tips = [f"{p} → {AMBIGUOUS_PHRASES[p]}" for p in hits]
            quality = 100
            if tokens > 300: quality -= 15
            if readability < 60: quality -= 15
            if hits: quality -= 10 * len(hits)
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': hits,
                'suggestions': tips,
                'quality_score': max(0, quality)
            })
    return chunks

st.set_page_config(page_title="GenAI Optimization Checker", layout="wide")
st.title("GenAI Optimization Checker")

option = st.radio("Select input method:", ["URL", "HTML file", "Raw HTML", ".txt file"])
html_content = ""

if option == "URL":
    url = st.text_input("Enter URL to analyze:", "https://example.com")
    if st.button("Fetch and Analyze"):
        soup = fetch_url_content(url)
        if soup:
            html_content = soup.prettify()
elif option == "HTML file":
    file = st.file_uploader("Upload HTML file", type="html")
    if file: html_content = file.read().decode("utf-8")
elif option == "Raw HTML":
    html_content = st.text_area("Paste raw HTML:")
elif option == ".txt file":
    txt = st.file_uploader("Upload .txt file", type="txt")
    if txt: html_content = txt.read().decode("utf-8")

if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = extract_chunks(soup)
    if not chunks:
        st.warning("No content chunks found.")
    else:
        df = pd.DataFrame(chunks)
        st.subheader("📄 Content Chunks & Quality Scores")
        for i, row in df.iterrows():
            st.markdown(f"### Chunk {i+1}")
            st.text(row['text'])
            st.markdown(f"**Tokens**: {row['token_count']} | **Readability**: {row['readability']:.2f} | **Quality**: {row['quality_score']}%")
            if row['ambiguous_phrases']:
                st.warning(f"⚠️ Ambiguous phrases: {', '.join(row['ambiguous_phrases'])}")
                for tip in row['suggestions']:
                    st.markdown(f"👉 {tip}")

        st.subheader("📊 Visualization")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['quality_score'], bins=10, kde=True, ax=ax1, color='orange')
        ax1.set_title("Quality Scores")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='token_count', y='readability', ax=ax2)
        ax2.set_title("Token Count vs Readability")
        st.pyplot(fig2)

        st.subheader("📥 Export")
        csv = df.to_csv(index=False)
        st.download_button("Download Chunk Report CSV", csv, "chunk_report.csv", "text/csv")

        st.success("Analysis complete! Improve the flagged areas for better AI understanding.")
