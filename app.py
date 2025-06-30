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

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Setup
stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")

AMBIGUOUS_PHRASES = [
    "click here", "learn more", "read more", "overview", "details",
    "more info", "start now", "this article", "the following",
    "some believe", "it could be argued"
]

EXCLUDED_TAGS = ['header', 'footer', 'nav', 'aside']

# Utility Functions
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
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def extract_chunks(soup):
    for tag in EXCLUDED_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            ambiguous_hits = [p for p in AMBIGUOUS_PHRASES if p in text.lower()]
            quality_score = 100
            if tokens > 300:
                quality_score -= 15
            if readability < 60:
                quality_score -= 15
            if ambiguous_hits:
                quality_score -= 10 * len(ambiguous_hits)
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': ambiguous_hits,
                'quality_score': max(0, quality_score)
            })
    return chunks

def extract_glossary(chunks):
    glossary = []
    pattern = re.compile(r"(?P<term>[A-Z][a-zA-Z0-9\- ]+?)\s+(is|refers to|means|can be defined as)\s+(?P<definition>.+?)\.")
    for chunk in chunks:
        matches = pattern.findall(chunk['text'])
        for match in matches:
            glossary.append({'term': match[0].strip(), 'definition': match[2].strip()})
    return glossary

def get_key_takeaways(chunks, top_n=3):
    sorted_chunks = sorted(chunks, key=lambda x: x['quality_score'], reverse=True)
    return sorted_chunks[:top_n]

def export_flagged_chunks(chunks):
    df = pd.DataFrame([c for c in chunks if c['quality_score'] < 70])
    return df

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Checker", layout="wide")
st.title("GenAI Optimization Checker")

# Sidebar Input
with st.sidebar:
    option = st.radio("Select input method:", ["URL", "HTML file", "Raw HTML", ".txt file"])
    html_content = ""

    if option == "URL":
        url = st.text_input("Enter URL to analyze:")
        if st.button("Fetch URL"):
            soup = fetch_url_content(url)
            html_content = soup.prettify() if soup else ""
    elif option == "HTML file":
        uploaded_file = st.file_uploader("Upload HTML file", type=["html"])
        if uploaded_file:
            html_content = uploaded_file.read().decode("utf-8")
    elif option == "Raw HTML":
        html_content = st.text_area("Paste raw HTML code:")
    elif option == ".txt file":
        uploaded_txt = st.file_uploader("Upload .txt file", type="txt")
        if uploaded_txt:
            html_content = uploaded_txt.read().decode("utf-8")

# Main Content
if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = extract_chunks(soup)

    if not chunks:
        st.warning("No meaningful content chunks found.")
    else:
        tabs = st.tabs(["🔍 Analysis", "📘 Glossary", "📌 Key Takeaways", "📊 Visualizations", "📤 Export"])

        with tabs[0]:
            st.subheader("Chunk Analysis")
            for i, chunk in enumerate(chunks):
                st.markdown(f"### Chunk {i+1}")
                st.markdown(f"**Tokens:** {chunk['token_count']} | **Readability:** {chunk['readability']:.2f} | **Quality Score:** {chunk['quality_score']}%")
                st.text(chunk['text'])
                if chunk['ambiguous_phrases']:
                    st.warning(f"⚠️ Ambiguous phrases: {', '.join(chunk['ambiguous_phrases'])}")

        with tabs[1]:
            st.subheader("📘 Glossary Builder")
            glossary = extract_glossary(chunks)
            if glossary:
                glossary_df = pd.DataFrame(glossary)
                st.dataframe(glossary_df)
            else:
                st.info("No glossary terms found.")

        with tabs[2]:
            st.subheader("📌 Key Takeaways")
            takeaways = get_key_takeaways(chunks)
            for i, chunk in enumerate(takeaways):
                st.markdown(f"**Takeaway {i+1}:** {chunk['text']}")

        with tabs[3]:
            st.subheader("📊 Content Quality Visualizations")
            df = pd.DataFrame(chunks)

            fig1, ax1 = plt.subplots()
            sns.histplot(df['quality_score'], bins=10, kde=True, ax=ax1, color='orange')
            ax1.set_title("Chunk Quality Score Distribution")
            ax1.set_xlabel("Quality Score")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x="token_count", y="readability", ax=ax2)
            ax2.set_title("Token Count vs Readability")
            st.pyplot(fig2)

        with tabs[4]:
            st.subheader("📤 Export Flagged Chunks")
            flagged_df = export_flagged_chunks(chunks)
            if not flagged_df.empty:
                st.dataframe(flagged_df)
                csv = flagged_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Flagged Chunks as CSV", csv, "flagged_chunks.csv", "text/csv")
            else:
                st.success("No flagged chunks to export!")

st.caption("🧠 Built for GenAI + LLM optimization. Analyze clarity, structure, and chunk quality for better AI visibility.")
