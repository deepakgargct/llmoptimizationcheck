import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from textstat import flesch_reading_ease
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
import re

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Setup
stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")
DANDELION_TOKEN = st.secrets.get("DANDELION_API_KEY")

AMBIGUOUS_PHRASES = {
    "overview": "Use specific headings like 'SEO strategy breakdown' or 'Audit Summary'.",
    "click here": "Use descriptive anchor text like 'See SEO case study'.",
    "read more": "Use direct phrasing like 'Explore keyword optimization methods'.",
    "details": "Replace with 'Technical SEO breakdown' or similar.",
    "learn more": "Be specific: e.g. 'Learn more about schema implementation'.",
    "start now": "Clarify what user is starting: 'Start optimizing your blog SEO'.",
    "some believe": "Avoid vague phrasing. Be precise or cite a source.",
    "it could be argued": "Replace with concrete data or authoritative quote."
}

EXCLUDED_TAGS = ['header', 'footer', 'nav', 'aside']

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
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        st.error(f"Error fetching content: {e}")
        return None

def extract_entities(text):
    if not DANDELION_TOKEN:
        return []
    try:
        payload = {
            'text': text,
            'lang': 'en',
            'token': DANDELION_TOKEN
        }
        r = requests.get("https://api.dandelion.eu/datatxt/nex/v1", params=payload)
        if r.status_code == 200:
            data = r.json()
            return [e['spot'] for e in data.get('annotations', [])]
    except Exception as e:
        st.error(f"Entity extraction failed: {e}")
    return []

def extract_chunks(soup):
    for tag in EXCLUDED_TAGS:
        [el.decompose() for el in soup.find_all(tag)]

    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            quality_score = 100
            ambiguous_hits = []
            fixes = []

            for phrase, suggestion in AMBIGUOUS_PHRASES.items():
                if phrase in text.lower():
                    ambiguous_hits.append(phrase)
                    fixes.append(f"👉 **{phrase}** → {suggestion}")

            if tokens > 300: quality_score -= 10
            if readability < 60: quality_score -= 15
            if ambiguous_hits: quality_score -= 10 * len(ambiguous_hits)

            entities = extract_entities(text) if DANDELION_TOKEN else []

            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': ambiguous_hits,
                'ambiguous_fixes': fixes,
                'quality_score': max(0, quality_score),
                'entities': entities
            })
    return chunks

def generate_recommendations(chunk):
    recs = []
    if chunk['readability'] < 60:
        recs.append("Improve sentence clarity and reduce jargon.")
    if chunk['token_count'] > 300:
        recs.append("Split this chunk into smaller logical sections.")
    if not chunk['entities']:
        recs.append("Include industry-relevant entities or terminology.")
    if not chunk['ambiguous_phrases']:
        recs.append("Consider adding direct answers or FAQs for LLM readiness.")
    return recs

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Checker", layout="wide")
st.title("🧠 GenAI Optimization Checker")

with st.sidebar:
    method = st.radio("Input method", ["URL", "HTML file", "Raw HTML", ".txt file"])
    html_content = ""

    if method == "URL":
        url = st.text_input("Enter URL")
        if st.button("Fetch URL"):
            soup = fetch_url_content(url)
            html_content = soup.prettify() if soup else ""
    elif method == "HTML file":
        f = st.file_uploader("Upload HTML", type=["html"])
        if f: html_content = f.read().decode("utf-8")
    elif method == "Raw HTML":
        html_content = st.text_area("Paste HTML here")
    elif method == ".txt file":
        f = st.file_uploader("Upload .txt", type=["txt"])
        if f: html_content = f.read().decode("utf-8")

if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = extract_chunks(soup)

    if not chunks:
        st.warning("No valid content chunks found.")
    else:
        df = pd.DataFrame(chunks)

        tabs = st.tabs(["🔍 Analysis", "📌 Recommendations", "📊 Visualizations", "📘 Entities", "📤 Export"])

        with tabs[0]:
            st.subheader("Chunk Analysis")
            for i, chunk in enumerate(chunks):
                st.markdown(f"### Chunk {i+1}")
                st.markdown(f"**Tokens:** {chunk['token_count']} | **Readability:** {chunk['readability']:.2f} | **Quality Score:** {chunk['quality_score']}%")
                st.text(chunk['text'])
                if chunk['ambiguous_phrases']:
                    for fix in chunk['ambiguous_fixes']:
                        st.warning(fix)

        with tabs[1]:
            st.subheader("📌 AI/LLM Optimization Recommendations")
            for i, chunk in enumerate(chunks):
                recs = generate_recommendations(chunk)
                st.markdown(f"**Chunk {i+1} Suggestions:**")
                for rec in recs:
                    st.markdown(f"- {rec}")

        with tabs[2]:
            st.subheader("📊 Visualization")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['quality_score'], bins=10, kde=True, ax=ax1, color='green')
            ax1.set_title("Quality Score Distribution")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x="token_count", y="readability", ax=ax2)
            ax2.set_title("Tokens vs Readability")
            st.pyplot(fig2)

        with tabs[3]:
            st.subheader("📘 Entities Found (via Dandelion)")
            for i, chunk in enumerate(chunks):
                if chunk['entities']:
                    st.markdown(f"**Chunk {i+1} Entities:** {', '.join(chunk['entities'])}")

        with tabs[4]:
            st.subheader("📤 Export Flagged Chunks (Quality < 70)")
            flagged = df[df['quality_score'] < 70]
            if not flagged.empty:
                st.dataframe(flagged)
                csv = flagged.to_csv(index=False).encode('utf-8')
                st.download_button("Download as CSV", csv, "flagged_chunks.csv", "text/csv")
            else:
                st.success("No poorly scored chunks found!")

st.caption("LLM + SEO Optimizer | Highlights structure, quality, ambiguous phrasing, and retrieval readiness")
