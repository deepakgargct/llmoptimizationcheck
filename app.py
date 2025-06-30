import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from textstat import flesch_reading_ease
import json
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

# === Naive tokenizers to avoid nltk punkt errors ===
def naive_sent_tokenize(text):
    return re.split(r'(?<=[.!?]) +', text)

def naive_word_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Streamlit App Configuration ===
st.set_page_config(page_title="GenAI Optimization Streamlit App", layout="wide")
st.title("GenAI Optimization Checker")

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Constants ===
tokenizer = tiktoken.get_encoding("cl100k_base")
SEMANTIC_MAP = {
    "section": "Generic Section",
    "article": "Article or Post",
    "aside": "Sidebar/Comment",
    "header": "Header",
    "footer": "Footer",
    "nav": "Navigation",
    "main": "Main Content",
    "figure": "Image or Chart Block",
    "blockquote": "Quote or Testimonial"
}

# === Helpers ===
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

def parse_uploaded_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        return BeautifulSoup(content, 'html.parser')
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def extract_chunks(soup):
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(naive_word_tokenize(text)) > 30:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            semantic_label = SEMANTIC_MAP.get(tag.name, f"{tag.name.title()} Block")
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'semantic_role': semantic_label
            })
    return chunks

def extract_basic_entities(text):
    tokens = naive_word_tokenize(text)
    return list(set([t for t in tokens if t[0].isupper() and len(t) > 2]))

# === UI Input Options ===
st.sidebar.header("Input Options")
input_type = st.sidebar.radio("Choose input type:", ["Enter URL", "Upload HTML File", "Paste HTML Code", "Upload .txt File"])

soup = None
if input_type == "Enter URL":
    url = st.sidebar.text_input("Enter a URL:", "https://example.com")
    if st.sidebar.button("Fetch from URL"):
        soup = fetch_url_content(url)
elif input_type == "Upload HTML File":
    uploaded_file = st.sidebar.file_uploader("Upload an HTML file", type=["html", "htm"])
    if uploaded_file:
        soup = parse_uploaded_file(uploaded_file)
elif input_type == "Paste HTML Code":
    html_code = st.sidebar.text_area("Paste HTML code here:", height=300)
    if st.sidebar.button("Analyze HTML"):
        soup = BeautifulSoup(html_code, 'html.parser')
elif input_type == "Upload .txt File":
    uploaded_txt = st.sidebar.file_uploader("Upload a .txt file", type="txt")
    if uploaded_txt:
        try:
            text = uploaded_txt.read().decode("utf-8")
            soup = BeautifulSoup(f"<p>{text}</p>", 'html.parser')
        except Exception as e:
            st.error(f"Error reading .txt file: {e}")

if soup:
    chunks = extract_chunks(soup)
    if not chunks:
        st.warning("No significant content chunks found.")
    else:
        st.subheader("🧩 Content Chunks Analysis")
        oversized_chunks = 0
        low_readability_chunks = 0

        for i, chunk in enumerate(chunks):
            st.markdown(f"### Chunk {i+1}: {chunk['semantic_role']}")
            st.markdown(f"**Tokens:** {chunk['token_count']}, **Readability:** {chunk['readability']:.2f}")
            st.text(chunk['text'])

            with st.expander("Detected Entities"):
                entities = extract_basic_entities(chunk['text'])
                st.write(entities if entities else "No entities detected.")

            if chunk['token_count'] > 300:
                oversized_chunks += 1
            if chunk['readability'] < 60:
                low_readability_chunks += 1

        # === Visualizations ===
        st.subheader("📊 Chunk Metrics Visualization")
        df = pd.DataFrame(chunks)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df['token_count'], bins=15, kde=False, ax=ax, color='skyblue')
        ax.set_title("Token Count per Chunk")
        ax.set_xlabel("Token Count")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(df['readability'], bins=15, kde=True, ax=ax2, color='lightgreen')
        ax2.set_title("Readability Score per Chunk")
        ax2.set_xlabel("Flesch Reading Ease")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        # === Recommendations ===
        st.subheader("🧠 Optimization Summary & Recommendations")
        if oversized_chunks > 0:
            st.markdown(f"- ✂️ **{oversized_chunks} content chunks exceed 300 tokens**. Consider splitting them for better LLM indexing.")
        if low_readability_chunks > 0:
            st.markdown(f"- 📉 **{low_readability_chunks} chunks have low readability scores (<60)**. Simplify language for easier LLM understanding.")
        if oversized_chunks == 0 and low_readability_chunks == 0:
            st.markdown("✅ All content chunks are well-structured and optimized for LLMs.")

        st.markdown("---")
        st.markdown("### 🔧 Further Optimization Suggestions")
        st.markdown("- Add schema.org JSON-LD markup to improve semantic clarity.")
        st.markdown("- Ensure the `robots.txt` and `llms.txt` files do not block AI crawlers.")
        st.markdown("- Use more descriptive internal link anchor text.")
        st.markdown("- Cover related entities or topics missing from detected entity set.")

        st.success("Analysis complete. Use this to improve LLM crawlability and visibility.")

st.markdown("---")
st.caption("Built for GenAI content optimization. Supports semantic tags and basic entity enrichment.")
