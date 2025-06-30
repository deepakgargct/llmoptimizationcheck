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
import logging

nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DANDELION_TOKEN = "YOUR_DANDELION_API_KEY"  # Replace with your API key

stop_words = set(stopwords.words('english'))
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
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            st.warning(f"Unsupported content type: {content_type}")
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def extract_chunks(soup):
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            tag_name = tag.name
            semantic_label = SEMANTIC_MAP.get(tag_name, f"{tag_name.title()} Block")
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'semantic_role': semantic_label
            })
    return chunks

def extract_entities(text):
    try:
        url = "https://api.dandelion.eu/datatxt/nex/v1"
        params = {
            'text': text,
            'lang': 'en',
            'include': 'types,categories,lod',
            'token': DANDELION_TOKEN
        }
        response = requests.get(url, params=params)
        data = response.json()
        entities = [ann['spot'] for ann in data.get('annotations', [])]
        return list(set(entities))
    except Exception as e:
        st.error(f"Entity extraction error: {e}")
        return []

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Streamlit App", layout="wide")
st.title("GenAI Optimization Checker")

mode = st.radio("Choose content input mode:", ["Fetch from URL", "Paste HTML manually", "Upload .txt or .html file"])
html = ""

if mode == "Fetch from URL":
    url = st.text_input("Enter a URL to analyze:", "https://example.com")
    run_button = st.button("Analyze")
    if run_button:
        with st.spinner("Fetching and analyzing content..."):
            soup = fetch_url_content(url)
            if not soup or not soup.get_text(strip=True):
                st.warning("⚠️ Page content could not be extracted. Try the manual paste or upload option.")
            else:
                html = str(soup)

elif mode == "Paste HTML manually":
    html = st.text_area("Paste full HTML content here:", height=300)
    run_button = st.button("Analyze Pasted Content")

elif mode == "Upload .txt or .html file":
    uploaded_file = st.file_uploader("Upload .txt or .html file", type=["txt", "html"])
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        html = file_content
        run_button = st.button("Analyze Uploaded File")
    else:
        run_button = False

if run_button and html:
    soup = BeautifulSoup(html, 'html.parser')
    chunks = extract_chunks(soup)
    if not chunks:
        st.warning("No significant content chunks found.")
    else:
        st.subheader("Content Chunks")
        oversized_chunks = 0
        low_readability_chunks = 0

        for i, chunk in enumerate(chunks):
            st.markdown(f"### Chunk {i+1}: {chunk['semantic_role']}")
            st.markdown(f"**Tokens:** {chunk['token_count']}, **Readability:** {chunk['readability']:.2f}")
            st.text(chunk['text'])
            with st.expander("Entities Detected"):
                entities = extract_entities(chunk['text'])
                if entities:
                    st.write(entities)
                else:
                    st.write("No entities detected.")

            if chunk['token_count'] > 300:
                oversized_chunks += 1
            if chunk['readability'] < 60:
                low_readability_chunks += 1

        # Visualizations
        st.subheader("🔍 Chunk Metrics Visualization")
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

        # Recommendations Summary
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
st.caption("Built for GenAI content optimization. Supports semantic tags and Dandelion entity enrichment.")
