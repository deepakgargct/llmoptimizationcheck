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
import re

nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")

# Semantic tag mapping
SEMANTIC_MAP = {
    "section": "Generic Section",
    "article": "Article or Post",
    "aside": "Sidebar/Comment",
    "header": "Header",
    "footer": "Footer",
    "nav": "Navigation",
    "main": "Main Content",
    "figure": "Image or Chart Block",
    "blockquote": "Quote or Testimonial",
    "p": "Paragraph",
    "div": "Content Block",
    "h2": "Subheading",
    "h3": "Sub-subheading"
}

# Load API key securely
DANDELION_TOKEN = st.secrets.get("DANDELION_API_KEY", "")

def count_tokens(text):
    return len(tokenizer.encode(text))

def analyze_readability(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0.0

def extract_entities_dandelion(text):
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

def fetch_content_from_input(input_mode):
    if input_mode == "URL":
        url = st.text_input("Enter a URL:")
        if st.button("Fetch from URL") and url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                if 'text/html' in response.headers.get('Content-Type', ''):
                    return BeautifulSoup(response.content, 'html.parser')
                else:
                    st.warning("URL did not return HTML content.")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

    elif input_mode == "Upload HTML File":
        uploaded_file = st.file_uploader("Upload an HTML file", type=["html"])
        if uploaded_file:
            return BeautifulSoup(uploaded_file.read(), 'html.parser')

    elif input_mode == "Paste HTML Code":
        html_code = st.text_area("Paste raw HTML code:")
        if st.button("Parse HTML") and html_code:
            return BeautifulSoup(html_code, 'html.parser')

    elif input_mode == "Upload .txt File":
        txt_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if txt_file:
            raw_text = txt_file.read().decode("utf-8")
            soup = BeautifulSoup("<p>" + raw_text.replace("\n", "</p><p>") + "</p>", 'html.parser')
            return soup

    return None

def extract_chunks(soup):
    chunks = []
    glossary_terms = set()

    for tag in soup.find_all(['h2', 'h3', 'section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 20:
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            tag_name = tag.name
            semantic_label = SEMANTIC_MAP.get(tag_name, f"{tag_name.title()} Block")
            
            # Glossary heuristic (detect capitalized or technical terms)
            glossary_terms.update(re.findall(r'\\b[A-Z][a-z]{2,}(?:\\s+[A-Z][a-z]{2,})*\\b', text))

            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'semantic_role': semantic_label,
                'is_self_contained': text.count('.') >= 2
            })

    return chunks, sorted(glossary_terms)

# UI
st.set_page_config(page_title="GenAI Optimization Toolkit", layout="wide")
st.title("🔎 GenAI Optimization Checker")

input_mode = st.radio("Choose content input method:", ["URL", "Upload HTML File", "Paste HTML Code", "Upload .txt File"])
soup = fetch_content_from_input(input_mode)

if soup:
    with st.spinner("Analyzing content..."):
        chunks, glossary = extract_chunks(soup)

        if not chunks:
            st.warning("No significant content chunks found.")
        else:
            st.subheader("📚 Individual Chunk Reports")
            oversized = low_readability = warnings = 0

            for i, chunk in enumerate(chunks):
                st.markdown(f"#### Chunk {i+1} - {chunk['semantic_role']}")
                st.markdown(f"**Tokens:** {chunk['token_count']} | **Readability:** {chunk['readability']:.2f} | **Self-contained:** {'✅' if chunk['is_self_contained'] else '⚠️'}")
                st.text(chunk['text'])
                entities = extract_entities_dandelion(chunk['text'])
                with st.expander("Entities"):
                    st.write(entities if entities else "No entities detected")

                if chunk['token_count'] > 300:
                    oversized += 1
                if chunk['readability'] < 60:
                    low_readability += 1
                if not chunk['is_self_contained']:
                    warnings += 1

            st.subheader("📊 Chunk Score Visualizations")
            df = pd.DataFrame(chunks)
            fig1, ax1 = plt.subplots()
            sns.histplot(df['token_count'], bins=10, ax=ax1, color='skyblue')
            ax1.set_title("Token Count per Chunk")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.histplot(df['readability'], bins=10, ax=ax2, color='lightgreen')
            ax2.set_title("Readability Distribution")
            st.pyplot(fig2)

            st.subheader("🧠 Summary & Warnings")
            if oversized:
                st.markdown(f"- ✂️ **{oversized} chunks exceed 300 tokens.** Consider splitting.")
            if low_readability:
                st.markdown(f"- 📉 **{low_readability} chunks have poor readability (<60).** Simplify the language.")
            if warnings:
                st.markdown(f"- 🧩 **{warnings} chunks are not self-contained.** Consider improving clarity.")
            if not (oversized or low_readability or warnings):
                st.success("All content chunks are within optimal ranges!")

            st.subheader("📘 Glossary Terms Detected")
            if glossary:
                st.write(glossary)
            else:
                st.write("No glossary candidates detected.")

            st.markdown("---")
            st.markdown("### 🔧 Further Optimization Suggestions")
            st.markdown("- Add schema.org markup for clarity.")
            st.markdown("- Ensure robots.txt and llms.txt allow GPTBot, Google-Extended, etc.")
            st.markdown("- Add bylines with credentials and update dates.")
            st.markdown("- Build FAQ, how-to, and comparison blocks with headings that reflect queries.")

            st.success("Analysis Complete ✅")

st.markdown("---")
st.caption("Built for GenAI optimization and AI visibility. Supports semantic tags, glossary extraction, and chunk validation.")
