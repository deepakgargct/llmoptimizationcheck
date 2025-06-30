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

def parse_html(content):
    return BeautifulSoup(content, 'html.parser')

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

def extract_internal_links(soup, base_url):
    base_domain = urlparse(base_url).netloc
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == base_domain:
            anchor_text = a.get_text(strip=True)
            links.append({
                'anchor_text': anchor_text,
                'descriptive': len(anchor_text.split()) > 2,
                'url': full_url
            })
    return links

def extract_basic_entities(text):
    tokens = word_tokenize(text)
    return list(set([word for word in tokens if word[0].isupper() and word.lower() not in stop_words and len(word) > 2]))

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Streamlit App", layout="wide")
st.title("GenAI Optimization Checker")

st.markdown("### Input Options")
input_type = st.radio("Choose input type:", ["Enter URL", "Upload HTML File", "Paste HTML Code", "Upload .txt File"])
html_content = ""
base_url = "https://example.com"

if input_type == "Enter URL":
    url = st.text_input("Enter a URL to analyze:", "https://example.com")
    if st.button("Fetch URL"):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            html_content = response.text
            base_url = url
        except Exception as e:
            st.error(f"Error fetching URL: {e}")

elif input_type == "Upload HTML File":
    html_file = st.file_uploader("Upload an HTML file", type="html")
    if html_file:
        html_content = html_file.read().decode("utf-8")

elif input_type == "Paste HTML Code":
    html_content = st.text_area("Paste HTML code here")

elif input_type == "Upload .txt File":
    txt_file = st.file_uploader("Upload a text file", type="txt")
    if txt_file:
        html_content = f"<p>{txt_file.read().decode('utf-8')}</p>"

if html_content:
    soup = parse_html(html_content)
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
                entities = extract_basic_entities(chunk['text'])
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

        # Link Analysis
        internal_links = extract_internal_links(soup, base_url)
        vague_links = [l for l in internal_links if not l['descriptive']]

        # Recommendations Summary
        st.subheader("🧠 Optimization Summary & Recommendations")
        if oversized_chunks > 0:
            st.markdown(f"- ✂️ **{oversized_chunks} content chunks exceed 300 tokens**. Consider splitting them.")
        if low_readability_chunks > 0:
            st.markdown(f"- 📉 **{low_readability_chunks} chunks have low readability (<60)**. Simplify language.")
        if vague_links:
            st.markdown(f"- 🔗 **{len(vague_links)} internal links use vague anchor text.** Improve them.")
        if oversized_chunks == 0 and low_readability_chunks == 0 and not vague_links:
            st.markdown("✅ All content chunks are optimized.")

        st.markdown("---")
        st.markdown("### 🔧 Further Optimization Suggestions")
        st.markdown("- Add schema.org markup to improve content structure.")
        st.markdown("- Include 'robots.txt' and 'llms.txt' allowing AI bots.")
        st.markdown("- Add author name, update dates, and credential markup.")
        st.markdown("- Build content clusters and cross-link conceptually.")
        st.markdown("- Create glossaries, FAQs, and how-to blocks.")

        st.success("Analysis complete. Optimize based on these recommendations.")

st.markdown("---")
st.caption("Built for GenAI content optimization. Supports multiple input types and AI visibility checks.")
