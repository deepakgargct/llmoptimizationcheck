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

QUESTION_TYPES = ["what", "how", "why", "when", "who", "which", "can", "does", "do", "is", "are", "should"]


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

def is_self_contained(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return False
    pronouns = ['this', 'that', 'these', 'those', 'it', 'they', 'them']
    first_sentence = sentences[0].lower()
    return not any(pronoun in first_sentence.split()[:3] for pronoun in pronouns)

def is_query_like_heading(text):
    text = text.lower().strip(" ?")
    return any(text.startswith(q) for q in QUESTION_TYPES)

def extract_chunks(soup):
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 20:
            sentences = sent_tokenize(text)
            chunk_text = ""
            for sentence in sentences:
                chunk_text += sentence + " "
                token_len = count_tokens(chunk_text.strip())
                if token_len >= 100:
                    trimmed = chunk_text.strip()
                    if token_len <= 300:
                        readability = analyze_readability(trimmed)
                        semantic_label = SEMANTIC_MAP.get(tag.name, f"{tag.name.title()} Block")
                        chunks.append({
                            'text': trimmed[:300] + '...' if len(trimmed) > 300 else trimmed,
                            'token_count': token_len,
                            'readability': readability,
                            'semantic_role': semantic_label,
                            'self_contained': is_self_contained(trimmed),
                            'query_like': is_query_like_heading(trimmed.split(".")[0])
                        })
                    chunk_text = ""
    return chunks

def analyze_internal_links(soup, base_url):
    links = []
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == base_domain:
            anchor = a.get_text(strip=True)
            links.append({
                'url': full_url,
                'anchor_text': anchor,
                'descriptive': len(anchor.split()) > 2
            })
    return links

def extract_entities_basic(text):
    tokens = word_tokenize(text)
    return list(set([w for w in tokens if w.istitle() and w.lower() not in stop_words and len(w) > 3]))

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Streamlit App", layout="wide")
st.title("GenAI Optimization Checker")

url = st.text_input("Enter a URL to analyze:", "https://example.com")
run_button = st.button("Analyze")

if run_button:
    with st.spinner("Fetching and analyzing content..."):
        soup = fetch_url_content(url)
        if soup:
            chunks = extract_chunks(soup)
            internal_links = analyze_internal_links(soup, url)

            if not chunks:
                st.warning("No significant content chunks found.")
            else:
                st.subheader("Content Chunks")
                oversized_chunks = 0
                low_readability_chunks = 0
                missing_query_headings = 0
                not_self_contained = 0
                all_entities = set()

                for i, chunk in enumerate(chunks):
                    st.markdown(f"### Chunk {i+1}: {chunk['semantic_role']}")
                    st.markdown(f"**Tokens:** {chunk['token_count']}, **Readability:** {chunk['readability']:.2f}")
                    st.text(chunk['text'])

                    entities = extract_entities_basic(chunk['text'])
                    all_entities.update(entities)
                    if entities:
                        st.markdown("**Entities:**")
                        st.write(entities)

                    if not chunk['self_contained']:
                        not_self_contained += 1
                    if not chunk['query_like']:
                        missing_query_headings += 1
                    if chunk['token_count'] > 300:
                        oversized_chunks += 1
                    if chunk['readability'] < 60:
                        low_readability_chunks += 1

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

                st.subheader("🧠 Optimization Summary & Recommendations")
                if oversized_chunks > 0:
                    st.markdown(f"- ✂️ **{oversized_chunks} chunks exceed 300 tokens**. Consider splitting them.")
                if low_readability_chunks > 0:
                    st.markdown(f"- 📉 **{low_readability_chunks} chunks have low readability scores (<60)**. Simplify language.")
                if not_self_contained > 0:
                    st.markdown(f"- 🔄 **{not_self_contained} chunks lack self-contained logic**. Clarify context or avoid pronouns early on.")
                if missing_query_headings > 0:
                    st.markdown(f"- ❓ **{missing_query_headings} sections don't start with query-like phrases**. Consider rephrasing.")
                if len(internal_links) > 0:
                    weak_anchors = [l for l in internal_links if not l['descriptive']]
                    if weak_anchors:
                        st.markdown(f"- 🔗 **{len(weak_anchors)} internal links use vague anchor text.** Use more descriptive labels.")
                if oversized_chunks + low_readability_chunks + not_self_contained + missing_query_headings == 0:
                    st.markdown("✅ All content chunks are well-structured and optimized.")

                st.markdown("---")
                st.markdown("### 🔧 Further Optimization Suggestions")
                st.markdown("- Add schema.org JSON-LD markup to improve semantic clarity.")
                st.markdown("- Ensure the `robots.txt` and `llms.txt` files do not block AI crawlers.")
                st.markdown("- Use more descriptive internal link anchor text.")
                st.markdown("- Cover related entities or topics missing from detected entity set.")

                st.success("Analysis complete. Use this to improve LLM crawlability and visibility.")

                st.subheader("📚 All Unique Entities Detected")
                if all_entities:
                    st.write(sorted(all_entities))
                else:
                    st.write("No named entities detected.")

st.markdown("---")
st.caption("Built for GenAI content optimization. Supports semantic tags, link analysis, and basic entity recognition.")
