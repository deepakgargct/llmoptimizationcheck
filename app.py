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
import io
import spacy
from collections import Counter
import re

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
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

def extract_text_from_input(source_type, url=None, uploaded_file=None, html_code=None):
    if source_type == "URL" and url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            if 'text/html' not in response.headers.get('Content-Type', ''):
                return None
            return BeautifulSoup(response.content, 'html.parser')
        except:
            return None
    elif source_type == "Upload HTML File" and uploaded_file:
        content = uploaded_file.read()
        return BeautifulSoup(content, 'html.parser')
    elif source_type == "Paste HTML Code" and html_code:
        return BeautifulSoup(html_code, 'html.parser')
    elif source_type == "Upload .txt File" and uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        soup = BeautifulSoup("<p>" + content + "</p>", 'html.parser')
        return soup
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
                'semantic_role': semantic_label,
                'raw_text': text
            })
    return chunks

def extract_basic_entities(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents]))

def detect_passive_voice(text):
    doc = nlp(text)
    passive_sentences = [sent.text for sent in doc.sents if any(tok.dep_ == "auxpass" for tok in sent)]
    return passive_sentences

def find_vague_phrases(text):
    vague_terms = ["some believe", "it could be argued", "might be", "perhaps", "suggests that"]
    found = [phrase for phrase in vague_terms if phrase in text.lower()]
    return found

def glossary_candidates(chunks):
    all_text = " ".join(chunk['raw_text'] for chunk in chunks)
    words = [w for w in word_tokenize(all_text) if w.isalpha() and w.lower() not in stop_words]
    freq = Counter(words)
    return [word for word, count in freq.items() if len(word) > 7 and count > 1]

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Streamlit App", layout="wide")
st.title("GenAI Optimization Checker")

source_type = st.radio("Select input method:", ["URL", "Upload HTML File", "Paste HTML Code", "Upload .txt File"])
url = uploaded_file = html_code = None

if source_type == "URL":
    url = st.text_input("Enter URL:")
elif source_type == "Upload HTML File":
    uploaded_file = st.file_uploader("Upload HTML File", type=["html"])
elif source_type == "Paste HTML Code":
    html_code = st.text_area("Paste your HTML code here:")
elif source_type == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload .txt File", type=["txt"])

run_button = st.button("Analyze")

if run_button:
    with st.spinner("Analyzing content..."):
        soup = extract_text_from_input(source_type, url, uploaded_file, html_code)
        if soup:
            chunks = extract_chunks(soup)
            if not chunks:
                st.warning("No significant content chunks found.")
            else:
                st.subheader("📘 Content Chunks")
                oversized_chunks = 0
                low_readability_chunks = 0
                passive_warnings = []
                vague_flags = []

                for i, chunk in enumerate(chunks):
                    st.markdown(f"### Chunk {i+1}: {chunk['semantic_role']}")
                    st.markdown(f"**Tokens:** {chunk['token_count']} | **Readability:** {chunk['readability']:.2f}")
                    st.text(chunk['text'])

                    with st.expander("🔍 Issues & Suggestions"):
                        passive = detect_passive_voice(chunk['raw_text'])
                        vague = find_vague_phrases(chunk['raw_text'])
                        if passive:
                            passive_warnings.extend(passive)
                            st.warning(f"⚠️ Passive voice detected in {len(passive)} sentence(s)")
                        if vague:
                            vague_flags.extend(vague)
                            st.warning(f"⚠️ Vague terms found: {', '.join(set(vague))}")

                    with st.expander("🧠 Entities Detected"):
                        entities = extract_basic_entities(chunk['raw_text'])
                        st.write(entities if entities else "No named entities found.")

                    if chunk['token_count'] > 300:
                        oversized_chunks += 1
                    if chunk['readability'] < 60:
                        low_readability_chunks += 1

                # Visualizations
                st.subheader("📊 Chunk Metrics Visualization")
                df = pd.DataFrame(chunks)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df['token_count'], bins=15, kde=False, ax=ax, color='skyblue')
                ax.set_title("Token Count per Chunk")
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                sns.histplot(df['readability'], bins=15, kde=True, ax=ax2, color='lightgreen')
                ax2.set_title("Readability Score per Chunk")
                st.pyplot(fig2)

                st.subheader("🧠 Optimization Summary & Recommendations")
                if oversized_chunks:
                    st.markdown(f"- ✂️ {oversized_chunks} chunk(s) exceed 300 tokens. Consider splitting.")
                if low_readability_chunks:
                    st.markdown(f"- 📉 {low_readability_chunks} chunk(s) have low readability scores (<60). Simplify language.")
                if passive_warnings:
                    st.markdown(f"- 🎯 Use active voice instead of passive in {len(passive_warnings)} sentence(s).")
                if vague_flags:
                    st.markdown(f"- 🚫 Replace vague terms such as {', '.join(set(vague_flags))} with more assertive alternatives.")

                glossary_terms = glossary_candidates(chunks)
                if glossary_terms:
                    st.markdown(f"- 📘 Consider adding a glossary for technical terms like: {', '.join(glossary_terms[:10])}...")

                st.markdown("---")
                st.markdown("### 🔧 Further Optimization Suggestions")
                st.markdown("- Add schema.org JSON-LD markup.")
                st.markdown("- Ensure robots.txt and llms.txt allow AI bots.")
                st.markdown("- Use descriptive internal anchor text.")
                st.markdown("- Break down flat paragraphs and avoid jargon.")
                st.markdown("- Consider FAQs, comparisons, glossaries, and use-case content blocks.")

                st.success("LLM optimization analysis complete.")
        else:
            st.error("❌ Could not parse content. Check your input.")

st.markdown("---")
st.caption("Built for GenAI content optimization. Supports URL, HTML, and TXT input with chunk-level analysis and entity detection.")
