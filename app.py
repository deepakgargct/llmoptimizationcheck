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
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Constants
DANDELION_TOKEN = "YOUR_DANDELION_API_KEY"
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

VAGUE_TERMS = ["some believe", "it could be argued", "many people think", "it seems that", "possibly", "might", "perhaps"]
PASSIVE_PATTERNS = re.compile(r"\b(is|are|was|were|be|been|being)\b\s+(\w+ed)\b")
ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,})\b")

# Helpers
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
        return BeautifulSoup(response.content, 'html.parser')
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

def glossary_builder(text):
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 6]
    common = Counter(filtered).most_common(10)
    return [word for word, _ in common]

def detect_passive_voice(text):
    return PASSIVE_PATTERNS.findall(text)

def detect_vague_phrases(text):
    found = [phrase for phrase in VAGUE_TERMS if phrase in text.lower()]
    return found

def find_acronyms(text):
    return list(set(ACRONYM_PATTERN.findall(text)))

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
            if not chunks:
                st.warning("No significant content chunks found.")
            else:
                df = pd.DataFrame(chunks)
                oversized_chunks = sum(c['token_count'] > 300 for c in chunks)
                low_readability_chunks = sum(c['readability'] < 60 for c in chunks)

                st.subheader("🔍 Content Quality Warnings")
                for i, chunk in enumerate(chunks):
                    st.markdown(f"#### Chunk {i+1}: {chunk['semantic_role']}")
                    vague = detect_vague_phrases(chunk['text'])
                    passive = detect_passive_voice(chunk['text'])
                    if vague:
                        st.markdown("**Vague Terms Detected:**")
                        st.write(vague)
                        st.markdown("*Suggestion: Replace with specific, assertive alternatives.*")
                    if passive:
                        st.markdown("**Passive Voice Phrases Detected:**")
                        st.write(passive)
                        st.markdown("*Suggestion: Rephrase in active voice.*")
                    if not vague and not passive:
                        st.markdown("✅ No quality issues detected in this chunk.")

                st.subheader("📚 Glossary Suggestions")
                all_text = " ".join([c['text'] for c in chunks])
                glossary_terms = glossary_builder(all_text)
                if glossary_terms:
                    st.write(glossary_terms)
                    st.markdown("*Suggestion: Define these terms in a glossary section.*")
                else:
                    st.write("No uncommon terms detected.")

                st.subheader("📈 Chunk Metrics Visualization")
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
                if oversized_chunks:
                    st.markdown(f"- ✂️ {oversized_chunks} content chunks exceed 300 tokens. Consider splitting.")
                if low_readability_chunks:
                    st.markdown(f"- 📉 {low_readability_chunks} chunks have low readability. Simplify language.")
                if not oversized_chunks and not low_readability_chunks:
                    st.markdown("✅ All content chunks are well-structured.")

                st.markdown("---")
                st.markdown("### 🔧 Further Optimization Suggestions")
                st.markdown("- Add schema.org JSON-LD markup.")
                st.markdown("- Check robots.txt and llms.txt files.")
                st.markdown("- Use consistent, descriptive internal anchor texts.")
                st.markdown("- Link glossary terms internally for semantic clarity.")

                st.success("Analysis complete.")

st.markdown("---")
st.caption("Built for GenAI content optimization. Now includes quality checker and glossary builder.")
