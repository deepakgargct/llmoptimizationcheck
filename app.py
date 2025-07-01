import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from textstat import flesch_reading_ease
from nltk.tokenize import sent_tokenize
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

# Config
DANDELION_TOKEN = st.secrets["DANDELION_API_KEY"]
stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")
EXCLUDED_TAGS = ['header', 'footer', 'nav', 'aside']

AMBIGUOUS_PHRASES = {
    "click here": "Use descriptive anchor text like 'See our pricing plans'",
    "learn more": "Specify what the user will learn, e.g., 'Learn more about SEO audit costs'",
    "read more": "Provide context like 'Read more about on-page SEO'",
    "overview": "Use specific headings like 'SEO strategy breakdown'",
    "details": "Replace with specifics like 'schema markup details'",
    "more info": "Add clarity, e.g., 'More info on canonical URLs'",
    "start now": "Explain what starts—e.g., 'Start optimizing your metadata now'",
    "this article": "Replace with the article title or topic",
    "the following": "Use clear transitions or lists",
    "some believe": "Avoid hedging—state factual backing or sources",
    "it could be argued": "Clarify who argues it and why, or rephrase objectively"
}

# --- Utilities ---
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
        return BeautifulSoup(response.content, 'html.parser')
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
            amb = [phrase for phrase in AMBIGUOUS_PHRASES if phrase in text.lower()]
            quality = 100 - (15 if tokens > 300 else 0) - (15 if readability < 60 else 0) - (10 * len(amb))
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': amb,
                'quality_score': max(0, quality),
                'optim_tip': generate_llm_recommendation(text)
            })
    return chunks

def generate_llm_recommendation(text):
    tips = []
    if '?' in text or re.search(r'(how|why|what|when|where)', text.lower()):
        tips.append("✅ Good use of natural questions")
    if '<h2>' not in text and len(text.split()) > 40:
        tips.append("🔧 Add H2 subheadings to break long paragraphs")
    if 'schema.org' not in text.lower():
        tips.append("🧩 Consider adding relevant schema markup")
    return " | ".join(tips) if tips else "No specific tips."

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
    return pd.DataFrame([c for c in chunks if c['quality_score'] < 70])

def extract_entities(text):
    try:
        url = "https://api.dandelion.eu/datatxt/nex/v1/"
        params = {
            'lang': 'en',
            'text': text,
            'include': 'types',
            'token': DANDELION_TOKEN
        }
        response = requests.get(url, params=params)
        data = response.json()
        return [ent['spot'] for ent in data.get('annotations', [])]
    except:
        return []

# --- Streamlit App ---
st.set_page_config(page_title="GenAI Optimization Checker", layout="wide")
st.title("GenAI Optimization Checker")

# Input
with st.sidebar:
    method = st.radio("Input Method", ["URL", "HTML file", "Raw HTML", ".txt file"])
    html_content = ""
    if method == "URL":
        url = st.text_input("Enter URL:")
        if st.button("Fetch URL"):
            soup = fetch_url_content(url)
            html_content = soup.prettify() if soup else ""
    elif method == "HTML file":
        file = st.file_uploader("Upload HTML file", type=["html"])
        if file:
            html_content = file.read().decode("utf-8")
    elif method == "Raw HTML":
        html_content = st.text_area("Paste raw HTML:")
    elif method == ".txt file":
        file = st.file_uploader("Upload .txt file", type=["txt"])
        if file:
            html_content = file.read().decode("utf-8")

if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = extract_chunks(soup)

    tabs = st.tabs(["🔍 Analysis", "📘 Glossary", "📌 Takeaways", "📊 Visuals", "🧠 AI Tips", "📤 Export"])

    with tabs[0]:
        st.subheader("Content Chunks")
        for i, c in enumerate(chunks):
            st.markdown(f"### Chunk {i+1}")
            st.markdown(f"**Tokens:** {c['token_count']} | **Readability:** {c['readability']:.1f} | **Score:** {c['quality_score']}%")
            st.text(c['text'])
            if c['ambiguous_phrases']:
                for phrase in c['ambiguous_phrases']:
                    st.warning(f"⚠️ Ambiguous: '{phrase}' → 💡 {AMBIGUOUS_PHRASES[phrase]}")
            st.info(f"LLM Tip: {c['optim_tip']}")
            entities = extract_entities(c['text'])
            if entities:
                st.success(f"Entities: {', '.join(entities)}")

    with tabs[1]:
        st.subheader("Glossary Terms")
        glossary = extract_glossary(chunks)
        if glossary:
            st.dataframe(pd.DataFrame(glossary))
        else:
            st.info("No glossary terms found.")

    with tabs[2]:
        st.subheader("Key Takeaways")
        for i, tk in enumerate(get_key_takeaways(chunks)):
            st.markdown(f"**{i+1}.** {tk['text']}")

    with tabs[3]:
        st.subheader("Visualizations")
        df = pd.DataFrame(chunks)
        fig, ax = plt.subplots()
        sns.histplot(df['quality_score'], kde=True, bins=10, ax=ax, color='skyblue')
        ax.set_title("Content Quality Distribution")
        st.pyplot(fig)

    with tabs[4]:
        st.subheader("LLM Optimization Summary")
        st.markdown("""
        ✅ Best practices:
        - Use clear H2/H3 headers with natural queries
        - Add relevant schema (e.g., FAQ, Article, HowTo)
        - Use synonyms, rephrase key points
        - Keep chunks below 300 tokens
        - Avoid vague anchors and generic headings
        """)
        st.markdown("🚀 Try checking your keywords on [Perplexity.ai](https://www.perplexity.ai) or [Brave Search](https://search.brave.com) after 7 days to validate inclusion in AI answers.")

    with tabs[5]:
        st.subheader("Export Flagged Content")
        flagged_df = export_flagged_chunks(chunks)
        if not flagged_df.empty:
            st.dataframe(flagged_df)
            st.download_button("Download CSV", flagged_df.to_csv(index=False).encode('utf-8'), "flagged_chunks.csv", "text/csv")
        else:
            st.success("No flagged chunks.")

st.caption("🧠 Built for optimizing content for GenAI and LLM visibility.")
