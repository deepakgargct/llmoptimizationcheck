import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
import nltk
import tiktoken
import pandas as pd
import re

# --- One-time NLTK setup ---
def ensure_nltk_data():
    from nltk.data import find
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

ensure_nltk_data()

stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")

AMBIGUOUS_PHRASES = {
    "click here": "Use descriptive anchor text like 'Download SEO checklist'.",
    "learn more": "Specify the subject, e.g., 'Learn more about on-page SEO'.",
    "read more": "Clarify the content, e.g., 'Read more about LLM optimization techniques'.",
    "overview": "Use topic-specific phrasing like 'SEO strategy breakdown'.",
    "details": "Replace with specific terms like 'technical implementation details'.",
    "more info": "Clarify with 'Find more information on schema markup'.",
    "start now": "Use goal-oriented CTAs like 'Start your free SEO audit'.",
    "this article": "Rephrase to 'This guide on AI content optimization...'.",
    "the following": "Introduce items with clear categories, e.g., 'The following strategies:'.",
    "some believe": "Replace with 'Research shows...' or cite specific studies.",
    "it could be argued": "Use clear evidence or expert quotes."
}

EXCLUDED_TAGS = ['header', 'footer', 'nav', 'aside']

# --- Functions ---
def count_tokens(text):
    return len(tokenizer.encode(text))

def analyze_readability(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0.0

def extract_text_from_html(html_code):
    soup = BeautifulSoup(html_code, 'html.parser')
    for tag in EXCLUDED_TAGS:
        for element in soup.find_all(tag):
            element.decompose()
    return soup, soup.get_text()

def fetch_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def generate_llm_tip(text):
    tips = []
    lower_text = text.lower()
    if "?" in text or "faq" in lower_text:
        tips.append("Wrap this section with FAQ schema.")
    if any(keyword in lower_text for keyword in ["how to", "step-by-step", "guide", "steps"]):
        tips.append("Consider using HowTo schema markup.")
    if "vs" in lower_text or "compare" in lower_text or "comparison" in lower_text:
        tips.append("This section could benefit from a Comparison schema.")
    if len(text.split()) > 150:
        tips.append("Split into smaller paragraphs to improve LLM retrieval.")
    if not tips:
        tips.append("Use clearer structure or schema markup for better LLM optimization.")
    return " ".join(tips)

def extract_chunks(soup):
    seen = set()
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30:
            clean_text = re.sub(r'\s+', ' ', text.strip())
            if clean_text in seen:
                continue
            seen.add(clean_text)

            tokens = count_tokens(clean_text)
            readability = analyze_readability(clean_text)
            ambiguous_hits = [p for p in AMBIGUOUS_PHRASES if p in clean_text.lower()]
            quality_score = 100
            if tokens > 300:
                quality_score -= 15
            if readability < 60:
                quality_score -= 15
            if ambiguous_hits:
                quality_score -= 10 * len(ambiguous_hits)
            chunks.append({
                'text': clean_text[:300] + '...' if len(clean_text) > 300 else clean_text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': ambiguous_hits,
                'quality_score': max(0, quality_score),
                'llm_tip': generate_llm_tip(clean_text)
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
    return pd.DataFrame([c for c in chunks if c['quality_score'] < 70])

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="LLM Optimization Checker", layout="wide")
st.title("🧠 LLM Optimization & Content Analyzer")

input_mode = st.selectbox("Select Input Method", ["Webpage URL", "Upload .txt File", "Upload .html File", "Paste HTML Code", "Direct Text Input"])
soup = None

if input_mode == "Webpage URL":
    url = st.text_input("Enter URL:")
    if st.button("Analyze") and url:
        soup = fetch_url_content(url)

elif input_mode == "Upload .txt File":
    txt_file = st.file_uploader("Upload .txt", type=["txt"])
    if txt_file:
        content = txt_file.read().decode("utf-8")
        soup, _ = extract_text_from_html(f"<p>{content}</p>")

elif input_mode == "Upload .html File":
    html_file = st.file_uploader("Upload .html", type=["html"])
    if html_file:
        html = html_file.read().decode("utf-8")
        soup, _ = extract_text_from_html(html)

elif input_mode == "Paste HTML Code":
    raw_html = st.text_area("Paste full HTML code here:")
    if raw_html:
        soup, _ = extract_text_from_html(raw_html)

elif input_mode == "Direct Text Input":
    plain_text = st.text_area("Paste plain content here:")
    if plain_text:
        soup, _ = extract_text_from_html(f"<p>{plain_text}</p>")

# -----------------------------
# Run Analysis & Display
# -----------------------------

if soup:
    with st.spinner("Analyzing content..."):
        chunks = extract_chunks(soup)

        st.subheader("📌 Key Takeaways")
        top_chunks = get_key_takeaways(chunks)
        for i, item in enumerate(top_chunks, 1):
            st.markdown(f"**{i}. {item['text']}**")
            st.markdown(f"> 💡 **LLM Tip:** {item['llm_tip']}")

        st.subheader("⚠️ Flagged Chunks (Low Quality Score)")
        flagged = export_flagged_chunks(chunks)
        if not flagged.empty:
            st.dataframe(flagged)
        else:
            st.success("No low-quality content detected.")

        st.subheader("📘 Glossary Terms (Detected Definitions)")
        glossary = extract_glossary(chunks)
        if glossary:
            st.table(glossary)
        else:
            st.info("No glossary terms were identified.")
