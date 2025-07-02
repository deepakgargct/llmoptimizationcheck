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

# Setup
stop_words = set(stopwords.words('english'))
tokenizer = tiktoken.get_encoding("cl100k_base")
DANDELION_API_KEY = st.secrets["DANDELION_API_KEY"]

AMBIGUOUS_PHRASES = {
    "click here": "Be specific like 'visit our pricing page'",
    "learn more": "Be specific like 'discover SEO packages'",
    "read more": "Use clearer CTA like 'continue SEO guide'",
    "overview": "Use topic-specific phrase like 'SEO strategy breakdown'",
    "details": "Specify what kind: 'implementation details'",
    "more info": "Use targeted phrase like 'keyword research techniques'",
    "start now": "Clarify action e.g. 'start link audit'",
    "this article": "Replace with specific reference to article topic",
    "the following": "Summarize clearly what will follow",
    "some believe": "Avoid vague attribution. Specify who.",
    "it could be argued": "Be assertive or reference real opinions."
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
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def remove_duplicates(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        text = chunk['text'].strip()
        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)
    return unique_chunks

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
            ambiguous_hits = [k for k in AMBIGUOUS_PHRASES.keys() if k in text.lower()]
            suggestions = [f"⚠️ Ambiguous phrase: '{k}' → {AMBIGUOUS_PHRASES[k]}" for k in ambiguous_hits]
            quality_score = 100
            if tokens > 300:
                quality_score -= 15
            if readability < 60:
                quality_score -= 15
            if ambiguous_hits:
                quality_score -= 10 * len(ambiguous_hits)
            chunks.append({
                'text': text[:300] + '...' if len(text) > 300 else text,
                'token_count': tokens,
                'readability': readability,
                'ambiguous_phrases': suggestions,
                'quality_score': max(0, quality_score)
            })
    return remove_duplicates(chunks)

def extract_glossary(chunks):
    glossary = []
    pattern = re.compile(r"(?P<term>[A-Z][a-zA-Z0-9\\- ]+?)\\s+(is|refers to|means|can be defined as)\\s+(?P<definition>.+?)\\.")
    for chunk in chunks:
        matches = pattern.findall(chunk['text'])
        for match in matches:
            glossary.append({'term': match[0].strip(), 'definition': match[2].strip()})
    return glossary

def get_key_takeaways(chunks, top_n=3):
    sorted_chunks = sorted(chunks, key=lambda x: x['quality_score'], reverse=True)
    return sorted_chunks[:top_n]

def export_flagged_chunks(chunks):
    df = pd.DataFrame([c for c in chunks if c['quality_score'] < 70])
    return df

def extract_entities_dandelion(text):
    url = "https://api.dandelion.eu/datatxt/nex/v1/"
    payload = {
        'text': text,
        'lang': 'en',
        'token': DANDELION_API_KEY
    }
    try:
        res = requests.post(url, data=payload)
        data = res.json()
        if "annotations" in data:
            return list(set(a['spot'] for a in data['annotations']))
        return []
    except:
        return []

# Streamlit UI
st.set_page_config(page_title="GenAI Optimization Checker", layout="wide")
st.title("GenAI Optimization Checker")

# Sidebar Input
with st.sidebar:
    option = st.radio("Select input method:", ["URL", "HTML file", "Raw HTML", ".txt file"])
    html_content = ""

    if option == "URL":
        url = st.text_input("Enter URL to analyze:")
        if st.button("Fetch URL"):
            soup = fetch_url_content(url)
            html_content = soup.prettify() if soup else ""
    elif option == "HTML file":
        uploaded_file = st.file_uploader("Upload HTML file", type=["html"])
        if uploaded_file:
            html_content = uploaded_file.read().decode("utf-8")
    elif option == "Raw HTML":
        html_content = st.text_area("Paste raw HTML code:")
    elif option == ".txt file":
        uploaded_txt = st.file_uploader("Upload .txt file", type="txt")
        if uploaded_txt:
            html_content = uploaded_txt.read().decode("utf-8")

if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = extract_chunks(soup)

    if not chunks:
        st.warning("No meaningful content chunks found.")
    else:
        tabs = st.tabs(["🔍 Analysis", "📘 Glossary", "📌 Takeaways", "🧠 AI Optimization", "📊 Visualizations", "📤 Export"])

        with tabs[0]:
            st.subheader("Chunk Quality & Phrasing")
            for i, chunk in enumerate(chunks):
                st.markdown(f"### Chunk {i+1}")
                st.markdown(f"**Tokens:** {chunk['token_count']} | **Readability:** {chunk['readability']:.2f} | **Score:** {chunk['quality_score']}%")
                st.text(chunk['text'])
                for fix in chunk['ambiguous_phrases']:
                    st.warning(fix)

        with tabs[1]:
            st.subheader("📘 Glossary")
            glossary = extract_glossary(chunks)
            if glossary:
                st.dataframe(pd.DataFrame(glossary))
            else:
                st.info("No glossary terms extracted.")

        with tabs[2]:
            st.subheader("📌 Key Takeaways")
            for takeaway in get_key_takeaways(chunks):
                st.success(takeaway['text'])

        with tabs[3]:
            st.subheader("🧠 AI Optimization Tips")
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Section {i+1}**")
                ai_tips = []
                if chunk['token_count'] > 300:
                    ai_tips.append("🔹 Split this chunk: Too long for optimal LLM retrieval.")
                if chunk['readability'] < 50:
                    ai_tips.append("🔹 Improve sentence simplicity for clarity.")
                if chunk['ambiguous_phrases']:
                    ai_tips.append("🔹 Replace vague phrasing for clearer intent.")
                if not ai_tips:
                    ai_tips.append("✅ Well-optimized section.")
                for tip in ai_tips:
                    st.info(tip)

                # Entity extraction
                entities = extract_entities_dandelion(chunk['text'])
                if entities:
                    st.markdown("**Entities:** " + ", ".join(entities))
                else:
                    st.markdown("*No entities extracted.*")

        with tabs[4]:
            st.subheader("📊 Visualizations")
            df = pd.DataFrame(chunks)
            fig1, ax1 = plt.subplots()
            sns.histplot(df['quality_score'], bins=10, kde=True, ax=ax1)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.scatterplot(x='token_count', y='readability', data=df, ax=ax2)
            st.pyplot(fig2)

        with tabs[5]:
            st.subheader("📤 Export Flagged Content")
            flagged_df = export_flagged_chunks(chunks)
            if not flagged_df.empty:
                st.dataframe(flagged_df)
                csv = flagged_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "flagged_chunks.csv", "text/csv")
            else:
                st.success("No flagged content found.")

st.caption("🔍 Built to help improve LLM visibility and clarity. Entity-rich, jargon-free, chunk-optimized content for GenAI.")
'''

# Save to file
file_path = "/mnt/data/genai_optimizer_app.py"
with open(file_path, "w") as f:
    f.write(code)

file_path
