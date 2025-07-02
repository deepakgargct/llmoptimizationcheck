# app.py
import streamlit as st
import requests, re, nltk, tiktoken, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.data import find
import spacy, spacy.cli

# -------------------- 1. SETUP --------------------
def ensure_nltk_data():
    for pkg in ["punkt", "stopwords"]:
        try:
            find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

def ensure_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

ensure_nltk_data()
nlp = ensure_spacy_model()
stop_words = set(stopwords.words("english"))
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

EXCLUDED_TAGS = ["header", "footer", "nav", "aside"]

# -------------------- 2. FUNCTIONS --------------------
def count_tokens(text):
    return len(tokenizer.encode(text))

def readability_score(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0.0

def fetch_url_content(url):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def clean_soup(html):
    soup = (
        html if isinstance(html, BeautifulSoup)
        else BeautifulSoup(html, "html.parser")
    )
    for tag in EXCLUDED_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    return soup

def generate_llm_tip(text):
    tips = []
    t = text.lower()
    if "?" in text or "faq" in t:
        tips.append("Wrap with FAQ schema.")
    if any(k in t for k in ["how to", "step-by-step", "guide", "steps"]):
        tips.append("Use HowTo schema.")
    if any(k in t for k in ["vs", " compare", "comparison"]):
        tips.append("Use Comparison schema.")
    if len(text.split()) > 150:
        tips.append("Split into smaller paragraphs.")
    if not tips:
        tips.append("Use clearer structure or schema markup.")
    return " ".join(tips)

def extract_chunks(soup):
    seen = set()
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        raw = tag.get_text(" ", strip=True)
        if len(raw.split()) < 30:
            continue
        txt = re.sub(r"\s+", " ", raw.strip())
        if txt in seen:
            continue
        seen.add(txt)
        tokens = count_tokens(txt)
        read_score = readability_score(txt)
        ambiguous = [p for p in AMBIGUOUS_PHRASES if p in txt.lower()]
        quality = 100 - (15 if tokens > 300 else 0) - (15 if read_score < 60 else 0) - 10 * len(ambiguous)
        chunks.append({
            "text": txt[:300] + "…" if len(txt) > 300 else txt,
            "token_count": tokens,
            "readability": read_score,
            "ambiguous_phrases": ambiguous,
            "quality_score": max(0, quality),
            "llm_tip": generate_llm_tip(txt)
        })
    return chunks

def extract_glossary(chunks):
    pat = re.compile(r"(?P<term>[A-Z][A-Za-z0-9\- ]+?)\s+(is|refers to|means|can be defined as)\s+(?P<def>.+?)\.")
    gloss = []
    for c in chunks:
        for term, _, definition in pat.findall(c["text"]):
            gloss.append({"term": term.strip(), "definition": definition.strip()})
    return gloss

def extract_entities(chunks):
    ents = []
    for c in chunks:
        doc = nlp(c["text"])
        for ent in doc.ents:
            ents.append((ent.text, ent.label_))
    seen = set(); unique = []
    for e, l in ents:
        key = (e, l)
        if key not in seen:
            seen.add(key)
            unique.append({"Entity": e, "Label": l})
    return unique

def show_quality_pie_chart(chunks):
    high = sum(1 for c in chunks if c["quality_score"] >= 85)
    med = sum(1 for c in chunks if 70 <= c["quality_score"] < 85)
    low = sum(1 for c in chunks if c["quality_score"] < 70)
    labels, sizes = ["High (≥85)", "Medium (70–84)", "Low (<70)"], [high, med, low]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

def color_for_score(score):
    if score >= 85: return "✅"
    elif score >= 70: return "🟡"
    else: return "🔴"

# -------------------- 3. STREAMLIT APP --------------------
st.set_page_config("LLM Optimizer", layout="wide")
st.title("🧠 LLM Optimization & Content Analyzer")

mode = st.selectbox("Choose Input Method", [
    "Webpage URL", "Upload .txt File", "Upload .html File",
    "Paste HTML Code", "Direct Text Input"
])

soup = None

if mode == "Webpage URL":
    url = st.text_input("Enter URL:")
    if st.button("Analyze") and url:
        soup = fetch_url_content(url)

elif mode == "Upload .txt File":
    f = st.file_uploader("Upload .txt", type=["txt"])
    if f: soup = clean_soup(f"<p>{f.read().decode('utf-8')}</p>")

elif mode == "Upload .html File":
    f = st.file_uploader("Upload .html", type=["html", "htm"])
    if f: soup = clean_soup(f.read().decode("utf-8"))

elif mode == "Paste HTML Code":
    code = st.text_area("Paste full HTML:")
    if code: soup = clean_soup(code)

elif mode == "Direct Text Input":
    txt = st.text_area("Paste plain text:")
    if txt: soup = clean_soup(f"<p>{txt}</p>")

# -------------------- 4. OUTPUT --------------------
if soup:
    with st.spinner("Analyzing content…"):
        chunks = extract_chunks(soup)
        glossary = extract_glossary(chunks)
        entities = extract_entities(chunks)

    st.subheader("📌 Key Takeaways")
    for i, c in enumerate(sorted(chunks, key=lambda x: x["quality_score"], reverse=True)[:3], 1):
        st.markdown(f"**{i}. {c['text']}**")
        st.markdown(f"> 💡 **LLM Tip:** {c['llm_tip']}")

    st.subheader("⚠️ Flagged Chunks")
    flagged = [c for c in chunks if c["quality_score"] < 70]
    st.dataframe(pd.DataFrame(flagged)) if flagged else st.success("No low-quality content.")

    st.subheader("📘 Glossary Terms")
    st.table(glossary) if glossary else st.info("No glossary terms found.")

    st.subheader("🗂 Named Entities")
    st.table(entities) if entities else st.info("No named entities found.")

    st.subheader("📊 Quality Score Distribution")
    show_quality_pie_chart(chunks)

    st.subheader("🔍 Full Chunk Analysis")
    for i, c in enumerate(chunks, 1):
        with st.expander(f"Chunk {i}: {c['text'][:60]}..."):
            st.markdown(f"**Text:** {c['text']}")
            st.markdown(f"**Token Count:** {c['token_count']}")
            st.markdown(f"**Readability (Flesch Score):** {c['readability']:.2f}")
            st.markdown(f"**Ambiguous Phrases:** {', '.join(c['ambiguous_phrases']) if c['ambiguous_phrases'] else 'None'}")
            st.markdown(f"**Quality Score:** {color_for_score(c['quality_score'])} {c['quality_score']} / 100")
            st.markdown(f"**LLM Tip:** {c['llm_tip']}")
