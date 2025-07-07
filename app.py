# app.py - Modular GenAI Optimization App

import streamlit as st
import requests, re, nltk, pandas as pd, matplotlib.pyplot as plt, subprocess
from bs4 import BeautifulSoup
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.data import find
import tiktoken
import spacy

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
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm", "--user"])
        return spacy.load("en_core_web_sm")

ensure_nltk_data()
nlp = ensure_spacy_model()
stop_words = set(stopwords.words("english"))
tokenizer = tiktoken.get_encoding("cl100k_base")

EXCLUDED_TAGS = ["header", "footer", "nav", "aside"]

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
def fetch_url_content(url):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def clean_soup(html):
    soup = html if isinstance(html, BeautifulSoup) else BeautifulSoup(html, "html.parser")
    for tag in EXCLUDED_TAGS:
        for el in soup.find_all(tag): el.decompose()
    return soup

def count_tokens(text):
    return len(tokenizer.encode(text))

def readability_score(text):
    try:
        return flesch_reading_ease(text)
    except:
        return 0.0

def generate_llm_tip(text):
    tips = []
    t = text.lower()
    if "?" in t or "faq" in t: tips.append("Wrap with FAQ schema.")
    if any(k in t for k in ["how to", "step-by-step", "guide", "steps"]): tips.append("Use HowTo schema.")
    if any(k in t for k in [" vs ", "compare", "comparison"]): tips.append("Use Comparison schema.")
    if len(text.split()) > 150: tips.append("Split into smaller paragraphs.")
    if not tips: tips.append("Use clearer structure or schema markup.")
    return " ".join(tips)

def extract_chunks(soup):
    seen = set()
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        raw = tag.get_text(" ", strip=True)
        if len(raw.split()) < 30: continue
        txt = re.sub(r"\s+", " ", raw.strip())
        if txt in seen: continue
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
    pat = re.compile(r"(?P<term>[A-Z][A-Za-z0-9\- ]+?)\\s+(is|refers to|means|can be defined as)\\s+(?P<def>.+?)\\.")
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
    labels = ["High (≥85)", "Medium (70–84)", "Low (<70)"]
    sizes = [high, med, low]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

def color_for_score(score):
    if score >= 85: return "✅"
    elif score >= 70: return "🟡"
    else: return "🔴"
st.set_page_config("LLM Optimizer", layout="wide")
st.title("🧠 GenAI Optimization App")

tabs = st.tabs(["🔍 Content Analyzer", "📡 AI Search Visibility", "🧩 Optional Modules"])
with tabs[0]:
    st.header("🔍 Content Analyzer")
    mode = st.selectbox("Input Method", ["Webpage URL", "Upload .txt", "Upload .html", "Paste HTML", "Direct Text"])

    soup = None
    if mode == "Webpage URL":
        url = st.text_input("Enter URL")
        if st.button("Analyze") and url:
            soup = fetch_url_content(url)
    elif mode == "Upload .txt":
        f = st.file_uploader("Upload Text File", type="txt")
        if f: soup = clean_soup(f"<p>{f.read().decode()}</p>")
    elif mode == "Upload .html":
        f = st.file_uploader("Upload HTML File", type=["html", "htm"])
        if f: soup = clean_soup(f.read().decode())
    elif mode == "Paste HTML":
        html = st.text_area("Paste HTML Code")
        if html: soup = clean_soup(html)
    elif mode == "Direct Text":
        txt = st.text_area("Enter plain text")
        if txt: soup = clean_soup(f"<p>{txt}</p>")

    if soup:
        with st.spinner("Processing..."):
            chunks = extract_chunks(soup)
            glossary = extract_glossary(chunks)
            entities = extract_entities(chunks)

        st.subheader("📌 Key Takeaways")
        for i, c in enumerate(sorted(chunks, key=lambda x: x["quality_score"], reverse=True)[:3], 1):
            st.markdown(f"**{i}. {c['text']}**\n\n> 💡 **LLM Tip:** {c['llm_tip']}")

        st.subheader("⚠️ Flagged Chunks")
        flagged = [c for c in chunks if c["quality_score"] < 70]
        if flagged:
            st.dataframe(pd.DataFrame(flagged))
        else:
            st.success("No low-quality content detected.")

        st.subheader("📘 Glossary")
        if glossary:
            st.dataframe(pd.DataFrame(glossary))
        else:
            st.info("No glossary terms found.")

        st.subheader("🗂 Named Entities")
        if entities:
            st.dataframe(pd.DataFrame(entities))
        else:
            st.info("No named entities found.")

        st.subheader("📊 Content Quality Overview")
        show_quality_pie_chart(chunks)

        st.subheader("🔎 Full Chunk Breakdown")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"Chunk {i}: {c['text'][:60]}..."):
                st.markdown(f"**Token Count:** {c['token_count']}  \n"
                            f"**Readability Score:** {c['readability']:.2f}  \n"
                            f"**Ambiguous Phrases:** {', '.join(c['ambiguous_phrases']) or 'None'}  \n"
                            f"**Quality Score:** {color_for_score(c['quality_score'])} {c['quality_score']}  \n"
                            f"**LLM Tip:** {c['llm_tip']}")
def check_brave_summary(keyword, domain):
    try:
        url = f"https://search.brave.com/search?q={keyword.replace(' ', '+')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        summary = soup.find("div", class_="snippet-summary")
        if summary:
            text = summary.get_text(" ", strip=True)
            mentioned = domain.lower() in text.lower()
            return {
                "Keyword": keyword,
                "Source": "Brave",
                "AI Overview": "Yes",
                "Domain Mentioned": "Yes" if mentioned else "No",
                "Summary": text,
                "Visibility Score": 50 + (50 if mentioned else 0)
            }
    except: pass
    return {
        "Keyword": keyword,
        "Source": "Brave",
        "AI Overview": "No", "Domain Mentioned": "No", "Summary": "-", "Visibility Score": 0
    }

def check_perplexity_summary(keyword, domain):
    try:
        url = f"https://www.perplexity.ai/search?q={keyword.replace(' ', '%20')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        answers = soup.find_all("div", class_=re.compile("answer"))
        for a in answers:
            text = a.get_text(" ", strip=True)
            if keyword.lower() in text.lower():
                mentioned = domain.lower() in text.lower()
                return {
                    "Keyword": keyword,
                    "Source": "Perplexity",
                    "AI Overview": "Yes",
                    "Domain Mentioned": "Yes" if mentioned else "No",
                    "Summary": text[:300] + "…" if len(text) > 300 else text,
                    "Visibility Score": 50 + (50 if mentioned else 0)
                }
    except: pass
    return {
        "Keyword": keyword,
        "Source": "Perplexity",
        "AI Overview": "No", "Domain Mentioned": "No", "Summary": "-", "Visibility Score": 0
    }

with tabs[1]:
    st.header("📡 AI Search Visibility Checker")
    keywords = st.text_area("Enter keywords (one per line)")
    domain = st.text_input("Enter domain (e.g., otolawn.com)")

    if st.button("Check Visibility") and keywords and domain:
        kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]
        results = []
        for kw in kw_list:
            results.append(check_brave_summary(kw, domain))
            results.append(check_perplexity_summary(kw, domain))

        df = pd.DataFrame(results)
        st.dataframe(df)

        rankings = df.groupby("Keyword")["Visibility Score"].mean().reset_index()
        rankings.columns = ["Keyword", "Avg Visibility Score"]
        st.subheader("📈 Keyword Ranking by AI Visibility")
        st.dataframe(rankings.sort_values("Avg Visibility Score", ascending=False))

        st.download_button("Download Results", df.to_csv(index=False).encode(), "ai_visibility.csv", "text/csv")
with tabs[2]:
    st.header("🧩 Optional Modules (Advanced)")

    schema = st.checkbox("Suggest Schema Types")
    score_entities = st.checkbox("Score Entity Coverage")
    cluster_keywords = st.checkbox("Keyword Clustering")

    if schema:
        st.info("💡 Coming soon: Auto-suggest schema based on content.")

    if score_entities:
        st.info("🧠 Planned: Entity coverage scoring with comparison to competitors or knowledge base.")

    if cluster_keywords:
        st.info("📚 Feature coming: Cluster keywords into content groups using embeddings.")
