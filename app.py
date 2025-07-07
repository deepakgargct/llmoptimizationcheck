# app.py
import streamlit as st
import requests, re, nltk, tiktoken, pandas as pd, matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from textstat import flesch_reading_ease
from nltk.corpus import stopwords
from nltk.data import find
import urllib.parse

# 1. Setup
def ensure_nltk_data():
    for pkg in ["punkt", "stopwords"]:
        try:
            find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)

ensure_nltk_data()
stop_words = set(stopwords.words("english"))
tokenizer = tiktoken.get_encoding("cl100k_base")

EXCLUDED_TAGS = ["header", "footer", "nav", "aside"]
AMBIGUOUS_PHRASES = {
    "click here": "...", "learn more": "...",  # trimmed for brevity
}

# Helper Functions
def count_tokens(text):
    return len(tokenizer.encode(text))

def readability_score(text):
    try: return flesch_reading_ease(text)
    except: return 0.0

def fetch_url_content(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def clean_soup(html):
    soup = html if isinstance(html, BeautifulSoup) else BeautifulSoup(html, "html.parser")
    for tag in EXCLUDED_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    return soup
def generate_llm_tip(text):
    tips = []
    t = text.lower()
    if "?" in text or "faq" in t: tips.append("Wrap with FAQ schema.")
    if any(k in t for k in ["how to", "guide", "step-by-step"]): tips.append("Use HowTo schema.")
    if any(k in t for k in ["vs", "compare"]): tips.append("Use Comparison schema.")
    if len(text.split()) > 150: tips.append("Split into smaller paragraphs.")
    if not tips: tips.append("Use clearer structure or schema markup.")
    return " ".join(tips)

def extract_chunks(soup):
    seen = set(); chunks = []
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
    pat = re.compile(r"(?P<term>[A-Z][A-Za-z0-9\- ]+?)\s+(is|refers to|means|can be defined as)\s+(?P<def>.+?)\.")
    gloss = []
    for c in chunks:
        for term, _, definition in pat.findall(c["text"]):
            gloss.append({"term": term.strip(), "definition": definition.strip()})
    return gloss

def extract_entities_dandelion(text):
    url = "https://api.dandelion.eu/datatxt/nex/v1/"
    params = {
        "text": text,
        "lang": "en",
        "token": st.secrets["dandelion"]["token"]
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return [{"Entity": a["spot"], "Label": a.get("label", "")} for a in data.get("annotations", [])]
    except Exception as e:
        return [{"Entity": "Error", "Label": str(e)}]

def extract_all_entities(chunks):
    all_ents = []
    seen = set()
    for c in chunks:
        ents = extract_entities_dandelion(c["text"])
        for ent in ents:
            key = (ent["Entity"], ent["Label"])
            if key not in seen:
                seen.add(key)
                all_ents.append(ent)
    return all_ents
# -------------------- STREAMLIT UI --------------------
st.set_page_config("GenAI Optimization & Visibility", layout="wide")
st.title("🧠 GenAI Optimization & AI Visibility Tool")
tabs = st.tabs([
    "🔍 Content Analyzer", 
    "📡 AI Search Visibility Checker",
    "📘 Schema Generator (coming soon)",
    "📊 Entity Coverage (coming soon)",
    "🧠 Keyword Clustering (coming soon)"
])

# -------------------- TAB 1: Content Analyzer --------------------
with tabs[0]:
    st.header("🔍 Content Analyzer")
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

    if soup:
        with st.spinner("Analyzing content…"):
            chunks = extract_chunks(soup)
            glossary = extract_glossary(chunks)
            entities = extract_all_entities(chunks)

        st.subheader("📌 Key Takeaways")
        for i, c in enumerate(sorted(chunks, key=lambda x: x["quality_score"], reverse=True)[:3], 1):
            st.markdown(f"**{i}. {c['text']}**")
            st.markdown(f"> 💡 **LLM Tip:** {c['llm_tip']}")

        st.subheader("⚠️ Flagged Chunks")
        flagged = [c for c in chunks if c["quality_score"] < 70]
        if flagged:
            st.dataframe(pd.DataFrame(flagged))
        else:
            st.success("No low-quality content.")

        st.subheader("📘 Glossary Terms")
        st.table(glossary) if glossary else st.info("No glossary terms found.")

        st.subheader("🗂 Named Entities (Dandelion)")
        st.table(entities) if entities else st.info("No named entities found.")

        st.subheader("📊 Quality Score Distribution")
        high = sum(1 for c in chunks if c["quality_score"] >= 85)
        med = sum(1 for c in chunks if 70 <= c["quality_score"] < 85)
        low = sum(1 for c in chunks if c["quality_score"] < 70)
        fig, ax = plt.subplots()
        ax.pie([high, med, low], labels=["High (≥85)", "Medium (70–84)", "Low (<70)"], autopct="%1.1f%%")
        st.pyplot(fig)

        st.subheader("🔍 Full Chunk Analysis")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"Chunk {i}: {c['text'][:60]}..."):
                st.markdown(f"**Text:** {c['text']}")
                st.markdown(f"**Token Count:** {c['token_count']}")
                st.markdown(f"**Readability (Flesch Score):** {c['readability']:.2f}")
                st.markdown(f"**Ambiguous Phrases:** {', '.join(c['ambiguous_phrases']) if c['ambiguous_phrases'] else 'None'}")
                st.markdown(f"**Quality Score:** {c['quality_score']} / 100")
                st.markdown(f"**LLM Tip:** {c['llm_tip']}")

# -------------------- TAB 2: AI Search Visibility Checker --------------------
with tabs[1]:
    st.header("📡 AI Search Visibility Checker")

    keyword_input = st.text_area("Enter keyword(s), one per line")
    domain_input = st.text_input("Enter your domain or brand (e.g., example.com)")

    def check_brave(keyword, domain):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            q = urllib.parse.quote_plus(keyword)
            r = requests.get(f"https://search.brave.com/search?q={q}", headers=headers)
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
            return {
                "Keyword": keyword, "Source": "Brave", "AI Overview": "No",
                "Domain Mentioned": "No", "Summary": "-", "Visibility Score": 0
            }
        except Exception as e:
            return {"Keyword": keyword, "Source": "Brave", "AI Overview": "Error", "Domain Mentioned": "-", "Summary": str(e), "Visibility Score": 0}

    def check_perplexity(keyword, domain):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            q = urllib.parse.quote_plus(keyword)
            r = requests.get(f"https://www.perplexity.ai/search?q={q}", headers=headers)
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
            return {
                "Keyword": keyword, "Source": "Perplexity", "AI Overview": "No",
                "Domain Mentioned": "No", "Summary": "-", "Visibility Score": 0
            }
        except Exception as e:
            return {"Keyword": keyword, "Source": "Perplexity", "AI Overview": "Error", "Domain Mentioned": "-", "Summary": str(e), "Visibility Score": 0}

    if st.button("Check Visibility") and keyword_input and domain_input:
        with st.spinner("Checking Brave and Perplexity…"):
            keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
            results = []
            for kw in keywords:
                results.append(check_brave(kw, domain_input))
                results.append(check_perplexity(kw, domain_input))

            df = pd.DataFrame(results)
            st.subheader("📊 Visibility Results")
            st.dataframe(df)

            avg_score = df.groupby("Keyword")["Visibility Score"].mean().reset_index()
            avg_score.columns = ["Keyword", "Avg Visibility Score"]
            st.subheader("🏆 Keyword Ranking")
            st.dataframe(avg_score.sort_values("Avg Visibility Score", ascending=False))

            st.download_button(
                "📥 Download Full Results", data=df.to_csv(index=False).encode(),
                file_name="ai_visibility_results.csv", mime="text/csv"
            )
            st.download_button(
                "📥 Download Ranking Only", data=avg_score.to_csv(index=False).encode(),
                file_name="ai_visibility_scores.csv", mime="text/csv"
            )

# -------------------- Placeholders for Future Tabs --------------------
with tabs[2]:
    st.info("Schema generation coming soon!")

with tabs[3]:
    st.info("Entity coverage scoring coming soon!")

with tabs[4]:
    st.info("Topic clustering coming soon!")
