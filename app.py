import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
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

nltk.download('punkt')
nltk.download('stopwords')

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

def extract_chunks(soup):
    for tag in EXCLUDED_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    seen = set()
    chunks = []
    for tag in soup.find_all(['section', 'article', 'div', 'p']):
        text = tag.get_text(strip=True)
        if len(text.split()) > 30 and text not in seen:
            seen.add(text)
            tokens = count_tokens(text)
            readability = analyze_readability(text)
            ambiguous_hits = [p for p in AMBIGUOUS_PHRASES if p in text.lower()]
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
                'ambiguous_phrases': ambiguous_hits,
                'quality_score': max(0, quality_score),
                'llm_tip': generate_llm_tip(text)
            })
    return chunks

def generate_llm_tip(text):
    tips = []
    if "FAQ" in text or "?" in text:
        tips.append("Consider wrapping this section with FAQ schema.")
    if any(word in text.lower() for word in ["how to", "guide", "steps"]):
        tips.append("This may qualify for HowTo schema.")
    if len(text.split()) > 150:
        tips.append("Break into smaller chunks to improve retrieval.")
    if "vs" in text.lower() or "comparison" in text.lower():
        tips.append("This could benefit from a Comparison schema.")
    return " ".join(tips)

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
    df = pd.DataFrame([c for c in chunks if c['quality_score'] < 70])
    return df
