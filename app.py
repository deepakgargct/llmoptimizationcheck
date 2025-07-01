import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import tiktoken
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import textstat
import wikipedia
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Ensure punkt_tab is downloaded as well

# Constants
DANDELION_TOKEN = st.secrets.get("DANDELION_API_KEY", "")
tokenizer = tiktoken.get_encoding("cl100k_base")
stop_words = set(stopwords.words("english"))
EXCLUDED_TAGS = ['header', 'footer', 'nav', 'aside', 'script', 'style']

# Enhanced context-aware ambiguous phrases
CONTEXT_AWARE_PHRASES = {
    "seo": {
        "overview": "SEO Strategy Summary",
        "learn more": "Explore Keyword Optimization Techniques",
        "click here": "Start Your SEO Audit",
        "read more": "Read Advanced SEO Tactics",
        "details": "SEO Implementation Checklist"
    },
    "marketing": {
        "overview": "Marketing Campaign Overview",
        "learn more": "Discover Marketing Strategies",
        "click here": "Launch Your Campaign",
        "read more": "Explore Marketing Case Studies",
        "details": "Marketing ROI Breakdown"
    },
    "technology": {
        "overview": "Technical Architecture Overview",
        "learn more": "Explore Technical Documentation",
        "click here": "Access Developer Tools",
        "read more": "Read Implementation Guide",
        "details": "Technical Specifications"
    },
    "default": {
        "overview": "Comprehensive Summary",
        "learn more": "Discover More Information",
        "click here": "Take Action Now",
        "read more": "Explore Full Content",
        "details": "Complete Information"
    }
}

# Expected entities for different topics
TOPIC_ENTITIES = {
    "seo": ["google", "keywords", "rankings", "backlinks", "content", "optimization", "search engine", "serp"],
    "marketing": ["audience", "campaign", "conversion", "roi", "brand", "customer", "engagement", "analytics"],
    "technology": ["api", "database", "server", "application", "software", "development", "programming", "system"],
    "business": ["revenue", "profit", "strategy", "market", "competition", "growth", "customer", "sales"]
}

def count_tokens(text):
    """Count tokens using tiktoken"""
    return len(tokenizer.encode(text))

def analyze_readability(text):
    """Enhanced readability analysis"""
    try:
        flesch = textstat.flesch_reading_ease(text)
        flesch_kincaid = textstat.flesch_kincaid_grade(text)
        return {
            'flesch_score': flesch,
            'grade_level': flesch_kincaid,
            'reading_time': len(text.split()) / 200  # avg reading speed 200 wpm
        }
    except:
        return {'flesch_score': 0, 'grade_level': 0, 'reading_time': 0}

def calculate_clarity_score(text):
    """Calculate clarity score based on sentence structure"""
    sentences = sent_tokenize(text)  # Ensure punkt is downloaded and available
    if not sentences:
        return 0
    
    avg_sentence_length = np.mean([len(word_tokenize(s)) for s in sentences])
    complex_words = len([w for w in word_tokenize(text.lower()) if len(w) > 6])
    total_words = len(word_tokenize(text))
    
    # Ideal sentence length is 15-20 words
    length_penalty = abs(avg_sentence_length - 17.5) / 17.5
    complexity_ratio = complex_words / total_words if total_words > 0 else 0
    
    clarity_score = max(0, 100 - (length_penalty * 30) - (complexity_ratio * 40))
    return round(clarity_score, 2)

def detect_content_type(text):
    """Detect content type for schema suggestions"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["what is", "how to", "why", "when", "where"]):
        return "FAQ"
    elif any(word in text_lower for word in ["review", "rating", "stars", "testimonial"]):
        return "Review"
    elif any(word in text_lower for word in ["product", "price", "$", "buy", "purchase"]):
        return "Product"
    elif any(word in text_lower for word in ["event", "date", "time", "location", "venue"]):
        return "Event"
    elif any(word in text_lower for word in ["recipe", "ingredients", "cooking", "bake"]):
        return "Recipe"
    elif any(word in text_lower for word in ["job", "career", "position", "employment"]):
        return "JobPosting"
    else:
        return "Article"

def calculate_schema_suitability_score(text, content_type):
    """Calculate how well content fits suggested schema"""
    text_lower = text.lower()
    
    schema_patterns = {
        "FAQ": ["question", "answer", "what", "how", "why", "when", "where"],
        "Review": ["review", "rating", "stars", "recommend", "experience", "quality"],
        "Product": ["product", "price", "features", "specifications", "buy", "purchase"],
        "Event": ["event", "date", "time", "location", "venue", "tickets"],
        "Recipe": ["ingredients", "instructions", "cook", "bake", "serve", "prep"],
        "JobPosting": ["job", "requirements", "qualifications", "salary", "benefits"],
        "Article": ["introduction", "conclusion", "section", "paragraph", "topic"]
    }
    
    patterns = schema_patterns.get(content_type, [])
    matches = sum(1 for pattern in patterns if pattern in text_lower)
    score = (matches / len(patterns)) * 100 if patterns else 50
    
    return min(100, score)

def fetch_url_content(url):
    """Fetch and parse URL content"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return None
            
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

def get_wikipedia_definition(entity, max_chars=200):
    """Get short definition from Wikipedia"""
    try:
        summary = wikipedia.summary(entity, sentences=2, auto_suggest=True)
        return summary[:max_chars] + "..." if len(summary) > max_chars else summary
    except:
        return f"Entity: {entity}"

def extract_entities_dandelion(text):
    """Extract entities using Dandelion API with enhanced processing"""
    if not DANDELION_TOKEN:
        # Fallback: simple named entity extraction
        entities = []
        words = word_tokenize(text)
        tagged = nltk.pos_tag(words)
        
        for word, pos in tagged:
            if pos in ['NNP', 'NNPS'] and len(word) > 2:  # Proper nouns
                entities.append(word.lower())
        
        return list(set(entities))
    
    try:
        url = "https://api.dandelion.eu/datatxt/nex/v1/"
        payload = {
            'text': text[:5000],  # Limit text length
            'lang': 'en',
            'token': DANDELION_TOKEN,
            'confidence': 0.6
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        entities = []
        for annotation in data.get("annotations", []):
            entity = {
                'text': annotation.get('spot', ''),
                'confidence': annotation.get('confidence', 0),
                'types': annotation.get('types', []),
                'definition': get_wikipedia_definition(annotation.get('spot', ''))
            }
            entities.append(entity)
            
        return entities
    except Exception as e:
        st.warning(f"Dandelion API error: {str(e)}")
        return []

def detect_topic_context(text):
    """Detect main topic context for context-aware suggestions"""
    text_lower = text.lower()
    
    topic_keywords = {
        "seo": ["seo", "search", "google", "keywords", "ranking", "optimization"],
        "marketing": ["marketing", "campaign", "audience", "conversion", "brand"],
        "technology": ["technology", "software", "api", "development", "programming"],
        "business": ["business", "revenue", "profit", "strategy", "market"]
    }
    
    topic_scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        topic_scores[topic] = score
    
    return max(topic_scores, key=topic_scores.get) if topic_scores else "default"

def get_context_aware_suggestions(text, ambiguous_phrases):
    """Generate context-aware suggestions for ambiguous phrases"""
    topic = detect_topic_context(text)
    context_phrases = CONTEXT_AWARE_PHRASES.get(topic, CONTEXT_AWARE_PHRASES["default"])
    
    suggestions = []
    for phrase in ambiguous_phrases:
        if phrase in context_phrases:
            suggestions.append(f"⚠️ Replace '{phrase}' → '{context_phrases[phrase]}'")
        else:
            suggestions.append(f"⚠️ Make '{phrase}' more specific to your content")
    
    return suggestions

def calculate_topic_coverage_score(text, entities):
    """Calculate topic coverage score"""
    topic = detect_topic_context(text)
    expected_entities = TOPIC_ENTITIES.get(topic, [])
    
    if not expected_entities:
        return 75  # Default score if no expected entities
    
    text_lower = text.lower()
    found_entities = sum(1 for entity in expected_entities if entity in text_lower)
    coverage_score = (found_entities / len(expected_entities)) * 100
    
    missing_entities = [e for e in expected_entities if e not in text_lower]
    
    return {
        'score': round(coverage_score, 2),
        'found': found_entities,
        'expected': len(expected_entities),
        'missing': missing_entities
    }

def detect_misaligned_titles(text):
    """Detect generic or non-descriptive headings"""
    generic_headings = [
        "overview", "introduction", "details", "more info", "information",
        "content", "section", "part", "chapter", "summary", "conclusion"
    ]
    
    # Simple heading detection (look for short lines that might be headings)
    lines = text.split('\n')
    potential_headings = [line.strip() for line in lines if len(line.strip().split()) <= 5 and len(line.strip()) > 3]
    
    misaligned = []
    for heading in potential_headings:
        if any(generic in heading.lower() for generic in generic_headings):
            topic = detect_topic_context(text)
            suggestion = f"Make '{heading}' more specific (e.g., '{topic.title()} {heading}')"
            misaligned.append({'heading': heading, 'suggestion': suggestion})
    
    return misaligned

def generate_llm_optimization_tips(chunk_data):
    """Generate comprehensive LLM optimization tips"""
    tips = []
    
    # Readability tips
    if chunk_data['readability']['flesch_score'] < 60:
        tips.append("💡 Simplify language: Use shorter sentences and common words")
    
    # Token count tips
    if chunk_data['token_count'] > 300:
        tips.append("💡 Split content: Break into focused sections (~200 tokens each)")
    
    # Clarity tips
    if chunk_data['clarity_score'] < 70:
        tips.append("💡 Improve clarity: Use active voice and direct statements")
    
    # Schema tips
    schema_type = chunk_data['content_type']
    if schema_type == "FAQ":
        tips.append("💡 FAQ optimization: Use question-answer format with clear headings")
    elif schema_type == "Product":
        tips.append("💡 Product optimization: Include price, features, and reviews")
    elif schema_type == "Article":
        tips.append("💡 Article optimization: Add clear intro, body, and conclusion")
    
    # Entity tips
    if chunk_data['topic_coverage']['score'] < 60:
        missing = chunk_data['topic_coverage']['missing'][:3]
        tips.append(f"💡 Add relevant topics: Consider including {', '.join(missing)}")
    
    return tips

def remove_duplicates(chunks):
    """Remove duplicate chunks based on text similarity"""
    seen = set()
    filtered = []
    
    for chunk in chunks:
        # Create a simplified version for comparison
        simplified = re.sub(r'\s+', ' ', chunk['text'].lower().strip())
        if simplified not in seen and len(simplified) > 50:
            filtered.append(chunk)
            seen.add(simplified)
    
    return filtered

def extract_enhanced_chunks(soup):
    """Extract and analyze content chunks with enhanced features"""
    # Remove unwanted elements
    for tag in EXCLUDED_TAGS:
        for elem in soup.find_all(tag):
            elem.decompose()
    
    chunks = []
    
    # Extract content from various elements
    content_selectors = [
        'article', 'section', 'div.content', 'div.post', 'div.article',
        'main', 'div[class*="content"]', 'p'
    ]
    
    processed_texts = set()
    
    for selector in content_selectors:
        elements = soup.select(selector) if '.' in selector or '[' in selector else soup.find_all(selector)
        
        for element in elements:
            text = element.get_text(strip=True)
            
            # Skip short content or duplicates
            if len(text.split()) < 30 or text in processed_texts:
                continue
                
            processed_texts.add(text)
            
            # Basic analysis
            token_count = count_tokens(text)
            readability = analyze_readability(text)
            clarity_score = calculate_clarity_score(text)
            content_type = detect_content_type(text)
            schema_score = calculate_schema_suitability_score(text, content_type)
            
            # Entity analysis
            entities = extract_entities_dandelion(text)
            topic_coverage = calculate_topic_coverage_score(text, entities)
            
            # Ambiguous phrase detection
            ambiguous_phrases = []
            for phrase in ["overview", "learn more", "click here", "read more", "details", "more info"]:
                if phrase in text.lower():
                    ambiguous_phrases.append(phrase)
            
            suggestions = get_context_aware_suggestions(text, ambiguous_phrases)
            
            # Misaligned titles
            misaligned_titles = detect_misaligned_titles(text)
            
            # Calculate overall quality score
            quality_score = np.mean([
                min(100, max(0, 100 - (token_count - 200) / 10)),  # Token score
                readability['flesch_score'] if readability['flesch_score'] > 0 else 60,  # Readability
                clarity_score,  # Clarity
                schema_score,  # Schema suitability
                topic_coverage['score']  # Topic coverage
            ])
            
            chunk_data = {
                'text': text[:500] + '...' if len(text) > 500 else text,
                'full_text': text,
                'token_count': token_count,
                'readability': readability,
                'clarity_score': clarity_score,
                'content_type': content_type,
                'schema_score': schema_score,
                'entities': entities,
                'topic_coverage': topic_coverage,
                'ambiguous_phrases': ambiguous_phrases,
                'suggestions': suggestions,
                'misaligned_titles': misaligned_titles,
                'quality_score': round(quality_score, 2)
            }
            
            # Generate LLM tips
            chunk_data['llm_tips'] = generate_llm_optimization_tips(chunk_data)
            
            chunks.append(chunk_data)
    
    return remove_duplicates(chunks)

# Streamlit UI
st.set_page_config(
    page_title="Enhanced GenAI Optimization Checker",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Enhanced GenAI Optimization Checker")
st.markdown("*Advanced content analysis for AI search engines and LLM readiness*")

# Sidebar for input
with st.sidebar:
    st.markdown("### 📝 Content Input")
    method = st.radio("Choose input method:", ["URL", "HTML file", "Raw HTML", "Text file"])
    
    html_content = ""
    
    if method == "URL":
        url = st.text_input("Enter URL:")
        if st.button("🔍 Analyze URL"):
            with st.spinner("Fetching content..."):
                soup = fetch_url_content(url)
                if soup:
                    html_content = str(soup)
                    st.success("Content fetched successfully!")
                else:
                    st.error("Failed to fetch content")
                    
    elif method == "HTML file":
        uploaded_file = st.file_uploader("Upload HTML file", type=['html', 'htm'])
        if uploaded_file:
            html_content = uploaded_file.read().decode('utf-8')
            
    elif method == "Raw HTML":
        html_content = st.text_area("Paste HTML content:", height=200)
        
    elif method == "Text file":
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        if uploaded_file:
            text_content = uploaded_file.read().decode('utf-8')
            html_content = f"<div>{text_content}</div>"

# Main analysis
if html_content:
    with st.spinner("Analyzing content..."):
        soup = BeautifulSoup(html_content, 'html.parser')
        chunks = extract_enhanced_chunks(soup)
        
        if not chunks:
            st.warning("No content chunks found. Please check your input.")
        else:
            # Create topic clusters
            chunks, cluster_data = create_topic_clusters(chunks)
            
            # Create tabs for different views
            tabs = st.tabs([
                "🧠 AI Analysis", 
                "📊 Multi-Dimension Scoring", 
                "🎯 Entity Coverage", 
                "📚 Glossary & Definitions",
                "🗺️ Topic Clustering",
                "📈 Visualizations", 
                "📤 Export Results"
            ])
            
            # AI Analysis Tab
            with tabs[0]:
                st.subheader("🧠 Comprehensive AI Analysis")
                
                for i, chunk in enumerate(chunks):
                    with st.expander(f"📄 Chunk {i+1} - Quality Score: {chunk['quality_score']}%"):
                        # Content preview
                        st.markdown("**Content Preview:**")
                        st.markdown(chunk['text'])
                        
                        # Basic metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tokens", chunk['token_count'])
                        with col2:
                            st.metric("Readability", f"{chunk['readability']['flesch_score']:.1f}")
                        with col3:
                            st.metric("Reading Time", f"{chunk['readability']['reading_time']:.1f}m")
                        
                        # Suggestions
                        if chunk['suggestions']:
                            st.markdown("**⚠️ Ambiguous Phrase Suggestions:**")
                            for suggestion in chunk['suggestions']:
                                st.warning(suggestion)
                        
                        # Misaligned titles
                        if chunk['misaligned_titles']:
                            st.markdown("**🧩 Title Alignment Issues:**")
                            for title_issue in chunk['misaligned_titles']:
                                st.error(f"Generic heading: '{title_issue['heading']}' - {title_issue['suggestion']}")
                        
                        # LLM Tips
                        if chunk['llm_tips']:
                            st.markdown("**🧠 LLM Optimization Tips:**")
                            for tip in chunk['llm_tips']:
                                st.info(tip)
                        
                        # Schema suggestion
                        st.markdown(f"**📘 Suggested Schema:** `{chunk['content_type']}` (Suitability: {chunk['schema_score']:.1f}%)")
            
            # Multi-Dimension Scoring Tab
            with tabs[1]:
                st.subheader("📊 Multi-Dimensional Scoring Analysis")
                
                # Create scoring dataframe
                scoring_data = []
                for i, chunk in enumerate(chunks):
                    scoring_data.append({
                        'Chunk': f"Chunk {i+1}",
                        'Overall Quality': chunk['quality_score'],
                        'Readability Score': chunk['readability']['flesch_score'],
                        'Clarity Score': chunk['clarity_score'],
                        'Schema Suitability': chunk['schema_score'],
                        'Topic Coverage': chunk['topic_coverage']['score'],
                        'Token Count': chunk['token_count']
                    })
                
                scoring_df = pd.DataFrame(scoring_data)
                
                # Display scoring table
                st.dataframe(scoring_df, use_container_width=True)
                
                # Scoring visualization
                fig = px.radar(
                    scoring_df.melt(id_vars=['Chunk'], 
                                   value_vars=['Readability Score', 'Clarity Score', 'Schema Suitability', 'Topic Coverage']),
                    theta='variable',
                    r='value',
                    color='Chunk',
                    title='Multi-Dimensional Content Scoring'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Entity Coverage Tab
            with tabs[2]:
                st.subheader("🎯 Entity & Topic Coverage Analysis")
                
                for i, chunk in enumerate(chunks):
                    with st.expander(f"📊 Chunk {i+1} Entity Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📈 Topic Coverage:**")
                            coverage = chunk['topic_coverage']
                            st.metric("Coverage Score", f"{coverage['score']:.1f}%")
                            st.metric("Entities Found", f"{coverage['found']}/{coverage['expected']}")
                            
                            if coverage['missing']:
                                st.markdown("**Missing Relevant Topics:**")
                                for missing in coverage['missing'][:5]:
                                    st.markdown(f"• {missing}")
                        
                        with col2:
                            st.markdown("**🏷️ Extracted Entities:**")
                            if chunk['entities']:
                                for entity in chunk['entities'][:8]:
                                    if isinstance(entity, dict):
                                        st.markdown(f"• **{entity['text']}** (confidence: {entity['confidence']:.2f})")
                                    else:
                                        st.markdown(f"• {entity}")
                            else:
                                st.info("No entities extracted")
            
            # Glossary Tab
            with tabs[3]:
                st.subheader("📚 Entity Definitions & Glossary")
                
                all_entities = []
                for chunk in chunks:
                    all_entities.extend(chunk['entities'])
                
                if all_entities:
                    unique_entities = {}
                    for entity in all_entities:
                        if isinstance(entity, dict):
                            key = entity['text']
                            if key not in unique_entities:
                                unique_entities[key] = entity
                    
                    st.markdown("### 📖 Entity Tooltips & Definitions")
                    
                    for entity_name, entity_data in list(unique_entities.items())[:10]:
                        with st.expander(f"📝 {entity_name}"):
                            st.markdown(f"**Definition:** {entity_data.get('definition', 'No definition available')}")
                            if 'types' in entity_data and entity_data['types']:
                                st.markdown(f"**Types:** {', '.join(entity_data['types'])}")
                            st.markdown(f"**Confidence:** {entity_data.get('confidence', 0):.2f}")
                else:
                    st.info("No entities found for glossary generation")
            
            # Topic Clustering Tab
            with tabs[4]:
                st.subheader("🗺️ Topic Clustering & Content Map")
                
                if cluster_data:
                    # Cluster visualization
                    fig = create_cluster_visualization(cluster_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster summary
                    st.markdown("### 📋 Cluster Summary")
                    cluster_summary = {}
                    for chunk in chunks:
                        cluster_id = chunk.get('cluster', 0)
                        if cluster_id not in cluster_summary:
                            cluster_summary[cluster_id] = []
                        cluster_summary[cluster_id].append(chunk['text'][:100] + '...')
                    
                    for cluster_id, texts in cluster_summary.items():
                        with st.expander(f"🎯 Cluster {cluster_id + 1} ({len(texts)} chunks)"):
                            for j, text in enumerate(texts):
                                st.markdown(f"**Chunk {j+1}:** {text}")
                else:
                    st.info("Topic clustering requires at least 2 content chunks")
            
            # Visualizations Tab
            with tabs[5]:
                st.subheader("📈 Advanced Visualizations")
                
                # Quality score distribution
                fig1 = px.histogram(
                    x=[chunk['quality_score'] for chunk in chunks],
                    title="Quality Score Distribution",
                    labels={'x': 'Quality Score', 'y': 'Count'}
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Multi-metric comparison
                metrics_df = pd.DataFrame([
                    {
                        'Chunk': f"Chunk {i+1}",
                        'Quality Score': chunk['quality_score'],
                        'Token Count': chunk['token_count'],
                        'Readability': chunk['readability']['flesch_score'],
                        'Clarity': chunk['clarity_score']
                    }
                    for i, chunk in enumerate(chunks)
                ])
                
                fig2 = px.scatter(
                    metrics_df,
                    x='Token Count',
                    y='Readability',
                    size='Quality Score',
                    color='Clarity',
                    hover_name='Chunk',
                    title='Token Count vs Readability (Size: Quality, Color: Clarity)'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Export Tab
            with tabs[6]:
                st.subheader("📤 Export Analysis Results")
                
                # Prepare export data
                export_data = []
                for i, chunk in enumerate(chunks):
                    export_data.append({
                        'chunk_id': i + 1,
                        'quality_score': chunk['quality_score'],
                        'token_count': chunk['token_count'],
                        'readability_score': chunk['readability']['flesch_score'],
                        'clarity_score': chunk['clarity_score'],
                        'schema_type': chunk['content_type'],
                        'schema_suitability': chunk['schema_score'],
                        'topic_coverage_score': chunk['topic_coverage']['score'],
                        'entities_found': len(chunk['entities']),
                        'ambiguous_phrases_count': len(chunk['ambiguous_phrases']),
                        'misaligned_titles_count': len(chunk['misaligned_titles']),
                        'content_preview': chunk['text'][:200] + '...',
                        'suggestions': '; '.join(chunk['suggestions']),
                        'llm_tips': '; '.join(chunk['llm_tips']),
                        'cluster_id': chunk.get('cluster', 'N/A')
                    })
                
                export_df = pd.DataFrame(export_data)
                
                # Summary statistics
                st.markdown("### 📊 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_quality = np.mean([chunk['quality_score'] for chunk in chunks])
                    st.metric("Average Quality Score", f"{avg_quality:.1f}%")
                
                with col2:
                    low_quality_count = sum(1 for chunk in chunks if chunk['quality_score'] < 70)
                    st.metric("Low Quality Chunks", low_quality_count)
                
                with col3:
                    total_entities = sum(len(chunk['entities']) for chunk in chunks)
                    st.metric("Total Entities Found", total_entities)
                
                with col4:
                    avg_readability = np.mean([chunk['readability']['flesch_score'] for chunk in chunks])
                    st.metric("Average Readability", f"{avg_readability:.1f}")
                
                # Flagged content for improvement
                st.markdown("### ⚠️ Content Flagged for Improvement")
                flagged_chunks = [chunk for chunk in chunks if chunk['quality_score'] < 70]
                
                if flagged_chunks:
                    flagged_data = []
                    for i, chunk in enumerate(flagged_chunks):
                        flagged_data.append({
                            'Chunk': f"Chunk {chunks.index(chunk) + 1}",
                            'Quality Score': chunk['quality_score'],
                            'Main Issues': ', '.join([
                                'Low Readability' if chunk['readability']['flesch_score'] < 60 else '',
                                'Too Long' if chunk['token_count'] > 300 else '',
                                'Unclear Content' if chunk['clarity_score'] < 70 else '',
                                'Poor Topic Coverage' if chunk['topic_coverage']['score'] < 60 else '',
                                'Ambiguous Phrases' if chunk['ambiguous_phrases'] else ''
                            ]).strip(', '),
                            'Priority': 'High' if chunk['quality_score'] < 50 else 'Medium'
                        })
                    
                    flagged_df = pd.DataFrame(flagged_data)
                    st.dataframe(flagged_df, use_container_width=True)
                    
                    # Download flagged content
                    csv = flagged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Flagged Content Report",
                        csv,
                        "flagged_content_report.csv",
                        "text/csv",
                        key='download-flagged'
                    )
                else:
                    st.success("🎉 No content flagged for improvement!")
                
                # Full analysis export
                st.markdown("### 📋 Complete Analysis Export")
                
                # Create comprehensive export
                full_csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Complete Analysis",
                    full_csv,
                    "genai_optimization_analysis.csv",
                    "text/csv",
                    key='download-full'
                )
                
                # JSON export for developers
                json_data = {
                    'analysis_summary': {
                        'total_chunks': len(chunks),
                        'average_quality_score': float(np.mean([chunk['quality_score'] for chunk in chunks])),
                        'total_entities': sum(len(chunk['entities']) for chunk in chunks),
                        'analysis_timestamp': pd.Timestamp.now().isoformat()
                    },
                    'chunks': [
                        {
                            'id': i + 1,
                            'quality_score': chunk['quality_score'],
                            'metrics': {
                                'token_count': chunk['token_count'],
                                'readability': chunk['readability'],
                                'clarity_score': chunk['clarity_score'],
                                'schema_score': chunk['schema_score'],
                                'topic_coverage': chunk['topic_coverage']
                            },
                            'content_type': chunk['content_type'],
                            'entities': chunk['entities'][:10],  # Limit for JSON size
                            'suggestions': chunk['suggestions'],
                            'llm_tips': chunk['llm_tips'],
                            'cluster_id': chunk.get('cluster', None)
                        }
                        for i, chunk in enumerate(chunks)
                    ]
                }
                
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    "📥 Download JSON Report (Developers)",
                    json_str.encode('utf-8'),
                    "genai_analysis_report.json",
                    "application/json",
                    key='download-json'
                )
                
                # Action items summary
                st.markdown("### 🎯 Recommended Action Items")
                
                action_items = []
                
                # High priority actions
                high_priority = [chunk for chunk in chunks if chunk['quality_score'] < 50]
                if high_priority:
                    action_items.append({
                        'priority': 'High',
                        'action': f"Rewrite {len(high_priority)} low-quality chunks",
                        'impact': 'Significant improvement in AI search visibility'
                    })
                
                # Medium priority actions
                ambiguous_chunks = [chunk for chunk in chunks if chunk['ambiguous_phrases']]
                if ambiguous_chunks:
                    action_items.append({
                        'priority': 'Medium',
                        'action': f"Fix ambiguous phrases in {len(ambiguous_chunks)} chunks",
                        'impact': 'Better user engagement and clarity'
                    })
                
                # Low priority actions
                long_chunks = [chunk for chunk in chunks if chunk['token_count'] > 300]
                if long_chunks:
                    action_items.append({
                        'priority': 'Low',
                        'action': f"Split {len(long_chunks)} lengthy chunks",
                        'impact': 'Improved readability and processing'
                    })
                
                if action_items:
                    action_df = pd.DataFrame(action_items)
                    st.dataframe(action_df, use_container_width=True)
                else:
                    st.success("🎉 Your content is well-optimized! No major action items.")

# Footer
st.markdown("---")
st.markdown(
    "⚡ **Enhanced GenAI Optimization Checker** - Advanced content analysis for AI search engines, "
    "LLM retrieval readiness, and comprehensive optimization insights."
)
