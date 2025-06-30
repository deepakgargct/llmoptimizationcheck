#!/usr/bin/env python3
"""
GenAI Optimization Automation Script
Automates content analysis and optimization for AI retrieval systems
"""

import requests
import json
import re
import csv
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, flesch_kincaid_grade
import pandas as pd
from urllib.parse import urljoin, urlparse
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContentChunk:
    """Represents a content chunk for analysis"""
    content: str
    heading: str
    token_count: int
    word_count: int
    readability_score: float
    has_self_contained_idea: bool
    semantic_tags: List[str]

@dataclass
class OptimizationReport:
    """Comprehensive optimization report"""
    url: str
    total_tokens: int
    total_chunks: int
    readability_score: float
    issues: List[str]
    recommendations: List[str]
    chunk_analysis: List[ContentChunk]
    internal_links: List[str]
    external_links: List[str]
    schema_markup: Dict
    meta_data: Dict

class GenAIOptimizer:
    """Main class for GenAI content optimization"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.stop_words = set(stopwords.words('english'))
        
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer"""
        return len(self.tokenizer.encode(text))
    
    def analyze_readability(self, text: str) -> Dict:
        """Analyze text readability"""
        return {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(sent_tokenize(text)),
            'avg_sentence_length': len(word_tokenize(text)) / max(1, len(sent_tokenize(text)))
        }
    
    def extract_content_from_url(self, url: str) -> BeautifulSoup:
        """Extract content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def analyze_chunk_structure(self, soup: BeautifulSoup) -> List[ContentChunk]:
        """Analyze content chunk structure"""
        chunks = []
        
        # Find all headings and content blocks
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section']):
            if element.name.startswith('h'):
                # Process heading and following content
                heading = element.get_text(strip=True)
                content_blocks = []
                
                # Collect content until next heading
                for sibling in element.next_siblings:
                    if hasattr(sibling, 'name'):
                        if sibling.name and sibling.name.startswith('h'):
                            break
                        if sibling.name in ['p', 'div', 'ul', 'ol', 'blockquote']:
                            text = sibling.get_text(strip=True)
                            if text:
                                content_blocks.append(text)
                
                if content_blocks:
                    full_content = ' '.join(content_blocks)
                    token_count = self.count_tokens(full_content)
                    word_count = len(word_tokenize(full_content))
                    readability = self.analyze_readability(full_content)
                    
                    chunk = ContentChunk(
                        content=full_content[:500] + '...' if len(full_content) > 500 else full_content,
                        heading=heading,
                        token_count=token_count,
                        word_count=word_count,
                        readability_score=readability['flesch_reading_ease'],
                        has_self_contained_idea=self.check_self_contained_idea(full_content),
                        semantic_tags=self.extract_semantic_tags(sibling if 'sibling' in locals() else element)
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def check_self_contained_idea(self, text: str) -> bool:
        """Check if content chunk contains a self-contained idea"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return False
        
        # Simple heuristic: check for pronouns without clear antecedents
        pronouns = ['this', 'that', 'these', 'those', 'it', 'they', 'them']
        first_sentence = sentences[0].lower()
        
        for pronoun in pronouns:
            if pronoun in first_sentence.split()[:3]:  # Pronoun in first 3 words
                return False
        
        return True
    
    def extract_semantic_tags(self, element) -> List[str]:
        """Extract semantic HTML tags"""
        tags = []
        if hasattr(element, 'name') and element.name:
            tags.append(element.name)
        
        # Check for semantic attributes
        if hasattr(element, 'attrs'):
            if 'class' in element.attrs:
                tags.extend(element.attrs['class'])
            if 'id' in element.attrs:
                tags.append(element.attrs['id'])
        
        return tags
    
    def analyze_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Analyze internal linking structure"""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            link_domain = urlparse(absolute_url).netloc
            
            if link_domain == base_domain:
                anchor_text = link.get_text(strip=True)
                links.append({
                    'url': absolute_url,
                    'anchor_text': anchor_text,
                    'is_descriptive': len(anchor_text.split()) > 2,
                    'context': self.get_link_context(link)
                })
        
        return links
    
    def get_link_context(self, link_element) -> str:
        """Get surrounding context for a link"""
        parent = link_element.parent
        if parent:
            text = parent.get_text(strip=True)
            return text[:200] + '...' if len(text) > 200 else text
        return ""
    
    def analyze_schema_markup(self, soup: BeautifulSoup) -> Dict:
        """Analyze schema.org markup"""
        schema_data = {
            'json_ld': [],
            'microdata': [],
            'rdfa': []
        }
        
        # JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                schema_data['json_ld'].append(data)
            except:
                pass
        
        # Microdata
        for element in soup.find_all(attrs={'itemscope': True}):
            item_type = element.get('itemtype', '')
            schema_data['microdata'].append(item_type)
        
        # RDFa
        for element in soup.find_all(attrs={'typeof': True}):
            typeof = element.get('typeof', '')
            schema_data['rdfa'].append(typeof)
        
        return schema_data
    
    def check_ai_crawlability(self, url: str) -> Dict:
        """Check if site is crawlable by AI bots"""
        try:
            robots_url = urljoin(url, '/robots.txt')
            response = requests.get(robots_url, timeout=10)
            robots_txt = response.text if response.status_code == 200 else ""
            
            blocked_bots = []
            ai_bots = ['GPTBot', 'Google-Extended', 'CCBot', 'anthropic-ai', 'Claude-Web']
            
            for bot in ai_bots:
                if f'User-agent: {bot}' in robots_txt and 'Disallow:' in robots_txt:
                    blocked_bots.append(bot)
            
            return {
                'robots_txt_exists': response.status_code == 200,
                'blocked_ai_bots': blocked_bots,
                'robots_content': robots_txt[:500] + '...' if len(robots_txt) > 500 else robots_txt
            }
        except:
            return {'robots_txt_exists': False, 'blocked_ai_bots': [], 'robots_content': ''}
    
    def generate_optimization_recommendations(self, chunks: List[ContentChunk], 
                                           links: List[Dict], 
                                           schema: Dict) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Chunk analysis
        oversized_chunks = [c for c in chunks if c.token_count > 300]
        undersized_chunks = [c for c in chunks if c.token_count < 50]
        low_readability_chunks = [c for c in chunks if c.readability_score < 60]
        
        if oversized_chunks:
            recommendations.append(f"Break down {len(oversized_chunks)} chunks that exceed 300 tokens")
        
        if undersized_chunks:
            recommendations.append(f"Expand {len(undersized_chunks)} chunks that are under 50 tokens")
        
        if low_readability_chunks:
            recommendations.append(f"Improve readability for {len(low_readability_chunks)} chunks with low scores")
        
        # Link analysis
        non_descriptive_links = [l for l in links if not l['is_descriptive']]
        if non_descriptive_links:
            recommendations.append(f"Improve {len(non_descriptive_links)} links with non-descriptive anchor text")
        
        # Schema analysis
        if not schema['json_ld'] and not schema['microdata']:
            recommendations.append("Add schema.org markup to improve content understanding")
        
        return recommendations
    
    def analyze_content(self, url: str) -> OptimizationReport:
        """Main analysis function"""
        logger.info(f"Starting analysis of {url}")
        
        soup = self.extract_content_from_url(url)
        if not soup:
            return None
        
        # Analyze chunks
        chunks = self.analyze_chunk_structure(soup)
        
        # Analyze links
        internal_links = self.analyze_internal_links(soup, url)
        external_links = [link for link in soup.find_all('a', href=True) 
                         if not link['href'].startswith(urlparse(url).netloc)]
        
        # Analyze schema
        schema = self.analyze_schema_markup(soup)
        
        # Check crawlability
        crawlability = self.check_ai_crawlability(url)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(chunks, internal_links, schema)
        
        # Calculate totals
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_readability = sum(chunk.readability_score for chunk in chunks) / max(1, len(chunks))
        
        # Identify issues
        issues = []
        if crawlability['blocked_ai_bots']:
            issues.append(f"AI bots blocked: {', '.join(crawlability['blocked_ai_bots'])}")
        if total_tokens > 100000:
            issues.append("Content may be too long for some AI context windows")
        if avg_readability < 60:
            issues.append("Overall readability could be improved")
        
        return OptimizationReport(
            url=url,
            total_tokens=total_tokens,
            total_chunks=len(chunks),
            readability_score=avg_readability,
            issues=issues,
            recommendations=recommendations,
            chunk_analysis=chunks,
            internal_links=[link['url'] for link in internal_links],
            external_links=[link.get('href', '') for link in external_links],
            schema_markup=schema,
            meta_data=crawlability
        )
    
    def test_ai_retrieval(self, url: str, test_queries: List[str]) -> Dict:
        """Test content retrieval on AI platforms"""
        results = {
            'url': url,
            'test_queries': test_queries,
            'results': {
                'perplexity_mentions': 0,
                'total_tests': len(test_queries),
                'successful_retrievals': []
            }
        }
        
        logger.info("Note: AI retrieval testing requires manual verification")
        logger.info("Recommended test queries:")
        for query in test_queries:
            logger.info(f"  - {query}")
        
        return results
    
    def export_report(self, report: OptimizationReport, filename: str = None):
        """Export optimization report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"genai_optimization_report_{timestamp}.json"
        
        # Convert report to JSON-serializable format
        report_dict = {
            'url': report.url,
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_tokens': report.total_tokens,
                'total_chunks': report.total_chunks,
                'avg_readability_score': report.readability_score,
                'issues_count': len(report.issues),
                'recommendations_count': len(report.recommendations)
            },
            'issues': report.issues,
            'recommendations': report.recommendations,
            'chunks': [
                {
                    'heading': chunk.heading,
                    'token_count': chunk.token_count,
                    'word_count': chunk.word_count,
                    'readability_score': chunk.readability_score,
                    'has_self_contained_idea': chunk.has_self_contained_idea,
                    'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                }
                for chunk in report.chunk_analysis
            ],
            'internal_links_count': len(report.internal_links),
            'external_links_count': len(report.external_links),
            'schema_markup': report.schema_markup,
            'crawlability': report.meta_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report exported to {filename}")
        return filename

def main():
    """Main execution function"""
    # Configuration
    config = {
        'target_chunk_size': 200,  # tokens
        'min_readability_score': 60,
        'max_content_length': 100000  # tokens
    }
    
    # Initialize optimizer
    optimizer = GenAIOptimizer(config)
    
    # Example usage
    url = input("Enter URL to analyze: ").strip()
    if not url:
        url = "https://example.com"  # Default for testing
    
    try:
        # Analyze content
        report = optimizer.analyze_content(url)
        
        if report:
            # Print summary
            print(f"\n{'='*50}")
            print(f"GenAI Optimization Report for: {url}")
            print(f"{'='*50}")
            print(f"Total Tokens: {report.total_tokens:,}")
            print(f"Total Chunks: {report.total_chunks}")
            print(f"Avg Readability Score: {report.readability_score:.1f}")
            print(f"Issues Found: {len(report.issues)}")
            print(f"Recommendations: {len(report.recommendations)}")
            
            if report.issues:
                print(f"\n🚨 Issues:")
                for issue in report.issues:
                    print(f"  - {issue}")
            
            if report.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in report.recommendations:
                    print(f"  - {rec}")
            
            # Export report
            filename = optimizer.export_report(report)
            print(f"\n📄 Detailed report saved to: {filename}")
            
            # Generate test queries
            print(f"\n🔍 Suggested test queries for AI platforms:")
            domain = urlparse(url).netloc.replace('www.', '')
            test_queries = [
                f"What does {domain} say about...",
                f"According to {domain}...",
                f"How does {domain} explain...",
                f"{domain} recommendations for..."
            ]
            
            for query in test_queries:
                print(f"  - {query}")
            
            print(f"\n✅ Analysis complete! Test these queries on Perplexity.ai and ChatGPT")
            
        else:
            print("❌ Could not analyze the provided URL")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()

# Additional utility functions for batch processing
def batch_analyze_urls(urls: List[str], config: Dict = None) -> List[OptimizationReport]:
    """Analyze multiple URLs in batch"""
    optimizer = GenAIOptimizer(config)
    reports = []
    
    for url in urls:
        logger.info(f"Analyzing {url}")
        try:
            report = optimizer.analyze_content(url)
            if report:
                reports.append(report)
            time.sleep(2)  # Be respectful to servers
        except Exception as e:
            logger.error(f"Failed to analyze {url}: {e}")
    
    return reports

def export_batch_summary(reports: List[OptimizationReport], filename: str = None):
    """Export summary of batch analysis to CSV"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_optimization_summary_{timestamp}.csv"
    
    data = []
    for report in reports:
        data.append({
            'URL': report.url,
            'Total Tokens': report.total_tokens,
            'Total Chunks': report.total_chunks,
            'Avg Readability': report.readability_score,
            'Issues Count': len(report.issues),
            'Recommendations Count': len(report.recommendations),
            'Internal Links': len(report.internal_links),
            'Has Schema': bool(report.schema_markup['json_ld'] or report.schema_markup['microdata'])
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Batch summary exported to {filename}")
    return filename

"""
Installation Requirements:
pip install requests beautifulsoup4 tiktoken nltk textstat pandas

Usage Examples:

1. Single URL Analysis:
   python genai_optimizer.py

2. Batch Analysis:
   from genai_optimizer import batch_analyze_urls
   urls = ['https://example1.com', 'https://example2.com']
   reports = batch_analyze_urls(urls)

3. Custom Configuration:
   config = {
       'target_chunk_size': 250,
       'min_readability_score': 70
   }
   optimizer = GenAIOptimizer(config)
"""
