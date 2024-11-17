"""
PubMed Paper Analyzer

This module provides functionality to download, process, and analyze scientific papers from PubMed.
It extracts results sections, generates summaries using OpenAI's GPT-4, and saves the data in structured formats.

Main features:
- Downloads papers from PubMed URLs by visiting the child pages also.
- Extracts text content from PDFs
- Generates paper summaries using OpenAI
- Extracts and structures Results sections
- Saves processed data in CSV and text formats
"""

import os
import logging
from time import sleep
from typing import Optional, List
from dataclasses import dataclass
from urllib.parse import urljoin
from functools import lru_cache
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import requests
from openai import OpenAI
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from extract_url import extract_urls_from_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_KEY = "your-api-key-here"  # Move to environment variable
DEFAULT_FOLDERS = {
    'downloads': 'papers',
    'summaries': 'summaries',
    'results': 'results_csv'
}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
CHUNK_SIZE = 4000  # Maximum tokens for GPT-4
SUMMARY_LENGTH = 250  # Target summary length in words

@dataclass
class PaperContent:
    """Data class to store paper content and metadata."""
    text: str
    filename: str
    url: str

class OpenAIClient:
    """Handles interactions with OpenAI API."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    @lru_cache(maxsize=100)
    def _get_important_sentences(self, text: str, num_sentences: int = 10) -> str:
        #Extract most important sentences using TF-IDF scoring.
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text
            
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            importance_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            top_indices = np.argsort(importance_scores)[-num_sentences:]
            top_indices = sorted(top_indices)
            
            return " ".join([sentences[i] for i in top_indices])
        except Exception as e:
            logger.error(f"Error in sentence extraction: {e}")
            return text

    def _chunk_text(self, text: str) -> List[str]:
        #Split text into chunks that fit within token limits.
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sent_tokenize(text):
            sentence_length = len(word_tokenize(sentence))
            
            if current_length + sentence_length > CHUNK_SIZE:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def generate_completion(self, prompt: str, model: str = "gpt-4") -> Optional[str]:
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def generate_summary(self, text: str) -> Optional[str]:
        
        try:
            # Extract important sentences first
            important_text = self._get_important_sentences(text)
            
            
            chunks = self._chunk_text(important_text)
            
            # Process chunks sequentially (since we're not using async here)
            chunk_summaries = []
            for chunk in chunks:
                prompt = f"Summarize this section of a scientific paper concisely:\n\n{chunk}"
                summary = self.generate_completion(prompt)
                if summary:
                    chunk_summaries.append(summary)
            
            if not chunk_summaries:
                return None
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # Generate final summary
            final_prompt = f"""
            Create a coherent {SUMMARY_LENGTH}-word summary from these section summaries:
            
            {combined_summary}
            """
            
            return self.generate_completion(final_prompt)
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return None

class PaperProcessor:
    """Handles downloading and processing of scientific papers."""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
        self._setup_folders()
    
    def _setup_folders(self):
        """Create necessary folders if they don't exist."""
        for folder in DEFAULT_FOLDERS.values():
            os.makedirs(folder, exist_ok=True)
    
    def download_paper(self, url: str) -> Optional[str]:
        """Download paper from given URL."""
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            filename = url.split("/")[-1]
            filepath = os.path.join(DEFAULT_FOLDERS['downloads'], f"{filename}.pdf")
            
            with open(filepath, 'wb') as file:
                file.write(response.content)
            
            logger.info(f"Downloaded paper: {filename}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading paper from {url}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text content from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = " ".join(page.extract_text() for page in reader.pages)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def extract_results_section(self, text: str) -> Optional[str]:
        """Extract the Results section from paper text."""
        try:
            lower_text = text.lower()
            start_idx = lower_text.find("results")
            end_idx = lower_text.find("discussion", start_idx)
            
            if start_idx == -1 or end_idx == -1:
                logger.warning("Results or Discussion section not found")
                return None
            
            return text[start_idx:end_idx].strip()
        except Exception as e:
            logger.error(f"Error extracting Results section: {e}")
            return None
    
    def process_results_to_csv(self, results_text: str) -> Optional[str]:
        """Convert Results section to CSV format using OpenAI."""
        prompt = """
        Convert the following Results section to CSV format:
        - Format tables as CSV rows with headers
        - For non-tabular data, use: Heading,Description
        - Ensure output is pure CSV format
        
        Text:
        {results_text}
        """
        return self.openai_client.generate_completion(prompt)
    
    def save_to_file(self, content: str, filename: str, folder: str) -> Optional[str]:
        """Save content to file in specified folder."""
        try:
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
            logger.info(f"Saved file: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return None

def get_pdf_url(pubmed_url: str) -> Optional[str]:
    """Extract PDF URL from PubMed page."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(pubmed_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        full_text_links = soup.find("div", class_="full-text-links-list")
        if not full_text_links:
            return None
        
        first_link = full_text_links.find("a", href=True)
        if not first_link:
            return None
        
        full_text_url = first_link["href"]
        full_text_response = requests.get(full_text_url, headers=headers, timeout=30)
        full_text_response.raise_for_status()
        
        full_text_soup = BeautifulSoup(full_text_response.content, "html.parser")
        pdf_links = [
            a["href"] for a in full_text_soup.find_all("a", href=True)
            if a["href"].endswith(".pdf")
        ]
        
        if not pdf_links:
            return None
        
        pdf_url = urljoin(full_text_url, pdf_links[0])
        return pdf_url
    except Exception as e:
        logger.error(f"Error extracting PDF URL from {pubmed_url}: {e}")
        return None

def main():
   
    openai_client = OpenAIClient(OPENAI_API_KEY)
    processor = PaperProcessor(openai_client)
    
    pubmed_urls = extract_urls_from_text()
    
    for url in pubmed_urls:
        try:
            pdf_url = get_pdf_url(url)
            if not pdf_url:
                continue
                
            logger.info(f"Processing PDF: {pdf_url}")
            
            # Download and process paper
            pdf_path = processor.download_paper(pdf_url)
            if not pdf_path:
                continue
                
            # Extract and process text
            paper_text = processor.extract_text_from_pdf(pdf_path)
            if not paper_text:
                continue
                
            # Process results section
            results_text = processor.extract_results_section(paper_text)
            if results_text:
                csv_content = processor.process_results_to_csv(results_text)
                if csv_content:
                    processor.save_to_file(
                        csv_content,
                        f"{os.path.basename(pdf_path)}_results.csv",
                        DEFAULT_FOLDERS['results']
                    )
            
            # Add delay to respect rate limits
            sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue

if __name__ == "__main__":
    main()
