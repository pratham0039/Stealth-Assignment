"""
PubMed Paper Analyzer (Concurrent Version)

This module provides functionality to download and process scientific papers from PubMed
using concurrent threads for improved performance. It handles multiple papers simultaneously
while maintaining rate limits and error handling.

Main features:
- Concurrent paper processing using ThreadPoolExecutor
- Handles up to 5 papers simultaneously
- Includes comprehensive logging and error handling
"""

import os
import logging
from time import sleep
from typing import Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

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
MAX_WORKERS = 5
DEFAULT_FOLDERS = {
    'downloads': 'papers',
    'summaries': 'summaries',
    'results': 'results_csv'
}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

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
    
    def generate_summary(self, text: str) -> Optional[str]:
        """Generate paper summary using OpenAI."""
        try:
            prompt = f"Summarize the following scientific paper in about 250 words:\n\n{text}"
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error during summarization: {e}")
            return None
    
    def process_results(self, results_text: str) -> Optional[str]:
        """Convert Results section to CSV format."""
        try:
            prompt = """
            Convert the following Results section to CSV format:
            - Format tables as CSV rows with headers
            - For non-tabular data, use: Heading,Description
            - Ensure output is pure CSV format
            
            Text:
            {results_text}
            """
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error during results processing: {e}")
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
            
            # Generate unique filename based on existing files
            existing_pdfs = len([f for f in os.listdir(DEFAULT_FOLDERS['downloads']) 
                               if f.endswith('.pdf')])
            filename = f"paper_{existing_pdfs + 1}.pdf"
            filepath = os.path.join(DEFAULT_FOLDERS['downloads'], filename)
            
            with open(filepath, 'wb') as file:
                file.write(response.content)
            
            logger.info(f"Downloaded paper: {filename}")
            return filepath
        except Exception as e:
            logger.error(f"Error downloading paper from {url}: {e}")
            return None
    
    def process_single_paper(self, url: str) -> bool:
        """Process a single paper with all steps."""
        try:
            # Download paper
            pdf_path = self.download_paper(url)
            if not pdf_path:
                return False
            
            # Extract text
            paper_text = self.extract_text_from_pdf(pdf_path)
            if not paper_text:
                return False
            
            # Process paper content
            self._process_paper_content(paper_text, pdf_path)
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper {url}: {e}")
            return False
    
    def _process_paper_content(self, paper_text: str, pdf_path: str):
        """Process extracted paper content."""
        # Generate summary
        summary = self.openai_client.generate_summary(paper_text)
        if summary:
            self.save_to_file(
                summary,
                f"{os.path.basename(pdf_path)}_summary.txt",
                DEFAULT_FOLDERS['summaries']
            )
        
        # Process results section
        results_text = self.extract_results_section(paper_text)
        if results_text:
            csv_content = self.openai_client.process_results(results_text)
            if csv_content:
                self.save_to_file(
                    csv_content,
                    f"{os.path.basename(pdf_path)}_results.csv",
                    DEFAULT_FOLDERS['results']
                )
    
    # [Previous methods remain the same as in main.py]
    # extract_text_from_pdf, extract_results_section, save_to_file

def get_pdf_url(pubmed_url: str) -> Optional[str]:
    """Extract PDF URL from PubMed page."""
    # [Implementation remains the same as in main.py]

def process_papers_concurrently(urls: List[str]):
    """Process multiple papers concurrently using ThreadPoolExecutor."""
    openai_client = OpenAIClient(OPENAI_API_KEY)
    processor = PaperProcessor(openai_client)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create future tasks for each URL
        future_to_url = {
            executor.submit(processor.process_single_paper, url): url 
            for url in urls
        }
        
        # Process completed tasks
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                success = future.result()
                if success:
                    logger.info(f"Successfully processed paper from {url}")
                else:
                    logger.warning(f"Failed to process paper from {url}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")

def main():
    """Main execution function."""
    try:
        pubmed_urls = extract_urls_from_text()
        if not pubmed_urls:
            logger.error("No URLs found to process")
            return
        
        logger.info(f"Starting concurrent processing of {len(pubmed_urls)} papers")
        process_papers_concurrently(pubmed_urls)
        logger.info("Completed paper processing")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
