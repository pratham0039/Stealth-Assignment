from flask import Flask, request, jsonify
import os
import requests
from openai import OpenAI
from PyPDF2 import PdfReader
from time import sleep
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Flask app setup
app = Flask(__name__)

# OpenAI API setup
client = 'your-open-ai-key'

# Function definitions
def get_pdf_url(pubmed_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(pubmed_url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    full_text_links = soup.find("div", class_="full-text-links-list")
    if not full_text_links:
        return None

    first_link = full_text_links.find("a", href=True)
    if not first_link:
        return None

    full_text_url = first_link["href"]
    full_text_response = requests.get(full_text_url, headers=headers)
    if full_text_response.status_code != 200:
        return None

    full_text_soup = BeautifulSoup(full_text_response.content, "html.parser")
    pdf_links = [a["href"] for a in full_text_soup.find_all("a", href=True) if a["href"].endswith(".pdf")]
    if not pdf_links:
        return None

    pdf_url = urljoin(full_text_url, pdf_links[0])
    return pdf_url

def download_paper(url, download_folder="papers"):
    headers = {"User-Agent": "Chrome/114.0.0.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        os.makedirs(download_folder, exist_ok=True)
        file_path = os.path.join(download_folder, "downloaded_paper.pdf")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    return None

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        return None

def summarize_paper_with_openai(paper_text):
    try:
        prompt = f"Summarize the following scientific paper in about 250 words:\n\n{paper_text}"
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        pubmed_url = data.get('url')
        if not pubmed_url:
            return jsonify({"error": "No URL provided"}), 400

        pdf_url = get_pdf_url(pubmed_url)
        if not pdf_url:
            return jsonify({"error": "Unable to find PDF URL"}), 400

        pdf_path = download_paper(pdf_url)
        if not pdf_path:
            return jsonify({"error": "Unable to download the paper"}), 500

        paper_text = extract_text_from_pdf(pdf_path)
        if not paper_text:
            return jsonify({"error": "Unable to extract text from the PDF"}), 500

        summary = summarize_paper_with_openai(paper_text)
        if not summary:
            return jsonify({"error": "Unable to summarize the paper"}), 500

        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
