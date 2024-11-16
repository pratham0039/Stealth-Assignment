## README for Assignment

### 1. **Setup Instructions**

To set up and run this project, follow these steps:

#### Prerequisites:
Ensure that you have Python 3.8 or later installed on your machine. You will also need to have `pip` installed for package management.

#### Steps:

1. **Install the required Python libraries:**
   You can install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API Key:**
   - Go to [OpenAI's API platform](https://platform.openai.com/) to obtain your API key.
   - In the code, replace the `OPENAI_API_KEY` constant with your API key:
   ```python
   OPENAI_API_KEY = "your-api-key-here"  # Move to environment variable
   ```

3. **Add the URLs:**
   To process papers from PubMed URLs which are in text format for now, just copy them from doc and paste it in the variable urls in extract_url.py file:

  
4. **Run the Script:**
   To process papers from PubMed URLs, simply execute the main script:
   ```bash
   python main.py
   ```

   The script will download papers, summarize them, and generate the results.

---

5. **Run the API:**
   To use the API:
   ```bash
   python app.py
   ```

   The script will open a local sever and you can send a request at url '/summarize' with a url in the format of
   {
    "url": "https://pubmed.ncbi.nlm.nih.gov/39367479/?utm_source=Other&utm_medium=rss&utm_campaign=pubmed-2&utm_content=1xUFNDUH9GgcD9fNW6SIziqcwgAz3kWw5uqQ3XvLCRGR9gXMXt&fc=20241006182642&ff=20241006182650&v=2.18.0.post9+e462414"
}

---

### 2. **Usage Guide**

The script is designed to:

1. Download papers from PubMed using their URLs.
2. Extract and summarize the content of the paper, getting the results of the paper..
3. Use OpenAI's GPT-3 model to generate a structured CSV of the 'Results' section.
4. Save the downloaded pdfs, summaries and structured results in output folders.

#### Main functions:

- **`download_paper(url, download_folder)`**:
   Downloads the paper from the provided URL and saves it to the specified folder.

- **`extract_text_from_pdf(pdf_path)`**:
   Extracts text content from the downloaded PDF file.

- **`summarize_paper_with_openai(paper_text)`**:
   Summarizes the extracted paper text using OpenAI GPT-3.

- **`extract_results_section(paper_text)`**:
   Extracts the 'Results' section from the paper text.

- **`process_results_with_openai(results_text)`**:
   Processes the 'Results' section using OpenAI to structure it as a CSV.

- **`save_summary_to_file(summary, pdf_filename)`**:
   Saves the summary content to a text file.

- **`save_results_to_csv(csv_content, pdf_filename)`**:
   Saves the CSV content to a file.

- **`process_papers(url)`**:
   Orchestrates the entire process from downloading to generating results.

---

### 3. **Explanation of Approach & Assumptions Made**

#### Approach:
The approach I used was indirect; you cannot download the PDFs directly from the provided links. Here is the step-by-step process to do so.

1. **Downloading Papers**: To download the papers, I first need to visit the URL. The URL doesn't provide a direct PDF, but instead links to the full text, which we need to access. After reaching that page, I must find the URL of the PDF and use that to download it.

During this process, some pages were blocking the request. I used headers to bypass some of these blocks, but there were still some pages that did not allow me to scrape or visit them.

2. **Extracting Text**: PDF content is extracted using the PyPDF2 library, which converts the text to a readable format.
3. **Summarizing Content**: The paper's content is summarized using OpenAI's GPT-3 model, providing a concise version of the original text.
4. **Extracting Results Section**: The 'Results' section is identified in the paper, I noticed that most of the papers do not have a table in their result sections. So I assumed the headings and description as the columns. I was not sure how to do it because i could not find a pattern in that that's why I used extractd the text between 'Result' section and 'Discussion' session and gave it to Open AI to convert it into a structured CSV. 

5. **Saving Outputs**: The summary and results are saved to respective files in designated output folders.

6. **For Better Performance**: I tried to use concurrent threads (you can find the code in Code_with_concurrent_threads.py) for better performance and doing 5 tasks together.

7. **API Development**: I made an API also which takes any PubMed URL and gives the summary of the paper linked to it. It included in this project for any other endpoints you might want to expose




#### Assumptions:
- The 'Results' section in the papers is always preceded by the word "Results" and followed by the word "Discussion". This assumption may need to be refined for papers with different formatting.
- The PDF is assumed to be in a readable format; some PDFs might not extract clean text, especially those with complex formatting.

---

### 4. **List of Third-Party Libraries Used**

1. **`requests`**:
   - Purpose: Handles HTTP requests for downloading the papers from PubMed and accessing full-text URLs.
   
2. **`PyPDF2`**:
   - Purpose: Extracts text content from the downloaded PDF papers.

3. **`openai`**:
   - Purpose: Interacts with OpenAI's API for text summarization and structured CSV generation using GPT-3.
   - We could use hugginf face transformers or any other open source LLM for this, but I prefer open AI because it generally gives better and fast response.

4. **`BeautifulSoup`**:
   - Purpose: Scrapes and parses the HTML content from PubMed pages to extract full-text links.

5. **`urllib.parse`**:
   - Purpose: Helps in joining URLs and resolving relative links.

6. **`concurrent.futures`**:
   - Purpose: Enables parallel processing with threads to download and process multiple papers at the same time.

7. **`extract_url`** (Custom Script):
   - Purpose: A helper script that extracts URLs from a provided text input, specifically designed for PubMed URLs.

---

### 5. **Configuration File** 

Requirements.txt file is in the code you can just use that to configure.

