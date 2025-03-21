import os
import sys
import json
import requests
import re
import resource
from typing import List, Dict, Optional
import google.generativeai as genai
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import tqdm
import time
from pathlib import Path

# Download nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set environment variable to avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Increase file descriptor limit
def increase_file_limit():
    """Increase the soft limit for the number of file descriptors."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        # Try to set soft limit to hard limit, or a reasonable number if that fails
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Increased file descriptor limit from {soft} to {new_soft}")
    except ValueError as e:
        print(f"Warning: Could not increase file descriptor limit: {e}")
        print("Current limits - Soft: {soft}, Hard: {hard}")

# API Configuration
OLLAMA_HOST = '127.0.0.1:11434'
OLLAMA_URI = f'http://{OLLAMA_HOST}/api/chat'
OPENROUTER_URI = 'https://openrouter.ai/api/v1/chat/completions'
API_TYPE = 'ollama'  # Can be 'ollama', 'openrouter', or 'google'
MODEL = 'gemma3:4b'  # Default model
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

def check_google_credentials():
    """Check and validate Google Cloud credentials."""
    if not GOOGLE_API_KEY:
        print("\nError: GOOGLE_API_KEY environment variable not set!")
        print("Please set it with: export GOOGLE_API_KEY=your-api-key")
        return False
    return True

# Initialize Google GenAI if needed
if API_TYPE == 'google':
    try:
        if check_google_credentials():
            genai.configure(api_key=GOOGLE_API_KEY)
            print("\nSuccessfully initialized Google GenAI")
        else:
            print("\nFailed to initialize Google GenAI due to missing credentials")
    except Exception as e:
        print(f"\nWarning: Failed to initialize Google GenAI: {e}")

# Memoization cache for API responses
response_cache = {}

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (faster and more accurate than PyPDF2)."""
    try:
        doc = fitz.open(file_path)
        text_parts = []

        # Process pages in parallel
        def extract_page_text(page_num):
            return doc[page_num].get_text()

        # Determine optimal number of workers based on CPU cores and document size
        num_workers = min(multiprocessing.cpu_count(), len(doc))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(extract_page_text, page_num) for page_num in range(len(doc))]
            text_parts = [future.result() for future in as_completed(futures)]

        return "".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    try:
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        print(f"Error splitting text into sentences: {e}")
        # Fallback to simple splitting
        return [s.strip() + '.' for s in text.split('.') if s.strip()]

def group_sentences_into_chunks(sentences: List[str], max_tokens: int = 512, overlap_sentences: int = 3) -> List[str]:
    """Group sentences into chunks that don't exceed max_tokens with optional overlap.
    
    Args:
        sentences: List of sentences to group
        max_tokens: Maximum number of tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks (default 2)
    """
    chunks = []
    current_chunk = []
    current_length = 0

    # Pre-compute token lengths to avoid repeated splits
    sentence_tokens = [len(sentence.split()) for sentence in sentences]

    i = 0
    while i < len(sentences):
        # Add current sentence
        current_chunk.append(sentences[i])
        current_length += sentence_tokens[i]
        
        # Check if we need to create a new chunk
        if current_length > max_tokens and len(current_chunk) > 1:
            # Remove the last sentence as it caused overflow
            current_chunk.pop()
            
            # Create chunk from current sentences
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            # Go back 'overlap_sentences' sentences unless we don't have enough sentences
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_length = sum(sentence_tokens[i - len(current_chunk):i])
            i -= len(current_chunk)  # Adjust i to reprocess overlapped sentences
        
        i += 1

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def is_json(data: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(data)
        return True
    except (ValueError, TypeError):
        return False

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from response text, handling different formats."""
    if not text:
        return None

    # Try direct parsing first
    if is_json(text):
        return json.loads(text)

    # Try to extract JSON from code blocks or backticks
    patterns = [
        r'```(?:json)?\s*([\s\S]*?)\s*```',  # Code blocks with or without json tag
        r'`([\s\S]*?)`',                      # Backticks
        r'{[\s\S]*?}'                         # Bare JSON object
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.S)
        for match in matches:
            if is_json(match):
                return json.loads(match)
            # Try adding brackets if it might be JSON without them
            if not match.strip().startswith('{'):
                bracketed = '{' + match + '}'
                if is_json(bracketed):
                    return json.loads(bracketed)

    return None

def make_api_request(prompt: str) -> Optional[str]:
    """Centralized API request handling for all providers."""
    try:
        if API_TYPE == 'google':
            if not GOOGLE_API_KEY:
                raise ValueError("Google API key is required but not set in environment")
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content(prompt)
            return response.text

        elif API_TYPE == 'openrouter':
            if not OPENROUTER_API_KEY:
                raise ValueError("OpenRouter API key is required but not set in environment")
            
            headers = {
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'HTTP-Referer': 'http://localhost:3000',
                'X-Title': 'PDF QA Extractor'
            }
            request_data = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            response = requests.post(
                OPENROUTER_URI,
                json=request_data,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        else:  # ollama
            response = requests.post(
                OLLAMA_URI,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")

    except Exception as e:
        print(f"API request error: {e}")
        return None

def handle_chunk_processing(chunk_idx: int, result: Optional[Dict]) -> None:
    """Centralized chunk processing status reporting."""
    if result:
        print(f"Successfully processed chunk {chunk_idx + 1}")
    else:
        print(f"Failed to process chunk {chunk_idx + 1}")

def save_results(output_path: str, responses: List[Dict]) -> None:
    """Centralized results saving with consistent formatting."""
    if responses:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2)
        print(f"Wrote {len(responses)} responses to {output_path}")
    else:
        print("No successful responses to write.")

def submit_to_api(chunk: str, retries: int = 3) -> Optional[Dict]:
    """Submit a chunk of text to the API with retry logic."""
    cache_key = hash(chunk)
    if cache_key in response_cache:
        return response_cache[cache_key]

    prompt = f"""You are an AI assistant that converts text into question-answer pairs in JSON format.
For the following text, create a single question and answer pair that captures key information.
Only respond with a JSON object containing "question" and "answer" fields. No other text.

Text to process:
{chunk.strip()}
"""

    for i in range(retries):
        content = make_api_request(prompt)
        
        if not content:
            print(f"Attempt {i + 1} failed with null response. Retrying after delay...")
            time.sleep(2 * (i + 1))
            continue

        json_data = extract_json_from_text(content)
        if json_data:
            response_cache[cache_key] = json_data
            return json_data

        print(f"Attempt {i + 1} failed with invalid JSON. Retrying after delay...")
        time.sleep(2 * (i + 1))

    print("Max retries exceeded. Skipping this chunk.")
    return None

def process_chunks_in_parallel(chunks: List[str], max_workers: Optional[int] = None) -> List[Dict]:
    """Process chunks with limited concurrency to avoid overwhelming the server."""
    if max_workers is None:
        max_workers = min(2, multiprocessing.cpu_count())
    
    responses = []
    batch_size = min(32, max_workers * 4)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(submit_to_api, chunk): idx + i 
                             for idx, chunk in enumerate(batch)}
            
            for future in tqdm.tqdm(as_completed(future_to_chunk), 
                                  total=len(batch),
                                  desc=f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}"):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        responses.append(result)
                    handle_chunk_processing(chunk_idx, result)
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx + 1}: {e}")
                
                time.sleep(1.0)
    
    return responses

def process_pdf(input_path: str, output_path: str = 'responses.json', max_workers: Optional[int] = None, max_tokens: int = 800, overlap: int = -2) -> None:
    """Process a PDF file and save extracted QA pairs."""
    try:
        # Extract text
        print("Extracting text from PDF...")
        text = extract_text_from_pdf(input_path)
        if not text:
            print("Failed to extract text from PDF.")
            return

        # Split into sentences
        print("Splitting text into sentences...")
        sentences = split_into_sentences(text)
        print(f"Found {len(sentences)} sentences")

        # Group into chunks with overlap
        print("Grouping sentences into chunks...")
        chunks = group_sentences_into_chunks(sentences, max_tokens, overlap)
        print(f"Created {len(chunks)} chunks with {overlap} sentence overlap")

        # Process chunks in parallel
        print("Processing chunks in parallel...")
        responses = process_chunks_in_parallel(chunks, max_workers)

        # Write responses to file
        save_results(output_path, responses)

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    # Increase file descriptor limit at startup
    increase_file_limit()
    
    import argparse
    from datetime import datetime

    # List of all available models
    MODELS = [
        "gemma3:4b",  # Ollama model
        "openai/gpt-3.5-turbo",  # OpenRouter model
        "anthropic/claude-2",     # OpenRouter model
        "gemini-pro",            # Google model
        "gemini-2.0-flash-001",  # Google model
        "meta-llama/llama-2-70b-chat",  # OpenRouter model
    ]

    parser = argparse.ArgumentParser(description='Process PDF into question-answer pairs')
    parser.add_argument('--workers', '-w', type=int, default=None,
                      help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--model', '-m', default=None,
                      help='Single model to test (default: test all models)')
    parser.add_argument('--api', '-a', choices=['ollama', 'openrouter', 'vertex', 'google'], default='ollama',
                      help='API to use (default: ollama)')
    parser.add_argument('--overlap', '-o', type=int, default=2,
                      help='Number of sentences to overlap between chunks (default: -2)')
    parser.add_argument('--max-tokens', '-t', type=int, default=800,
                      help='Maximum number of tokens per chunk (default: 800)')
    
    args = parser.parse_args()

    # Set the API type based on command line argument
    API_TYPE = args.api

    # Validate API configurations
    if API_TYPE == 'openrouter' and not OPENROUTER_API_KEY:
        print("Error: OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)
    elif API_TYPE in ['vertex', 'google']:
        if not check_google_credentials():
            sys.exit(1)

    # Initialize Google GenAI if needed
    if API_TYPE == 'google':
        try:
            genai.configure(
                api_key=GOOGLE_API_KEY
            )
            print(f"\nSuccessfully initialized Google GenAI")
        except Exception as e:
            print(f"\nError: Failed to initialize Google GenAI: {e}")
            sys.exit(1)

    # Results dictionary to store timing information
    results = {}
    
    # If a specific model is specified, only test that one
    models_to_test = [args.model] if args.model else MODELS

    # Define source and results directories
    ebooks_dir = Path.home() / "ebooks"
    results_dir = Path("results")

    # Create results directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)

    # Find all PDF files in the ebooks directory
    pdf_files = list(ebooks_dir.rglob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    print("=" * 80)

    for model in models_to_test:
        MODEL = model
        print(f"\nProcessing files using model: {model}")
        print("-" * 60)

        # Clear the response cache for each new model
        response_cache.clear()

        for pdf_file in pdf_files:
            # Calculate the relative path from ebooks_dir
            rel_path = pdf_file.relative_to(ebooks_dir)
            
            # Create corresponding output directory structure
            output_dir = results_dir / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename (change extension from .pdf to .json)
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            # Skip if output file already exists
            if output_file.exists():
                print(f"\nSkipping already processed file: {pdf_file}")
                print(f"Output file exists: {output_file}")
                continue
            
            print(f"\nProcessing: {pdf_file}")
            print(f"Output to: {output_file}")
            
            start_time = datetime.now()
            
            try:
                process_pdf(
                    str(pdf_file), 
                    str(output_file), 
                    args.workers,
                    args.max_tokens,
                    args.overlap
                )
                
                end_time = datetime.now()
                duration = end_time - start_time
                
                # Store results
                results[str(pdf_file)] = {
                    'model': model,
                    'duration': str(duration),
                    'output_file': str(output_file)
                }
                
                print(f"Completed in: {duration}")
                
            except Exception as e:
                print(f"Error processing file {pdf_file}: {e}")
                results[str(pdf_file)] = {
                    'model': model,
                    'error': str(e),
                    'output_file': str(output_file)
                }
            
            print("-" * 60)
    
    # Print summary of all results
    print("\nSummary of Results:")
    print("=" * 80)
    for file_path, result in results.items():
        print(f"\nFile: {file_path}")
        print(f"Model: {result['model']}")
        if 'duration' in result:
            print(f"Duration: {result['duration']}")
            print(f"Output: {result['output_file']}")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
    
    print("\nProcessing complete!")