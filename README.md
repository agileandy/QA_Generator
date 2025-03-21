# PDF Question-Answer Pair Extractor

# Objective

This application processes PDF documents to automatically generate question-answer pairs using various AI models. It supports processing multiple PDFs in parallel, maintains a directory structure for outputs, and can work with different AI providers including:
- Local Ollama models
- OpenRouter API
- Google Gemini models

The tool is particularly useful for creating educational content, study materials, or documentation from existing PDF resources.

# Config

The application supports several command-line arguments for customization:

```bash
--workers, -w     Number of parallel workers (default: number of CPU cores)
--model, -m       Model to use (default: processes all available models)
--api, -a         API provider to use: ['ollama', 'openrouter', 'google'] (default: ollama)
--overlap, -o     Number of sentences to overlap between chunks (default: -2)
--max-tokens, -t  Maximum tokens per chunk (default: 800)
```

Available Model Providers:
- Ollama: Any local Ollama model.
- Google API : gemini-pro, gemini-2.0-flash-001 etc from Google.
- OpenRouter: Use Openrouter models, including free ones.  (watch for rate limits though).

# Environment Variables

Required environment variables depend on the chosen API:

For Google AI:
```
GOOGLE_API_KEY=your-google-api-key
```

For OpenRouter:
```
OPENROUTER_API_KEY=your-openrouter-api-key
```

For Ollama:
- No environment variables needed
- Ensure Ollama is running locally on port 11434

Additional configuration:
```
TOKENIZERS_PARALLELISM=false  # Set automatically by the app
```

# Install

1. Clone the repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a directory structure:
```
~/ebooks/    # Source PDFs
./results/   # Output JSON files
```

4. Place your PDF files in the `~/ebooks/` directory. The tool will maintain the same directory structure in the `./results/` folder.

5. Run the application:

E.g. 
```bash
pip install -r requirements.txt  
python BookParse.py --api ollama --model gemma3:4b
```


The application will process all PDFs in the ebooks directory, skipping any that have already been processed (based on existing JSON files in the results directory).

---

Feedback and contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

Note: This tool is designed to handle large PDF files efficiently and includes features like automatic file descriptor limit adjustment and parallel processing. For optimal performance, consider adjusting the --workers and --max-tokens parameters based on your system's capabilities.
