
# 🤖 RAG ChatBot – PDF Question Answering Assistant

RAG ChatBot is an AI-powered tool that allows you to ask natural language questions about the content of a PDF file. It uses **Retrieval-Augmented Generation (RAG)** with a language model (Vicuna 7B) and multilingual sentence embeddings to return precise answers from documents.

---

## Features

- 📄 PDF content extraction
- ✂️ Semantic chunking with embeddings
- 🔍 Context-aware question answering
- 🧠 Vicuna-7B language model (4-bit quantized)
- 🌍 Multilingual support (e.g., Vietnamese)
- 🎛️ Clean Streamlit interface

---

## Requirements

- Python >= 3.10
- CUDA-enabled GPU (recommended: >=16GB VRAM)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv rag_env
source rag_env/bin/activate        # macOS/Linux
rag_env\Scripts\activate.bat     # Windows
```

Or using `conda`:

```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install the following:

```txt
transformers==4.52.4
bitsandbytes==0.46.0
accelerate==1.7.0
langchain==0.3.25
langchainhub==0.1.21
langchain-chroma==0.2.4
langchain_experimental==0.3.4
langchain-community==0.3.24
langchain_huggingface==0.2.0
python-dotenv==1.1.0
streamlit==1.46.0
pypdf
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## How to Use

1. Upload your PDF file.
2. Click the "Press to process file" button.
3. Ask a question related to the PDF content.
4. Receive an AI-generated answer based on document context.

---

## Project Structure

```
rag-chatbot/
├── app.py               # Main Streamlit app
├── requirements.txt     # Required Python packages
└── README.md            # Project overview and usage
└── RAG_chatbot.ipynb    # Notebook for the project 
```

---

## Notes

- This tool loads Vicuna-7B using 4-bit quantization for reduced memory usage.
- Ensure your environment supports `bitsandbytes` and CUDA if running on GPU.
- For production deployment, consider quantized/hosted models for better performance.

---

## License

MIT License – feel free to use and modify with attribution.
