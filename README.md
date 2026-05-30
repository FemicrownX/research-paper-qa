# Research Paper Question Answering System

Upload a research paper and ask questions about it. That is the whole idea.

You drop in a PDF, the system breaks it into chunks, embeds them into a vector database, and when you ask a question it finds the most relevant parts and sends them to a local LLM to generate an answer. Everything runs on your machine — no API keys, no cloud calls.

## Stack

- Gradio — the interface
- PyMuPDF — reads the PDF
- sentence-transformers (all-MiniLM-L6-v2) — turns text into vectors
- ChromaDB — stores and searches the vectors
- Llama 3.2 via Ollama — generates the answers

## Project Structure

```
research-paper-qa/
│
├── app.py           # interface and main logic
├── ingest.py        # loads and chunks the PDF
├── retriever.py     # ChromaDB and similarity search
├── llm.py           # Ollama integration
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

You need Python 3.10+ and [Ollama](https://ollama.com) installed. Pull the model first:

```bash
ollama pull llama3.2
```

Then clone and install:

```bash
git clone https://github.com/FemicrownX/research-paper-qa.git
cd research-paper-qa

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Running

Make sure Ollama is running, then:

```bash
python app.py
```

Go to `http://127.0.0.1:7860` in your browser, upload a PDF, click Process PDF, and start asking questions.

## Good questions to ask

- What is this paper about?
- What problem does it solve?
- What datasets were used?
- What were the main results?
- What limitations do the authors mention?

## How retrieval works

The system uses similarity search with k=10. ChromaDB scans the full collection by default so no fetch_k is needed — every chunk is a candidate before the top 10 are selected.

## Notes

- The app answers only from what is in the uploaded document. If something is not there, it says so.
- PDF files are excluded from the repo via `.gitignore` — bring your own papers.
- To switch to a cloud model like Gemini, update `llm.py` and add your key to `.env`. Nothing else changes.

## What I plan to add

- Support for multiple PDFs at once
- Show which part of the document the answer came from
- Chat history so follow-up questions work naturally
- A shareable public link option

## Libraries

- [Gradio](https://gradio.app/)
- [LangChain](https://langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [sentence-transformers](https://www.sbert.net/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Ollama](https://ollama.com/)