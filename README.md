# DriveWise

## Setup
```
pip install -r requirements.txt
```

DriveWise uses local Hugging Face embeddings for retrieval and Groq (free, fast
Llama inference) for grounded answer generation.

Get a free Groq API key at https://console.groq.com/keys (no credit card needed).

Add it locally in a `.env` file:
```
GROQ_API_KEY=your_key_here
```
or in Streamlit Cloud secrets:
```toml
GROQ_API_KEY = "your_key_here"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

## Add brochures
Place PDFs at: `data/brochures/{Brand}/{Model}/name_v1.pdf`
(include `v<number>` in filename for version tracking)

## Ingest
The real Hyundai brochure PDFs use embedded fonts whose bullet glyphs extract as
raw `/uniXXXX` escapes and odd unicode spacing in spec tables - `ingest.py` cleans
this (see `config.clean_text`) before chunking, so both the vector index and the
LLM context are readable.

Ingest a single model:
```
python ingest.py "Hyundai" "i20"
```

Ingest everything already in `data/brochures/` (all 13 models):
```
python ingest.py
```

Builds a FAISS index per brand/model at `indexes/Brand_Model/`.
**Run this once after pulling the project (indexes are not shipped) and again any
time you add/replace a brochure PDF.**

## Run app
```
streamlit run app.py
```

## Architecture
- `config.py` - brands/models, section keywords (incl. EV terms like battery/charging for Ioniq 5 & Creta EV), PDF text cleanup, Groq/embedding setup
- `ingest.py` - cleans PDF text, splits into section-classified chunks, builds a FAISS index per model
- `retriever.py` - loads the selected brand/model index, filters/re-ranks matching chunks
- `rag_chain.py` - builds the controlled context window and calls Groq for grounded answer generation, with source details
- `evaluation.py` - faithfulness, context relevance, answer correctness
- `logger.py` - JSONL logs: query, response time, retrieval, status/errors
- `app.py` - Streamlit UI
