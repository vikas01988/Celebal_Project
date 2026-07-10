import glob
import os
import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    BRANDS,
    BROCHURE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SECTION,
    INDEX_DIR,
    SECTION_KEYWORDS,
    clean_text,
    get_embeddings,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings = get_embeddings()


def classify_section(text: str) -> str:
    """Metadata-based section tagging using the logical brochure sections defined
    in config.SECTION_KEYWORDS (engine/performance, mileage/efficiency, safety,
    dimensions, interior/comfort, infotainment, etc.)."""
    clean = text.lower()
    scores = {
        section: sum(clean.count(keyword) for keyword in keywords)
        for section, keywords in SECTION_KEYWORDS.items()
    }
    best_section = max(scores, key=scores.get)
    return best_section if scores[best_section] > 0 else DEFAULT_SECTION


def extract_version(filename: str) -> str:
    """Document version metadata, read from the filename (e.g. creta_v2.pdf).
    Falls back to '1.0' when the brochure filename carries no explicit version."""
    match = re.search(r"v(\d+(\.\d+)?)", filename, re.IGNORECASE)
    return match.group(1) if match else "1.0"


def ingest_brochures(brand: str, model: str):
    folder = os.path.join(BROCHURE_DIR, brand, model)
    pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No brochures found in {folder}")
    documents = []
    for pdf_path in pdf_paths:
        version = extract_version(os.path.basename(pdf_path))
        pages = PyPDFLoader(pdf_path).load()
        for page in pages:
            # Real Hyundai brochure PDFs use embedded fonts whose bullet glyphs
            # extract as raw '/uniXXXX' escapes, plus non-breaking spaces inside
            # spec tables - clean before chunking so both the embeddings and the
            # LLM context are readable.
            page_text = clean_text(page.page_content)
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                if len(chunk.strip()) < 30:
                    continue
                section = classify_section(chunk)
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "brand": brand,
                            "model": model,
                            "section": section,
                            "page": page.metadata.get("page", 0) + 1,
                            "version": version,
                            "source": os.path.basename(pdf_path),
                        },
                    )
                )
    if not documents:
        raise ValueError("No usable text extracted from brochures.")
    vector_store = FAISS.from_documents(documents, embeddings)
    out_dir = os.path.join(INDEX_DIR, f"{brand}_{model}".replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    vector_store.save_local(out_dir)
    return len(documents), out_dir


def ingest_all():
    results = []
    for brand, models in BRANDS.items():
        for model in models:
            try:
                chunk_count, index_path = ingest_brochures(brand, model)
                results.append((brand, model, chunk_count, index_path, None))
            except Exception as e:
                results.append((brand, model, 0, None, str(e)))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        brand, model = sys.argv[1], sys.argv[2]
        chunk_count, index_path = ingest_brochures(brand, model)
        print(f"Indexed {chunk_count} chunks -> {index_path}")
    else:
        print("No brand/model given - ingesting every brochure listed in config.BRANDS...")
        for brand, model, chunk_count, index_path, error in ingest_all():
            if error:
                print(f"FAILED {brand} {model}: {error}")
            else:
                print(f"Indexed {chunk_count} chunks -> {index_path}")
