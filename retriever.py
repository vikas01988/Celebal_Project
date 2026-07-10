import os
import re
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from config import BRANDS, DEFAULT_SECTION, INDEX_DIR, SECTION_KEYWORDS, TOP_K_RETRIEVE, TOP_N_FINAL, get_embeddings
embeddings = get_embeddings()
ALIASES = {
    "milage": "mileage",
    "break": "brake",
    "breaks": "brakes",
    "a/c": "ac air conditioning",
    "touch screen": "touchscreen",}
EXTRA_WORDS = {
    "mileage": "mileage fuel tank fuel type fuel consumption fuel efficiency kmpl km/l range distance to empty battery range charging time fast charging dc charging ac charging home charging",
    "dimensions": "dimensions overall length overall width overall height wheelbase ground clearance boot space",
    "engine": "engine displacement max power max torque transmission manual automatic amt dct ivt electric motor motor type battery pack battery capacity kwh rwd fwd awd drivetrain",
    "suspension": "suspension front rear mcpherson strut torsion beam coil spring",
    "brakes_tyres": "brakes front rear disc drum tyre tire wheel alloy steel",}
SEDAN_MODELS = {"Aura", "Verna"}
@lru_cache(maxsize=20)
def load_index(brand, model):
    path = os.path.join(INDEX_DIR, f"{brand}_{model}".replace(" ", "_"))
    if not os.path.isdir(path):
        return None
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def clean_q(q):
    q = q.lower()
    for a, b in ALIASES.items():
        q = re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, q)
    return q
def has(text, word):
    text, word = clean_q(text), word.lower()
    if " " in word or "/" in word:
        return word in text
    return re.search(rf"\b{re.escape(word)}s?\b", text) is not None

def normalize_query(q):
    return clean_q(q)
def classify_query(q):
    q = clean_q(q)
    for name, words in SECTION_KEYWORDS.items():
        if any(has(q, w) for w in words):
            return name
    return DEFAULT_SECTION
def query_sections(q):
    q = clean_q(q)
    found = [name for name, words in SECTION_KEYWORDS.items() if any(has(q, w) for w in words)]
    return found or [DEFAULT_SECTION]
def is_comparison_query(q):
    q = q.lower()
    return any(x in q for x in ["compare", "comparison", " vs ", " versus ", "better", "which one", "among", "between"])
def wants_all_models(q):
    q = q.lower()
    return any(x in q for x in ["all cars", "all models", "every car", "every model", "among all"])
def mentioned_models(brand, model, q):
    q = q.lower()
    if has(q, "sedan"):
        return [name for name in BRANDS.get(brand, []) if name in SEDAN_MODELS]
    if wants_all_models(q):
        return BRANDS.get(brand, [model])
    found = [model]
    for name in sorted(BRANDS.get(brand, []), key=len, reverse=True):
        if name != model and re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", q):
            found.append(name)
    return found
def search_words(q, section):
    return q if section == DEFAULT_SECTION else f"{q} {EXTRA_WORDS.get(section, ' '.join(SECTION_KEYWORDS[section]))}"
def good_doc(doc, section):
    if section == DEFAULT_SECTION:
        return True
    text = doc.page_content.lower()
    words = EXTRA_WORDS.get(section, " ".join(SECTION_KEYWORDS[section])).split()
    return doc.metadata.get("section") == section or any(w in text for w in words)

def rank(rows, q, brand, model, section):
    q_words = set(re.findall(r"[a-z0-9]+", clean_q(q)))
    need = EXTRA_WORDS.get(section, " ".join(SECTION_KEYWORDS.get(section, []))).split()
    scored = []
    for doc, score in rows:
        text = doc.page_content.lower()
        meta = doc.metadata
        points = len(q_words & set(re.findall(r"[a-z0-9]+", text)))
        points += 3 * sum(w in text for w in need)
        points += 4 if meta.get("model") == model else 0
        points += 3 if meta.get("brand") == brand else 0
        points += 2 if meta.get("section") == section else 0
        points += 1 / (1 + max(float(score), 0))
        scored.append((points, doc, score))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(doc, score) for points, doc, score in scored]


def retrieve(brand, model, q, k=TOP_K_RETRIEVE, top_n=TOP_N_FINAL, section_override=None):
    index = load_index(brand, model)
    if not index:
        return []
    section = section_override or classify_query(q)
    top_n = min(top_n, 4) if section != DEFAULT_SECTION else top_n
    rows = index.similarity_search_with_score(search_words(q, section), k=k)
    rows = [(doc, score) for doc, score in rows if good_doc(doc, section)]
    rows = rank(rows, q, brand, model, section)[:top_n]
    return [{"text": d.page_content, "metadata": d.metadata, "vector_score": float(s)} for d, s in rows]
def dedupe(chunks):
    seen, out = set(), []
    for c in chunks:
        m = c["metadata"]
        key = (m.get("model"), m.get("page"), " ".join(c["text"].split())[:100])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
def retrieve_for_question(brand, model, q):
    models = mentioned_models(brand, model, q)
    sections = query_sections(q)
    compare_many = is_comparison_query(q) or wants_all_models(q) or has(q, "sedan")
    if not compare_many or len(models) == 1:
        return retrieve(brand, model, q)
    chunks = []
    if sections == [DEFAULT_SECTION]:
        sections = ["safety", "comfort", "engine", "mileage", "dimensions"]
    for car in models:
        for sec in sections:
            chunks += retrieve(brand, car, search_words(q, sec), top_n=2, section_override=sec)
    return dedupe(chunks)
