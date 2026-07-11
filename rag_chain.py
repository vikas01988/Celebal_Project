from langchain_core.messages import HumanMessage, SystemMessage
from config import SECTION_KEYWORDS, clean_text, get_llm
from retriever import query_sections
def car_name(chunk):
    m = chunk["metadata"]
    return f"{m.get('brand', '')} {m.get('model', '')}".strip()
def source_section(chunk, q):
    text = chunk["text"].lower()
    for sec in query_sections(q):
        if any(w.lower() in text for w in SECTION_KEYWORDS.get(sec, [])):
            return sec
    return chunk["metadata"].get("section", "general")
def make_sources(chunks, q):
    """Source attribution: brochure name, section, page, version, chunk reference."""
    out, seen = [], set()
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        s = {
            "brochure": m["source"],
            "section": source_section(c, q),
            "page": m["page"],
            "version": m["version"],
            "chunk_ref": f"C{i}",
        }
        key = (s["brochure"], s["section"], s["page"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
def build_context(chunks):
    """Controlled context window: only the retrieved, re-ranked chunks, tagged for citation."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        header = f"[C{i}] {car_name(c)} | section: {m.get('section', 'general')} | {m['source']} p.{m['page']} v{m['version']}"
        blocks.append(f"{header}\n{clean_text(c['text'])}")
    return "\n\n".join(blocks)
SYSTEM_PROMPT = (
    "You are DriveWise, a brochure-grounded car assistant for the {brand} {model}.\n"
    "Answer the user's question using ONLY the brochure context chunks given below, "
    "each tagged like [C1], [C2]. Never use outside knowledge and never invent a "
    "specification that is not present in the context.\n\n"
    "Rules:\n"
    "- If the context does not contain the answer, say so plainly instead of guessing.\n"
    "- Write concise, well-formatted markdown: use bullet points for feature/spec lists.\n"
    "- If the context includes more than one car model, build a markdown comparison "
    "table of the relevant specs, then give a short, clearly-reasoned recommendation "
    "grounded only in the compared facts.\n"
    "- Refer to facts plainly; do not print the [C#] tags in your final answer, the "
    "app shows sources separately.\n"
    "- Keep the answer focused, factual, and free of filler."
)
def generate_answer(brand, model, question, chunks):
    """Generation layer: language-model response grounded in retrieved, re-ranked brochure chunks."""
    if not chunks:
        return "No relevant brochure content found for this vehicle and question.", []
    context = build_context(chunks)
    system = SYSTEM_PROMPT.format(brand=brand, model=model)
    human = f"Brochure context:\n{context}\n\nQuestion: {question}"
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    answer = response.content.strip()
    
    if isinstance(response.content, list):
        answer = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content)
    else:
        answer = response.content
    answer = answer.strip()
    return answer, make_sources(chunks, question)