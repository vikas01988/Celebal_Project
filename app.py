import time
import streamlit as st
import retriever
from config import BRANDS, GOOGLE_API_KEY
from evaluation import evaluate
from logger import log_event
from rag_chain import generate_answer
VERSION = "simple-v1"
def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- **{s['brochure']}** - {s['section']} - p.{s['page']} - v{s['version']} - {s.get('chunk_ref', '')}")
def show_error(error):
    if error:
        with st.expander("Error details"):
            st.code(error)
st.set_page_config(page_title="DriveWise", page_icon="car", layout="centered")
st.title("DriveWise - Brochure-Grounded Car Assistant")
brand = st.selectbox("Brand", ["Hyundai"], disabled=True)
model = st.selectbox("Model", BRANDS[brand])
if st.session_state.get("version") != VERSION:
    st.session_state.history = []
    st.session_state.version = VERSION
for msg in st.session_state.get("history", []):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        show_sources(msg.get("sources"))
        show_error(msg.get("error"))
q = st.chat_input(f"Ask about the Hyundai {model}...")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    start = time.time()
    status, error = "success", None
    try:
        chunks = retriever.retrieve_for_question(brand, model, q)
        answer, sources = generate_answer(brand, model, q, chunks)
        status = "success" if chunks else "no_context"
    except Exception as e:
        chunks, sources = [], []
        answer, status, error = "Something went wrong answering that.", "failed", str(e)

    sec = round(time.time() - start, 2)
    with st.chat_message("assistant"):
        st.write(answer)
        show_sources(sources)
        show_error(error)
        if chunks:
            m = evaluate(q, answer, chunks)
            st.caption(f"{sec}s | faithfulness: {m['faithfulness']} | context relevance: {m['context_relevance']}")
    log_event(brand=brand, model=model, query=q, response_time=sec, retrieved=[c["metadata"] for c in chunks], status=status, error=error)
    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources, "error": error})
import streamlit as st

st.write("API Key Loaded:", GOOGLE_API_KEY is not None)