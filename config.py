import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BROCHURE_DIR = os.path.join(BASE_DIR, "data", "brochures")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
LOG_FILE = os.path.join(BASE_DIR, "logs", "drivewise.log")
BRANDS = {"Hyundai": [
    "Alcazar", "Aura", "Creta", "Creta EV", "Creta N Line", "Exter",
    "Grand i10 Nios", "i20", "i20 N Line", "Ioniq 5", "Venue",
    "Venue N Line", "Verna",]}
SECTION_KEYWORDS = {
    "engine": "engine engines performance horsepower power torque displacement cc bhp ps transmission gearbox manual automatic amt dct ivt cylinder".split()
    + ["electric motor", "motor type", "pmsm", "battery pack", "battery type", "kwh", "rwd", "fwd", "awd", "drivetrain"],
    "mileage": "mileage milage kmpl km/l kpl range fuel tank capacity type efficiency economy consumption average instantaneous".split()
    + ["charging time", "fast charging", "dc charging", "ac charging", "home charging", "wallbox"],
    "safety": "airbag airbags abs safety ncap esc isofix crash ebd adas smartsense tpms camera".split(),
    "dimensions": "length width height wheelbase dimensions dimension boot cargo luggage".split() + ["ground clearance", "boot space", "turning radius"],
    "suspension": "suspension front rear mcpherson strut torsion beam coil spring shock absorber".split(),
    "brakes_tyres": "brake brakes braking disc drum tyre tyres tire tires wheel wheels alloy alloys steel".split() + ["wheel size", "front brake", "rear brake"],
    "comfort": "seat seats seating upholstery ac comfort recline legroom armrest sunroof".split() + ["a/c", "air conditioner", "air conditioning", "climate control", "rear ac", "wireless charger", "wireless charging"],
    "infotainment": "touchscreen infotainment bluetooth carplay speaker speakers music audio bose display screen connectivity bluelink".split() + ["touch screen", "android auto", "apple carplay", "connected car"],}
DEFAULT_SECTION = "general"
TOP_K_RETRIEVE = 20
TOP_N_FINAL = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def clean_text(text):
    """Clean raw PyPDFLoader text: real brochure PDFs use embedded fonts whose
    bullet/marker glyphs extract as literal '/uniXXXX' escapes, plus non-breaking/
    thin unicode spaces in spec tables (e.g. '4 330' mm) and footnote markers
    glued to numbers (e.g. '1 635^^^', '1 625***')."""
    if not text:
        return ""
    text = re.sub(r"/uni([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)
    for ch in ["\xa0", "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
               "\u2006", "\u2007", "\u2008", "\u2009", "\u200a", "\u202f", "\u3000"]:
        text = text.replace(ch, " ")
    text = re.sub(r"(?<=[0-9A-Za-z\)])[\^\*#~]{1,4}(?=[\s.,;:]|$)", "", text)
    return " ".join(text.split())
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def _get_secret(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None

GROQ_API_KEY = _get_secret("GROQ_API_KEY")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

def get_llm():
    api_key = GROQ_API_KEY or _get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file (GROQ_API_KEY=...) "
            "or to Streamlit Cloud secrets. Get a free key at https://console.groq.com/keys"
        )
    model = _get_secret("GROQ_MODEL") or GROQ_MODEL or "llama-3.3-70b-versatile"
    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0.2,
    )
