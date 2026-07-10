import numpy as np
from config import get_embeddings
embeddings = get_embeddings()
def similarity(a, b):
    va, vb = [np.array(x) for x in embeddings.embed_documents([a, b])]
    return round(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)), 4)
def evaluate(query, answer, chunks, reference=None):
    context = " ".join(c["text"] for c in chunks)
    return {
        "faithfulness": similarity(answer, context) if answer and context else 0.0,
        "context_relevance": similarity(query, context) if query and context else 0.0,
        "answer_correctness": similarity(answer, reference) if reference else None,
    }
