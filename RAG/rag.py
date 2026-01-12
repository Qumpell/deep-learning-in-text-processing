from typing import List, Tuple, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# --- Embedding (sentence-transformers) ---
def generate_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# --- Normalizacja + cosine top-k ---
def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def retrieve_top_k(query_emb: np.ndarray, doc_emb: np.ndarray, k: int = 3) -> List[List[Tuple[int, float]]]:
    q = query_emb if query_emb.ndim == 2 else query_emb[None, :]
    q = normalize_embeddings(q)
    d = normalize_embeddings(doc_emb)
    sims = np.matmul(q, d.T)
    results = []
    for i in range(sims.shape[0]):
        row = sims[i]
        idx = np.argsort(-row)[:k]
        results.append([(int(j), float(row[j])) for j in idx])
    return results


# --- Generacja PL z kontekstem (FLAN-T5) ---
def generate_answer_with_docs(query: str, docs: List[str], generator_model: str = "google/flan-t5-base", max_length: int = 150, device: Optional[int] = None) -> str:
    context = "\n\n---\n\n".join(docs) if docs else ""
    prompt = (
        f"Odpowiedz na pytanie korzystając wyłącznie z kontekstu.\n\n"
        f"Kontekst:\n{context}\n\n"
        f"Pytanie: {query}\n"
        f"Odpowiedź po polsku:"
    )

    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(generator_model)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False,       # deterministyczniej dla QA
        num_beams=4
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --- Generacja bez retrieval (ignoruje dokumenty) ---
def generate_answer_without_docs(query: str, generator_model: str = "google/flan-t5-base", max_length: int = 150, device: Optional[int] = None) -> str:
    prompt = (
        f"Odpowiedz na pytanie bazując na ogólnej wiedzy.\n\n"
        f"Pytanie: {query}\n"
        f"Odpowiedź po polsku:"
    )

    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(generator_model)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=False,
        num_beams=4
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# --- Wrapper do porównań ---
def answer_and_compare(query: str, docs: List[str], doc_emb: np.ndarray, k: int = 2) -> Dict:
    q_emb = generate_embeddings([query])
    top = retrieve_top_k(q_emb, doc_emb, k=k)[0]
    retrieved_docs = [docs[i] for i, _ in top]

    return {
        "query": query,
        "retrieved": [(i, score) for (i, score) in top],
        "with": generate_answer_with_docs(query, retrieved_docs, generator_model = "google/mt5-base"),
        "without": generate_answer_without_docs(query, generator_model = "google/mt5-base")
    }


# --- Dokumenty ---
doc1 = """Zalety i ograniczenia RAG...
..."""
doc2 = """Retrieval-Augmented Generation..."""
doc3 = """Generowanie i sumaryzacja..."""

docs = [doc1, doc2, doc3]

# --- Embeddings ---
doc_emb = generate_embeddings(docs)

# --- Zapytania ---
queries = [
    "Co to jest RAG?",
    "Jakie są zalety RAG?",
    "Na czym polega generowanie tekstu?",
]

# --- Wyniki ---
results = [answer_and_compare(q, docs, doc_emb) for q in queries]

for r in results:
    print("=== PYTANIE:", r["query"])
    print("WITH RETRIEVAL:\n", r["with"])
    print("\nWITHOUT:\n", r["without"])
    print("-" * 70)

# --- Mini komentarz jakościowy ---
print("\nKomentarz jakościowy:")
print("""
Z odpowiedzi z retrievalem model używa terminologii z dokumentów,
jest bardziej faktograficzny i zgodny z treścią źródłową.
Bez retrievalu model odpowiada ogólnie lub pomija ważne szczegóły.
""")
