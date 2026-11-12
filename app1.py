import hashlib
import json
import os
import re
import time
import urllib.request

import fitz
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# -------------------- Configuration --------------------

st.set_page_config(
    page_title="LexiSyncAI",
    layout="wide",
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_config_value(name, default=""):
    """Read a setting from environment variables or Streamlit Secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


OLLAMA_BASE_URL = get_config_value("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = get_config_value("OLLAMA_API_KEY", "")
OLLAMA_IS_CLOUD = OLLAMA_BASE_URL.startswith("https://ollama.com")
OLLAMA_MODEL = get_config_value(
    "OLLAMA_MODEL",
    "gemma4:31b" if OLLAMA_IS_CLOUD else "qwen3.5:4b",
)
OLLAMA_API_ROOT = OLLAMA_BASE_URL if OLLAMA_BASE_URL.endswith("/api") else f"{OLLAMA_BASE_URL}/api"


# -------------------- Dark theme --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #111827, #1e293b);
        color: #f8fafc;
        padding: 26px 32px;
        border: 1px solid #334155;
        border-radius: 14px;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 650;
    }

    .hero p {
        color: #cbd5e1;
        margin: 8px 0 0 0;
        font-size: 1rem;
    }

    [data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
        font-size: 1rem;
        line-height: 1.55;
    }

    [data-testid="stWidgetLabel"] p {
        color: #e5e7eb;
        font-size: 1rem;
    }

    [data-testid="stTextArea"] textarea {
        background: #111827;
        color: #f8fafc;
        font-size: 1rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #111827;
        border: 1px dashed #475569;
    }

    [data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    div.stButton > button {
        font-size: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <div class="hero">
      <h1>LexiSyncAI</h1>
      <p>Evidence-grounded analysis for legal and compliance documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Upload 2–5 public legal PDFs, enter a question, and select Analyze. "
    "The app prepares the document index automatically."
)


# -------------------- Load retrieval models --------------------

@st.cache_resource(show_spinner="Loading retrieval models…")
def load_models():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return embedding_model, reranker


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


# -------------------- PDF extraction and chunking --------------------

def is_heading(line):
    patterns = [
        r"^Article\s+\d+[A-Za-z]?",
        r"^CHAPTER\s+[IVXLC0-9]+",
        r"^\d+(?:\.\d+){0,3}\s*[-–]\s+\S+",
    ]

    return any(
        re.match(pattern, line.strip(), flags=re.IGNORECASE)
        for pattern in patterns
    )


def split_long_text(text, max_chars=1800):
    """Split long text while retaining page and section metadata."""
    pieces = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)

        if split_at < max_chars // 2:
            split_at = max_chars

        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def extract_page_chunks(document_name, page_number, page_text):
    lines = [line.strip() for line in page_text.splitlines()]
    sections = []
    current_section = "Page text"
    current_lines = []

    for line in lines:
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_heading(line):
            if current_lines:
                sections.append(
                    (current_section, "\n".join(current_lines).strip())
                )

            current_section = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            (current_section, "\n".join(current_lines).strip())
        )

    rows = []
    paragraph_number = 0

    for section, section_text in sections:
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n", section_text)
            if part.strip()
        ]

        for paragraph in paragraphs:
            for passage in split_long_text(paragraph):
                paragraph_number += 1
                rows.append({
                    "chunk_id": (
                        f"{document_name}::p{page_number}::c{paragraph_number}"
                    ),
                    "document": document_name,
                    "page_number": page_number,
                    "section": section,
                    "paragraph": paragraph_number,
                    "text": passage,
                })

    if not rows and page_text.strip():
        rows.append({
            "chunk_id": f"{document_name}::p{page_number}::c1",
            "document": document_name,
            "page_number": page_number,
            "section": "Unclassified section",
            "paragraph": 1,
            "text": page_text.strip()[:1800],
        })

    return rows


def build_index(uploaded_files):
    all_chunks = []
    page_count = 0
    low_text_pages = []

    for uploaded_file in uploaded_files:
        document_name = uploaded_file.name
        pdf_bytes = uploaded_file.getvalue()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            page_count += len(pdf)

            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                page_text = page.get_text("text").strip()

                if len(page_text) < 30:
                    low_text_pages.append(
                        f"{document_name}, PDF page {page_number}"
                    )

                all_chunks.extend(
                    extract_page_chunks(
                        document_name,
                        page_number,
                        page_text,
                    )
                )

    if not all_chunks:
        raise ValueError(
            "No selectable text was extracted. The PDFs may be scanned."
        )

    chunks = pd.DataFrame(all_chunks)
    embedding_model, reranker = load_models()

    embeddings = embedding_model.encode(
        chunks["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    bm25 = BM25Okapi(
        [tokenize(text) for text in chunks["text"]]
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "bm25": bm25,
        "embedding_model": embedding_model,
        "reranker": reranker,
        "documents": [file.name for file in uploaded_files],
        "page_count": page_count,
        "low_text_pages": low_text_pages,
    }


# -------------------- Hybrid retrieval and reranking --------------------

def retrieve_evidence(question, index_data, top_k=3):
    chunks = index_data["chunks"]

    bm25_order = np.argsort(
        index_data["bm25"].get_scores(tokenize(question))
    )[::-1]

    query_embedding = index_data["embedding_model"].encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dense_order = np.argsort(
        index_data["embeddings"] @ query_embedding
    )[::-1]

    # Reciprocal Rank Fusion merges BM25 and dense rankings.
    fusion_scores = {}

    for ranking in (bm25_order, dense_order):
        for rank, chunk_index in enumerate(ranking, start=1):
            chunk_index = int(chunk_index)
            fusion_scores[chunk_index] = (
                fusion_scores.get(chunk_index, 0.0) + 1 / (60 + rank)
            )

    candidate_indices = sorted(
        fusion_scores,
        key=fusion_scores.get,
        reverse=True,
    )[:15]

    candidates = chunks.iloc[candidate_indices].copy()

    pairs = [
        [question, str(row["text"])[:1600]]
        for _, row in candidates.iterrows()
    ]

    candidates["reranker_score"] = index_data["reranker"].predict(pairs)

    candidates = candidates.sort_values(
        "reranker_score",
        ascending=False,
    ).head(top_k)

    evidence = []

    for number, (_, row) in enumerate(candidates.iterrows(), start=1):
        evidence.append({
            "evidence_id": f"E{number}",
            "document": str(row["document"]),
            "page_number": int(row["page_number"]),
            "section": str(row["section"]),
            "text": str(row["text"]),
            "reranker_score": float(row["reranker_score"]),
        })

    return evidence


# -------------------- Answer generation and citation validation --------------------

def generate_answer(question, evidence):
    if not evidence:
        return "The documents do not contain enough evidence to answer this."

    context = "\n\n".join(
        f"[{item['evidence_id']}] {item['document']}, "
        f"PDF page {item['page_number']}, section {item['section']}:\n"
        f"{item['text'][:1200]}"
        for item in evidence
    )

    prompt = f"""
Answer the question using only the evidence below.
Be concise. Cite factual claims with evidence IDs such as [E1].
Do not invent facts, sections, or page numbers.
If the evidence is insufficient, reply exactly:
The documents do not contain enough evidence to answer this.

Question:
{question}

Evidence:
{context}

Answer:
"""

    options = {
        "temperature": 0,
        "num_predict": 180,
    }

    if OLLAMA_IS_CLOUD:
        if not OLLAMA_API_KEY:
            raise RuntimeError(
                "OLLAMA_API_KEY is missing. Add it to Streamlit Secrets."
            )
        endpoint = f"{OLLAMA_API_ROOT}/chat"
        request_payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
        }
    else:
        endpoint = f"{OLLAMA_API_ROOT}/generate"
        request_payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": options,
        }
        headers = {"Content-Type": "application/json"}

    request_body = json.dumps(request_payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        if OLLAMA_IS_CLOUD:
            message = (
                f"Cloud Ollama request failed: {error}. "
                "Check the model name and OLLAMA_API_KEY."
            )
        else:
            message = (
                f"Local Ollama request failed: {error}. "
                f"Check that Ollama is running and {OLLAMA_MODEL} is available."
            )
        raise RuntimeError(message) from error

    if OLLAMA_IS_CLOUD:
        answer = result.get("message", {}).get("content", "").strip()
    else:
        answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError("The language model returned an empty answer.")

    if "do not contain enough evidence" in answer.lower():
        return answer

    valid_ids = {item["evidence_id"] for item in evidence}
    cited_ids = set(re.findall(r"\[(E\d+)\]", answer))

    if not cited_ids or not cited_ids.issubset(valid_ids):
        return (
            "The documents do not contain enough verified evidence to answer this."
        )

    return answer


# -------------------- Upload and analyze interface --------------------

with st.sidebar:
    st.header("Document bundle")
    st.write("Upload 2–5 public PDF documents.")
    st.caption(f"Answer model: {OLLAMA_MODEL}")
    st.caption("Low-text pages are flagged; OCR is not run in this version.")

uploaded_files = st.file_uploader(
    "Choose legal PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

question = st.text_area(
    "What would you like to know?",
    placeholder="Example: What are the two main GDPR administrative fine tiers?",
    height=100,
)

analyze_clicked = st.button(
    "Analyze documents",
    type="primary",
    use_container_width=True,
)

if analyze_clicked:
    if not uploaded_files or not 2 <= len(uploaded_files) <= 5:
        st.error("Upload between 2 and 5 PDF files.")
    elif not question.strip():
        st.error("Enter a question first.")
    else:
        # Reuse the index until the uploaded PDFs change.
        signature_hash = hashlib.sha256()

        for uploaded_file in uploaded_files:
            signature_hash.update(uploaded_file.name.encode("utf-8"))
            signature_hash.update(uploaded_file.getvalue())

        current_signature = signature_hash.hexdigest()

        try:
            if st.session_state.get("file_signature") != current_signature:
                with st.spinner("Preparing PDFs and building the index…"):
                    st.session_state.index_data = build_index(uploaded_files)
                    st.session_state.file_signature = current_signature

            index_data = st.session_state.index_data

            with st.spinner("Retrieving evidence and generating the answer…"):
                start_time = time.time()
                evidence = retrieve_evidence(question, index_data)
                answer = generate_answer(question, evidence)
                elapsed = time.time() - start_time

            if "do not contain enough evidence" in answer.lower():
                confidence = "INSUFFICIENT EVIDENCE"
                badge_color = "#8a3c3c"
            elif evidence and evidence[0]["reranker_score"] >= 5:
                confidence = "HIGH"
                badge_color = "#24734a"
            elif evidence and evidence[0]["reranker_score"] >= 2:
                confidence = "MEDIUM"
                badge_color = "#946200"
            else:
                confidence = "LOW"
                badge_color = "#8a3c3c"

            st.markdown("### Answer")
            st.markdown(
                f'<span style="background:{badge_color};color:white;'
                f'padding:6px 11px;border-radius:14px;font-weight:700">'
                f'{confidence}</span> '
                f'<span style="color:#64748b">Response time: '
                f'{elapsed:.1f} seconds</span>',
                unsafe_allow_html=True,
            )
            st.markdown(answer)

            if evidence:
                st.markdown("### Cited evidence")

                for item in evidence:
                    title = (
                        f"{item['evidence_id']} · {item['document']} · "
                        f"PDF p. {item['page_number']} · {item['section']}"
                    )
                    with st.expander(title):
                        st.markdown(item["text"])

            if index_data["low_text_pages"]:
                st.warning(
                    "Low-text pages flagged for review: "
                    + "; ".join(index_data["low_text_pages"])
                )

            st.caption(
                "Confidence is a heuristic indicator, not a calibrated probability."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
