# LexiSyncAI: Evidence-Grounded Legal Document Analyst

**Project assignment:** DA3 – Deep Learning & Generative AI  
**Project type:** Academic prototype  
**Document domain:** GDPR and administrative fines  
**Status:** Local application and Docker build run successfully; public deployment is pending.

---

## 1. Executive summary

LexiSyncAI is a legal-document question-answering prototype designed to answer questions from a small collection of public legal PDFs. It combines document parsing, section-aware chunking, hybrid information retrieval, cross-encoder reranking, and locally hosted language generation.

The system is designed to ground its answers in retrieved source passages. It displays document and PDF-page citations and can abstain when the available evidence is insufficient. The project also evaluates retrieval and answer behavior on a held-out question set, including questions that the source documents cannot answer.

The prototype uses the official GDPR Regulation and the European Data Protection Board's Guidelines 04/2022 on the calculation of administrative fines. It is an academic demonstration, not a substitute for legal research or professional advice.

## 2. Problem statement

Legal rules are distributed across long documents, numbered provisions, paragraphs, tables, and explanatory guidance. A useful document analyst must do more than produce a plausible summary. It should:

1. Find passages that answer the question, including exact references such as article and paragraph numbers.
2. Support each answer with traceable document and page citations.
3. Combine evidence across sections or documents when a question requires it.
4. Recognize when the collection does not contain enough evidence and abstain.
5. Make retrieval quality, answer quality, citation behavior, and latency measurable.

LexiSyncAI implements and evaluates these capabilities on a focused GDPR document collection.

## 3. Project objectives

- Ingest public legal PDFs and extract page-level text.
- Detect pages with unusually little extractable text for manual review or OCR fallback.
- Create section-aware chunks that preserve legal structure and page metadata.
- Compare BM25, dense, hybrid, and reranked retrieval.
- Generate answers from retrieved evidence using a local language model.
- Display citations with the source PDF, page number, and section when available.
- Abstain on questions that cannot be supported by the supplied documents.
- Evaluate the system with answerable, multi-hop, and unanswerable questions.
- Demonstrate selected extensions: OCR fallback, table extraction, bounding-box citation, and cross-document evidence use.

## 4. Source documents

The document collection contains two public PDFs:

| Document | Description | PDF pages |
|---|---|---:|
| `gdpr_regulation.pdf` | Regulation (EU) 2016/679, the General Data Protection Regulation | 88 |
| `edpb_fines_guidelines.pdf` | EDPB Guidelines 04/2022 on the calculation of administrative fines under the GDPR | 48 |

The PDFs are public source materials. No confidential client or case documents are included in the example collection.

## 5. System architecture

The application follows a retrieval-augmented generation workflow:

1. **Upload and document inspection** — The user supplies legal PDF files. The application checks the files and extracts page text.
2. **Extraction and metadata capture** — Text is associated with its source document, PDF page, and detected legal section or paragraph.
3. **Section-aware chunking** — Extracted text is divided into passages that aim to preserve article, section, and paragraph context.
4. **Candidate retrieval** — BM25 finds lexical matches, which are useful for exact legal references and terminology. Dense embeddings find passages with related meaning.
5. **Hybrid ranking and reranking** — Candidate lists are combined, and a cross-encoder reranks candidate passages for relevance to the question.
6. **Evidence-grounded generation** — A local Ollama model drafts an answer using the retrieved passages.
7. **Citation and abstention handling** — The interface displays source citations and an evidence-confidence status. When evidence is insufficient, the system returns an abstention instead of a substantive answer.
8. **Evaluation** — Retrieval results, answer decisions, manual answer review, citations, and latency are recorded for analysis.

### Architecture overview

```text
PDFs
  → text extraction and page metadata
  → section-aware chunks
  → BM25 + dense retrieval
  → hybrid merge
  → cross-encoder reranking
  → local LLM answer from retrieved evidence
  → cited answer or abstention
```

## 6. Technical implementation

### Document extraction and chunking

- **PyMuPDF** is used to read PDF pages and extract text in page order.
- A page inventory records the document name, PDF page number, extracted character count, scan-review flag, and text.
- Legal structure is retained through article, section, and paragraph metadata where detected.
- The processed collection contains **723 legal chunks** in the current project run.
- Pages with very little native text are flagged for visual inspection; a low character count is a review signal, not definitive proof that a page is scanned.

### Retrieval

The retrieval experiments compare:

- **BM25:** lexical scoring that emphasizes exact terms and references such as “Article 83(5)”.
- **Dense retrieval:** sentence-transformer embeddings using `all-MiniLM-L6-v2`.
- **Hybrid RRF:** reciprocal-rank fusion of BM25 and dense results.
- **Hybrid plus reranker:** hybrid candidates reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Answer generation

- **Language model:** `qwen3.5:4b`
- **Runtime:** Ollama, running locally during development.
- The prompt provides retrieved passages as the answer context and requests a structured response, citations, and abstention when support is inadequate.
- Model output is not treated as independently authoritative. Users should check the linked source passage.

### Interface and container

- **User interface:** Streamlit application in `app1.py`.
- **Container:** Docker image for local containerized execution.
- The local Docker container was run successfully. Public hosting and a cloud-accessible inference endpoint are not yet configured.

## 7. Evaluation design

The held-out evaluation set contains **15 questions**:

- 12 answerable questions.
- 3 unanswerable questions.
- Questions cover factual lookup, questions requiring evidence from multiple passages, and questions that should trigger abstention.

Expected evidence pages were recorded for answerable questions. Generated answers were reviewed manually, and retrieval rankings were evaluated by page-level Recall@k.

### Page Recall@k results

| Retrieval configuration | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.361 | 0.444 | 0.551 |
| Dense | 0.371 | 0.429 | 0.457 |
| Hybrid RRF | 0.357 | 0.357 | 0.499 |
| Hybrid + cross-encoder reranker | **0.419** | **0.531** | **0.593** |

In this run, hybrid retrieval with reranking produced the highest page Recall@k at all three reported cutoffs. The result is specific to this small document set and evaluation dataset.

### Answer review and abstention results

| Measure | Recorded result |
|---|---:|
| Held-out questions | 15 |
| Answerable questions | 12 |
| Unanswerable questions | 3 |
| Manually correct answers among answerable questions | 5 / 12 |
| Correct abstentions | 3 / 3 |
| Overall correct system decisions | 12 / 15 (80%) |
| Mean response latency | 8.02 seconds |

These results show that abstention worked for the three unanswerable evaluation questions, while answer correctness on answerable questions remains an area for improvement. The 80% decision figure includes both answer correctness and correct abstentions; it should not be interpreted as 80% answer accuracy. Citation precision was not included in the reported summary.

## 8. Additional demonstrations

### OCR fallback

The EDPB PDF's page 11 produced very little native text because it is primarily a diagram. The page was flagged for review and OCR was applied as a fallback. OCR produced 828 characters, recorded in `data/evaluation/ocr_fallback_results.csv`. OCR output was inspected as a fallback demonstration; a formal OCR character-accuracy score against a manually transcribed ground truth was not reported.

### Table extraction

Six tables were detected in the EDPB guidelines PDF during the extraction experiment. Sample outputs were inspected against the visible document. This demonstrates table extraction, while complex merged-cell and spanning-header accuracy has not yet been comprehensively benchmarked.

### Bounding-box citation

A page-region citation demonstration was created and saved as `data/evaluation/bbox_citation_demo.png`. It illustrates how a citation can point to a region on a PDF page. A formal region-level citation precision score was not reported.

### Multi-document cross-referencing

The held-out set includes a question about how the GDPR and EDPB guidance address concurrent infringements. This tests retrieval across the GDPR and the EDPB guidelines, rather than relying on a single source document.

## 9. Repository structure

The project directory is organized approximately as follows:

```text
LexiSyncAI/
├── app1.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── inspect_pdfs.ipynb
├── inspect_pdfs2.ipynb
├── clause_aware_chunking.ipynb
├── bm25_retrieval.ipynb
├── evaluation_dataset.ipynb
└── data/
    ├── raw/
    │   ├── gdpr_regulation.pdf
    │   └── edpb_fines_guidelines.pdf
    ├── processed/
    │   ├── page_inventory.csv
    │   ├── legal_chunks.csv
    │   └── dense_embeddings.npy
    └── evaluation/
        ├── questions.csv
        ├── recall_at_k_results.csv
        ├── answer_evaluation_results.csv
        ├── manual_answer_review.csv
        ├── ocr_fallback_results.csv
        └── bbox_citation_demo.png
```

Some generated files may be excluded from version control to keep the repository compact. The exact files included should be determined by the project's `.gitignore` and deployment configuration.

## 10. Local setup and run instructions

### Prerequisites

- Python 3.10 or newer.
- Docker Desktop for containerized execution.
- Ollama installed and running locally.
- The `qwen3.5:4b` model pulled in Ollama.

### Run with Python

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3.5:4b
streamlit run app1.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Run with Docker

Build the image:

```bash
docker build -t lexisyncai .
```

Run the container while forwarding the Ollama service running on the Mac:

```bash
docker run --rm -p 8502:8501 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  lexisyncai
```

Open `http://localhost:8502`. If port 8502 is already in use, choose another host port and keep the container port as 8501, for example `-p 8503:8501`.

## 11. Limitations and future work

- The source collection is small and focused on GDPR fines; results may not generalize to other legal domains or document layouts.
- Legal language can be nuanced. Retrieval or generation errors may omit exceptions, qualifications, or cross-references.
- Manual answer review is limited to 15 questions, so the reported metrics have substantial uncertainty.
- Only 5 of 12 answerable responses were manually marked correct in the recorded run; improving answer quality is an important next step.
- Citation precision and bounding-box precision need a defined ground-truth protocol and systematic evaluation.
- OCR character accuracy requires a manually transcribed sample for comparison.
- Table extraction should be tested on merged cells and spanning headers against manual ground truth.
- Local Ollama inference is not automatically accessible to a publicly hosted application. Public deployment requires a cloud-accessible model endpoint or hosted inference service.
- Fine-tuning, agent-based verification, figure-only question answering, and a page anomaly autoencoder were not implemented as completed features in this version.

Potential next steps include expanding the evaluation set, improving answer verification, measuring citation precision, strengthening table reconstruction, and deploying the application with a securely configured cloud inference endpoint.

## 12. Responsible use

LexiSyncAI is an educational prototype. It can make retrieval, extraction, citation, and generation mistakes. It does not provide legal advice, determine legal outcomes, or replace review by a qualified lawyer. Verify any important statement against the cited official source.

## 13. Acknowledgements and sources

- European Union, Regulation (EU) 2016/679 (General Data Protection Regulation).
- European Data Protection Board, Guidelines 04/2022 on the calculation of administrative fines under the GDPR.
- Sentence Transformers models: `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Ollama local model runtime.
