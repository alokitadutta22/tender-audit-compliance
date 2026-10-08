# ⚡ Argus Bid AI — Automated Tender Audit & Compliance Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> **An enterprise-grade, deterministic AI-assisted procurement audit platform.** Automatically cross-examines hundreds of pages of vendor bid submissions against complex Notice Inviting Tender (NIT) requirements, delivering verifiable compliance matrices with page-level citations and zero hallucination risk.

---

## 📌 Table of Contents
- [The Problem & The Solution](#-the-problem--the-solution)
- [System Architecture](#-system-architecture)
- [End-to-End Pipelines & Flowcharts](#-end-to-end-pipelines--flowcharts)
  - [1. Document Ingestion & Multi-Stage Waterfall OCR](#1-document-ingestion--multi-stage-waterfall-ocr)
  - [2. Hybrid Retrieval & Advanced RAG Flow](#2-hybrid-retrieval--advanced-rag-flow)
  - [3. Deterministic Gate & Compliance Decision Matrix](#3-deterministic-gate--compliance-decision-matrix)
- [Core Technical Innovations](#-core-technical-innovations)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Evaluation & Audit Matrix](#-evaluation--audit-matrix)

---

## 🎯 The Problem & The Solution

### 🚨 The Problem
Public procurement and commercial tender auditing require verifying multiple vendor bid packages—each often spanning 100 to 500+ pages of PDFs—against strict Notice Inviting Tender (NIT) clauses. 
- **Time Bottleneck:** Evaluation officers spend days cross-referencing PAN cards, GST certificates, audited balance sheets, technical datasheets, and Manufacturer's Authorization Forms (MAFs).
- **Human Error & Bias:** High-volume reading leads to missed non-compliances, missed deviations, or overlooked eligibility thresholds.
- **Why Pure Generative AI Fails:** Raw LLMs hallucinate numbers, fail at strict numeric boundaries (e.g. ₹5.0 Cr turnover vs ₹4.9 Cr), and lack legally defensible audit citations.

### 💡 The Solution
**Argus Bid AI** combines **deterministic logic** for legal checks (regex, fixed boundary comparisons, document isolation) with **Advanced Hybrid RAG** for contextual analysis. Every decision is traceable down to the exact document, section, and page number.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MASTER TENDER (NIT)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                     [ Dynamic NIT Requirement Extractor ]
             ├── Pre-Qualification Criteria (PQC: Exp, Turnover)
             ├── Mandatory Documents Checklist (PAN, GST, MAF, etc.)
             └── Technical Specifications (Mandatory & Preferred)
                                       │
                                       ▼
                       [ AUDIT RULES ENGINE MATRIX ]
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           ▼                                                       ▼
   [ Vendor Package A ]                                    [ Vendor Package B ]
  (100s of uploaded pages)                               (100s of uploaded pages)
           │                                                       │
           ▼                                                       ▼
 [ 4-Stage Waterfall OCR ]                               [ 4-Stage Waterfall OCR ]
           │                                                       │
           ▼                                                       ▼
[ Batched Classification ]                              [ Batched Classification ]
           │                                                       │
           ▼                                                       ▼
 [ Parent-Child Hybrid RAG ]                            [ Parent-Child Hybrid RAG ]
           │                                                       │
           └───────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │        FINAL COMPLIANCE SCORECARD            │
                │  - Pass / Fail Eligibility Gates             │
                │  - Side-by-Side Comparative Matrix           │
                │  - Exact Page Citations & Evidence Snippets  │
                └──────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Pipelines & Flowcharts

### 1. Document Ingestion & Multi-Stage Waterfall OCR

Uploaded vendor packages undergo a multi-layered extraction triage to balance processing speed against scanned OCR accuracy:

```
                      UPLOADED VENDOR PDF DOCUMENTS
                   (e.g., "financials.pdf", "maf_cert.pdf")
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    Step 1: 4-Layer Waterfall OCR                      │
│                                                                       │
│  Layer 1: PyPDF         → Native digital text extraction (fastest)    │
│  Layer 2: pdfplumber    → Precision layout and table extraction       │
│  Layer 3: Tesseract OCR → Scanned document image OCR (CPU)            │
│  Layer 4: EasyOCR       → Deep-learning fallback for degraded scans   │
│                                                                       │
│  OUTPUT: Raw cleaned text with provenance markers: "--- PAGE X ---"   │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│              Step 2: Batched Document Classification (LLM)            │
│                                                                       │
│  - Single batched LLM call inspects previews of all vendor files.     │
│  - Classifies each into canonical doc_types (MAF, Balance Sheet, PAN) │
│  - Prevents Cross-Contamination: Financial queries only query Balance  │
│    Sheets, technical queries only query Datasheets.                  │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   Step 3: Parent-Child Chunking                       │
│                                                                       │
│  Parent Chunks: 1,000 characters (kept in RAM docstore for context)   │
│  Child Chunks :   300 characters (embedded for high-precision search) │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│              Step 4: Vector & Sparse Index Ingestion                  │
│                                                                       │
│  - Dense: all-MiniLM-L6-v2 embeddings cached via SHA-256 hashes       │
│  - Sparse: Okapi BM25 index built over tokenized child chunks         │
│  - Storage: In-memory ChromaDB isolated per vendor session            │
└───────────────────────────────────────────────────────────────────────┘
```

---

### 2. Hybrid Retrieval & Advanced RAG Flow

When validating requirements, Argus Bid AI does not run naive vector search. It enriches the prompt, conducts dual retrieval, and reranks candidate passages:

```
                               RAW AUDIT QUERY
                       "Verify average annual turnover"
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Step 1: Query Enrichment                         │
│                                                                        │
│  • Numerical / Financial checks (PQC, Turnover, Exp):                  │
│    → HyDE (Hypothetical Document Embeddings): LLM generates a          │
│      mock CA-certified balance sheet paragraph to embed.               │
│                                                                        │
│  • Spec & Entity checks (MAF, Datasheet, Deviation):                   │
│    → Query Expansion: LLM enriches query with domain terminology,      │
│      procurement acronyms (OEM, GFR, MSE), and synonyms.               │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Step 2: Filtered Dual Search                    │
│                                                                        │
│  • Filter: Constrained to target doc_type collection only              │
│  • Dense Search : Cosine similarity over child chunk vectors (Chroma)  │
│  • Sparse Search: Okapi BM25 keyword matching over tokenized chunks    │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Step 3: Reciprocal Rank Fusion (RRF)                 │
│                                                                        │
│  Merges dense & sparse rankings using RRF constant (k = 60):          │
│                    Score = Σ 1 / (k + rank)                           │
│  → Yields pooled candidate set (~15 chunks)                            │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Step 4: Cross-Encoder Reranking                      │
│                                                                        │
│  • Model: cross-encoder/ms-marco-MiniLM-L-6-v2                         │
│  • Evaluates [Original Raw Query + Candidate Chunk] as a joint pair.   │
│  • Produces deep cross-attention semantic scores and picks top k.      │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Step 5: Parent Resolution                         │
│                                                                        │
│  Small child chunks (300 chars) are mapped back to their enclosing    │
│  parent chunks (1,000 chars) to provide complete context to the LLM.   │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Step 6: LLM Audit Verification                      │
│                                                                        │
│  LLM reviews verified context, extracts actual vendor values, evaluates│
│  pass/fail against NIT rule, and outputs structured JSON rationale.    │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Deterministic Gate & Compliance Decision Matrix

```
                     VENDOR EVALUATION PIPELINE
                                  │
                                  ▼
                     [ Gate 1: Mandatory Checklist ]
               Did the vendor provide all mandatory documents?
                    ├── NO  → REJECTED (Missing Critical File)
                    └── YES → Proceed
                                  │
                                  ▼
                       [ Gate 2: MAF Validation ]
                   Is the OEM MAF present & valid?
                    ├── NO  → REJECTED (Missing / Invalid MAF)
                    └── YES → Proceed
                                  │
                                  ▼
                       [ Gate 3: PQC Thresholds ]
         Experience ≥ Required? & Turnover ≥ Required? (Numeric)
                    ├── NO  → REJECTED (Failed Pre-Qualification)
                    └── YES → Proceed
                                  │
                                  ▼
                     [ Gate 4: Technical Specs & Deviations ]
             Evaluate Mandatory Specs & Detect Unapproved Deviations
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │    WEIGHTED RESPONSIVENESS SCORE    │
               │         Score ∈ [0, 100%]           │
               └─────────────────────────────────────┘
```

---

## 💡 Core Technical Innovations

| Feature | Technical Implementation | Benefit |
| :--- | :--- | :--- |
| **Document Triage** | Regex for deterministic tokens (PAN, GSTIN); LLM for semantic verification | 0 ms instant verification for deterministic patterns; avoids LLM token waste. |
| **Cross-Contamination Firewall** | Files classified into categories (`DOC_TYPES`) before vector indexing | A query about turnover will never accidentally match a number in a technical spec sheet. |
| **Parent-Child Chunking** | 300-char children for indexing, 1000-char parents for context window | High-precision vector cosine matching without losing surrounding context. |
| **Hybrid RAG + RRF** | Dense Chroma embeddings + Okapi BM25 merged via Reciprocal Rank Fusion | Overcomes dense embedding blind spots for specific technical part numbers. |
| **HyDE Query Translation** | LLM creates hypothetical document paragraphs before embedding | Bridges the semantic gap between short queries and legal-style vendor certificates. |
| **Cross-Encoder Rerank** | `ms-marco-MiniLM-L-6-v2` sequence-pair reranking | Replaces approximate distance metrics with full bidirectional token attention. |
| **Evidence Provenance** | Every chunk carries `{ "source": file, "page": page_number }` metadata | Every pass/fail verdict includes exact page numbers and verbatim text citations. |

---

## 🛠️ Tech Stack

- **Core & Runtime:** Python 3.11
- **UI Framework:** Streamlit with custom glassmorphism styles (`ui_styles.py`)
- **Document Parsing & OCR:** `pdfplumber`, `pypdf`, `pytesseract`, `easyocr`
- **Vector Database:** `ChromaDB` (In-memory, session-isolated)
- **Sparse Search:** `rank-bm25` (Okapi BM25)
- **Embedding & Reranker Models:**
  - Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (Local CPU) / Gemini Cloud Embeddings
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Engine:** LangChain supporting Local LLMs (via Ollama: Llama-3, Qwen) or Cloud APIs (Google Gemini, Groq, Anthropic Claude)

---

## 📂 Repository Structure

```
tender-audit-compliance/
├── tender_audit_platform.py         # Main interactive Streamlit application
├── audit_engine.py                  # Deterministic compliance rules engine & gates
├── rag_engine.py                    # Hybrid RAG, ParentDocumentRetriever, HyDE, Reranker
├── ui_styles.py                     # Custom CSS, glassmorphic UI cards, KPI metrics
├── requirements.txt                 # Project dependencies
├── run.bat                          # Quick-launch Windows executable script
├── render.yaml                      # Cloud web deployment blueprint
├── interview_qs.md                  # Comprehensive interview preparation guide
├── important_questions_from_a_project.md # Technical deep-dive Q&A documentation
└── presentation_guide.md            # Architecture walkthrough & demonstration script
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11 installed
- (Optional for scanned PDFs) Tesseract OCR installed on your system
- (Optional for local LLMs) Ollama running locally with `llama3` or `qwen2.5`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/alokitadutta22/tender-audit-compliance.git
cd tender-audit-compliance

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Platform
```bash
streamlit run tender_audit_platform.py
```
Or simply double-click **`run.bat`** on Windows.

---

## 📊 Evaluation & Audit Matrix

When an audit is executed, Argus Bid AI renders:
1. **Executive Eligibility Banner:** Immediate Qualified / Disqualified status with failure rationale.
2. **Pre-Qualification Criteria (PQC) Table:** Min Experience, Average Annual Turnover, and GeM registration status.
3. **Mandatory Documents Checklist:** Status of required forms with detected document links.
4. **Technical Specifications Matrix:** Parameter-by-parameter audit against NIT requirements.
5. **Auditable Evidence Dossier:** Verbatim textual evidence with direct PDF page-number citations.

---

<div align="center">
  <sub>Developed by <b>Alokita Dutta</b> • Designed for Deterministic, Trustworthy AI in Public Procurement</sub>
</div>
