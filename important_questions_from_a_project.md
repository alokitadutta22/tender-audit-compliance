# 📌 Important Questions from a Project
*(Source: Instagram Reel — Generic data/AI project interview questions. Answered with reference to Argus Bid AI.)*

---

## 🟢 2-Minute Project Overview

**Argus Bid AI** is an end-to-end AI-powered compliance audit platform built for PSU (Public Sector Undertaking) procurement workflows — specifically designed for scenarios like IOCL (Indian Oil Corporation Limited) evaluating vendor bids for high-value government tenders.

### 🔍 The Problem
When a government organisation like IOCL floats a tender — say, for 500 Layer-3 network switches worth ₹5 crore — they receive bids from 10+ vendors. Each vendor submits a package of PDFs: PAN cards, GST certificates, audited balance sheets, Manufacturer's Authorization Forms (MAFs), technical datasheets, and experience certificates. A human procurement officer then manually reads through hundreds of pages per vendor to check compliance. That process takes days, is error-prone, and has no audit trail.

**Argus Bid AI automates that entire pipeline.**

---

### ⚙️ How It Works

#### Step 1 — Master Tender Parsing (The Rules Engine)
The officer uploads the Master Tender / NIT document. The system extracts all compliance rules from it using two mechanisms in tandem:

- **LLM-based extraction** (`rag_engine.py → parse_master_bid()`): A structured JSON prompt is sent to the LLM (Groq/Gemini/Ollama) along with the first 25,000 characters of the tender document. The LLM extracts: Tender ID, Pre-Qualification Criteria (PQC) like minimum experience (3 years) and minimum turnover (₹5 Crore), mandatory documents list (MAF, PAN, GST, Balance Sheet etc.), mandatory technical specs (Layer-3 switch, 800 Gbps throughput, 48 ports, 36-month warranty), and preferred specs (IPv6, MACsec, hardware stacking).

- **Regex-based deterministic fallback** (`audit_engine.py → parse_master_bid()`): If the LLM call fails, a battery of regex patterns extracts numeric thresholds like `r"minimum of\s*(\d+)\s*years?\s*of\s*experience"` and `r"turnover[^0-9]*?(?:inr|rs\.?)?\s*([\d,\.]+)\s*crore"` directly from the tender text.

#### Step 2 — Vendor Document Ingestion (4-Layer OCR Waterfall)
Each vendor's uploaded PDFs are passed through a 4-layer extraction pipeline (`audit_engine.py → extract_text_from_pdf_bytes()`):
1. **pypdf** — Fast text extraction for digitally-generated PDFs (the fast-path)
2. **pdfplumber** — More accurate extraction with table detection (Markdown table formatting included)
3. **pytesseract** — OCR for scanned/image-based pages, with vowel-ratio and vocabulary validation to discard garbage output
4. **easyocr** — Final OCR fallback using neural text recognition (GPU-optional, CPU mode by default)

Each extracted page is tagged with `--- PAGE X ---` markers to preserve page-level provenance throughout the entire pipeline.

#### Step 3 — Document Classification
Each uploaded file is classified into a document type (PAN Card, GST Registration, MAF, Audited Balance Sheet, etc.) using an LLM call with a structured JSON prompt (`rag_engine.py → classify_all_documents()`). This is done in a **single batched LLM call** for all vendor files simultaneously, with individual per-file fallback if the batch call fails. Classification prevents cross-contamination — a query about MAF only searches MAF-classified documents.

#### Step 4 — Vector Store & Retrieval (Hybrid RAG)
Documents are ingested into an **in-memory ChromaDB** vector store using a **Parent-Child chunking** strategy:
- **Child chunks**: 300 characters, embedded with `all-MiniLM-L6-v2` (HuggingFace, runs on CPU) and stored in ChromaDB
- **Parent chunks**: 1000 characters, kept in an `InMemoryStore` for richer context retrieval

All embeddings are cached via **SHA-256 hashing** (`CachedEmbeddings` class): `hash = SHA256("model_name:chunk_text")`. If the hash exists in `.embeddings_cache.pkl`, the vector is loaded instantly. API is only called on a cache miss.

A **BM25 (Okapi BM25)** sparse index is pre-built on all child chunks at ingestion time.

During retrieval, both the dense vector search (ChromaDB cosine similarity) and the sparse BM25 search are run in parallel. Results are merged using **Reciprocal Rank Fusion (RRF)** with constant k=60. The merged pool of ~15 candidates is then re-ranked using a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) which reads each `[query + chunk]` pair together as a sequence and produces a relevance score.

Query enrichment is applied before search:
- For financial/experience checks → **HyDE** (Hypothetical Document Embeddings): The LLM generates a *hypothetical vendor paragraph* and that is embedded instead of the raw query
- For MAF/datasheet checks → **Query Expansion**: The LLM rewrites the query with synonyms, abbreviations (OEM, GFR, MSE), and typical procurement phrases

Retrieved child chunks are swapped out for their **parent chunks** for richer LLM context (`ParentDocumentRetriever` pattern).

#### Step 5 — Compliance Checks & Scoring
Each vendor is evaluated against:
- **MAF Validity** — Does the MAF reference the correct Tender ID or a GeM bid number? Is it on OEM letterhead?
- **PQC** — Does the vendor meet experience and turnover thresholds? Verified by LLM against balance sheets and experience certificates.
- **Mandatory Specs** — Does the datasheet meet every technical parameter? Numeric comparisons (≥ 800 Gbps, ≥ 48 ports) and boolean checks (Layer-3 architecture present?).
- **Preferred Specs** — Bonus scoring for IPv6, MACsec, hardware stacking, extended warranty.
- **Document Inventory** — Is every mandatory document present? Missing docs trigger automatic disqualification.

#### Step 6 — Output
A **Comparative Statement Matrix** is generated — the same format a government officer would produce manually, showing each vendor ranked by score, with disqualification reasons, page-level citations, and a document checklist.

#### LLM Routing & Failover
A `RoutingChatModel` wraps all LLM calls. If the primary model (e.g., Groq Llama-3.3-70b) hits a 429 rate limit, it automatically fails over to Gemini 1.5 Flash, and then to a local Ollama model — zero manual intervention required.

---

## 🔴 5 Critical Interview Questions

*(From an Instagram reel — generic but answerable from this project)*

---

### ❓ Q1 — Where did the data come from and what was wrong with it?
> *"Real data is always broken. If yours was clean, you downloaded it."*

**Answer:**

The data in this project is government procurement documents — PDFs submitted by vendors in response to a Notice Inviting Tender (NIT). These are not clean, structured datasets. They are the messiest category of real-world documents that exist.

Here is specifically what was broken:

**Problem 1: Mixed PDF types.** Some PDFs are digitally generated (text is directly selectable). Others are scanned physical documents (pure images with no embedded text). A single vendor's submission could have both — a digitally-generated GST certificate alongside a hand-stamped, scanned MAF letter. `pypdf` extracts the text PDFs perfectly but returns empty strings for the scanned ones. So a single-engine approach would silently miss entire documents.

**Problem 2: OCR noise and garbage output.** When pytesseract runs on a low-resolution scan, it often produces strings like `"|\\|||/l"` or `"AAXUNILNBER"` (a mangled "PAN NUMBER"). If you trust that output and search for a PAN number regex, you get a false negative. The system detects this by checking vowel ratio (< 0.10 vowel ratio = garbage) and presence of common English/tender vocabulary before accepting OCR output as valid.

**Problem 3: Table data in PDFs.** Technical datasheets and compliance matrices have tabular data (rows for spec parameters, columns for required vs. provided values). `pypdf` cannot extract tables — it just produces unsorted raw text. `pdfplumber` extracts the table structure (`page.extract_tables()`) and the code converts it to Markdown format before embedding it, so the LLM can read it as a table rather than as a garbled string.

**Problem 4: Junk bytes and malformed PDFs.** Some PDFs had leading junk bytes before the `%PDF` header marker, which caused both `pypdf` and `pdfplumber` to throw exceptions. The solution was a sanitization step: scan for the first `%PDF` byte marker, strip everything before it, then proceed with extraction.

**Problem 5: Index / TOC pages contaminating results.** Vendor submissions include an index page listing "PAN Card — Page 2, MAF — Page 4, Balance Sheet — Page 6." When the system tries to find the PAN card page, the TOC page scores high because it mentions "PAN Card" — but it is not the actual PAN card. The `resolve_actual_document_page()` function addresses this with a scoring system that penalizes any page containing index signals (`"table of contents"`, `"checklist"`, `"enclosed herewith"`) that also mention multiple other document types simultaneously.

---

### ❓ Q2 — How many rows?
> *"People who built it know, people who watched a tutorial guess."*

**Answer:**

This project does not work with a fixed-size tabular dataset, so "rows" is not the right unit — but here are the actual numbers from what the system processes:

- **Vendor PDFs per audit run**: Typically 5–15 files per vendor, with 3–10 vendors per tender. So a full audit involves 15–150 uploaded files in one session.
- **Pages extracted**: Across a real tender evaluation, the system processes 200–800 PDF pages total (after the 4-layer OCR waterfall).
- **Text chunks stored in ChromaDB (child chunks)**: Approximately 800–3,000 child chunks of 300 characters each per vendor, depending on the volume of their documents. The BM25 index is built on all of these.
- **Parent chunks in InMemoryStore**: Same count as child chunks, at 1,000 characters each.
- **Embedding cache entries**: After a full audit run, `.embeddings_cache.pkl` contains several thousand SHA-256 keyed vectors. On subsequent runs with similar documents (e.g., the same vendor re-submitting with one updated file), the vast majority of embeddings are cache hits — zero API calls.
- **Compliance checks per vendor**: Typically 12–20 (6 PQC checks + 6 mandatory spec checks + 4–8 preferred spec checks + 1 MAF check + document inventory).
- **Rules extracted from a tender document**: In the mock IOCL NIT included in the codebase, the system extracts 2 PQC rules, 6 mandatory specs, 3 preferred specs, and 5 mandatory documents.

---

### ❓ Q3 — What did you throw away and why?
> *"Every real analysis drops something. If nothing was excluded, no decisions were made."*

**Answer:**

Several explicit exclusion decisions were made — and each was deliberate:

**1. Table extraction for non-technical files was disabled.**
Running `pdfplumber.extract_tables()` is computationally expensive — it increases parsing time by 10–15x because it renders every page and scans for line geometries. Technical datasheets and BOQs genuinely have tabular data worth extracting. But a PAN card, an MAF letter, and a GST certificate are text-only documents. The code explicitly checks the filename: if it does not contain `"datasheet"`, `"spec"`, `"compliance"`, `"nit"`, `"bid"`, or `"technical"` — table extraction is switched off for that file. A fast-path with `pypdf` is used instead. This alone cut average parsing time by ~60%.

**2. Low-quality OCR output was discarded.**
When pytesseract produces output with < 15 alphanumeric characters, < 10% vowel ratio, or no common English/tender vocabulary, the output is thrown away. The system does not use it — it falls through to EasyOCR instead, or accepts the page as blank. Using garbage OCR output would poison the vector store with noise chunks that retrieve incorrectly and mislead the LLM.

**3. The first 25,000 characters only from the tender document are sent to the LLM for rule extraction.**
The full tender document can be 60+ pages. The pre-qualification criteria, mandatory documents, and technical specifications almost always appear in the first 10 pages. The remaining content (boilerplate financial terms, payment schedules, penalty clauses) is irrelevant for rule extraction. Sending 60 pages to an LLM wastes context window and increases latency and cost. The first 25,000 characters (~10 pages) are sent; the rest is discarded for this specific step.

**4. ChromaDB is in-memory only — it is not persisted.**
The vector store is destroyed at the end of every audit session. Persisting it would require a stable disk-based vector store (like Chroma with `persist_directory`) and a session management system. That complexity was intentionally excluded from the prototype. Instead, what is persisted is the *embedding cache* (`.embeddings_cache.pkl`) — so the expensive part (API calls) is cached, and the cheap part (in-memory indexing, which takes < 1 second) is rebuilt fresh each run.

**5. Only the top k=3 documents are passed to the LLM after reranking.**
The retrieval pool has 15 candidates after RRF fusion. After Cross-Encoder reranking, only the top 3 are assembled into context. The rest are dropped. Sending 15 chunks to the LLM would exceed the token budget and dilute attention. The Cross-Encoder's job is to identify the 3 that are *actually relevant*, and trust that those 3 are sufficient. This is a deliberate precision-over-recall tradeoff.

---

### ❓ Q4 — Who acted on it?
> *"A dashboard nobody opened is a screenshot. Tell me what changed because it existed."*

**Answer:**

This is a fair challenge for a prototype. Here is an honest answer:

The system was built and demonstrated for the **procurement evaluation workflow at PSU-style organizations** — specifically modeled after IOCL Haldia Refinery's IT department tender process. In the demo context, **a procurement officer using the tool** would get the same Comparative Statement Matrix in **under 5 minutes** that previously took 2–3 working days of manual review.

What actually changed in practice during development:

**The output format changed because of how it was used.** The initial version produced only a simple pass/fail verdict per vendor. But when walking through a sample audit, it became clear that a procurement officer cannot act on "Vendor B: DISQUALIFIED." They need to be able to defend their decision to a tender committee. The system was reworked to produce **page-level citations with source file names and page numbers** for every finding — so the officer can open the exact PDF page that was evidence for a disqualification. That is actionable. A score without a source is just a claim.

**The MAF validation rule changed after testing.** The first version of the MAF validator required the Tender ID to appear verbatim in the MAF letter. In reality, many OEMs issue MAF letters referencing a GeM Bid Number (e.g., `GEM/2026/B/12345`) rather than the NIT Tender ID. The first version was disqualifying valid MAFs. The prompt and validation logic were updated: a valid MAF can reference the Tender ID *or* a GeM Bid Number *or* a generic phrase like "above-mentioned tender." That change directly came from testing with real-world sample MAF documents.

**The document page resolver was built because the naive approach failed.** The first version pointed to the page where the document name appeared first — which was almost always the index page. A vendor's PAN card, physically at page 5, was being cited as "found on page 1" because page 1 was the index. The `resolve_actual_document_page()` function with TOC penalty scoring was built specifically because this mistake was caught during a walkthrough.

---

### ❓ Q5 — What would you build differently now?
> *"Nothing is the worst possible answer."*

**Answer:**

Quite a few things:

**1. Replace Streamlit session state with a proper job queue.**
Right now, all audit state lives in `st.session_state` — a per-browser-tab dictionary. Two users uploading simultaneously would corrupt each other's session. In a production version, every audit run would be a **UUID-namespaced job** submitted to a **Celery worker queue** backed by **Redis**. The Streamlit (or React) frontend would poll for job status. The audit engine is already stateless — it takes inputs, returns outputs. The infrastructure around it needs hardening.

**2. Use a persistent vector store with proper namespacing.**
ChromaDB in-memory is rebuilt from scratch every session. For a production system handling hundreds of audits, a persistent Chroma instance (or a managed vector DB like Pinecone or Weaviate) with vendor-namespaced collections would eliminate re-embedding entirely, not just reduce API calls via cache.

**3. Implement a proper cross-encoder fine-tuned on procurement documents.**
The Cross-Encoder used (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is trained on MS MARCO — a web search dataset. It has never seen the phrase "hot-swappable PSU" or "GeM consignee receipt." A fine-tuned cross-encoder on Indian government procurement text would have significantly better reranking accuracy for this domain.

**4. Add structured evaluation metrics with a labelled test set.**
Right now, there is no quantitative measurement of retrieval quality. Precision@3, Recall@3, and NDCG are not measured because there is no ground-truth dataset of "query → correct page" pairs. Building even 50 labelled examples from real IOCL tenders would allow systematic comparison of retrieval strategies — e.g., does HyDE actually outperform straight query expansion for balance sheet checks? Currently that decision is based on intuition, not measurement.

**5. Decouple the document classifier from the main audit run.**
Right now, document classification happens during the audit run — it blocks progress. In a better design, classification would happen immediately upon file upload (async background task), with results cached. By the time the officer clicks "Run Audit", classification is already done. This would shave 15–30 seconds off every audit run (which is where most of the visible latency currently comes from).

**6. Build an explicit feedback loop for officer corrections.**
If the officer looks at the result and says "this MAF is actually invalid" or "this balance sheet shows a different financial year", there is currently no way to record that correction. A structured feedback mechanism — where officer overrides are logged, timestamped, and used to refine prompts for future runs — would make the system genuinely improve over time from real-world usage.

---

*File created: September 2026 | Project: Argus Bid AI — Tender Audit & Compliance*
