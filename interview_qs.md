# 🎙️ Technical Interview Preparation Guide — Argus Bid AI

---

## 🟢 SECTION 1: WARM-UP & PROJECT OVERVIEW

### ❓ Q1 — The Opening (Walkthrough)
> **"Alright, walk me through your project. What does it do, how does it work, and why did you build it?"**

#### 📝 Script:
> "So the problem I was solving is this — imagine a government oil company like IOCL is buying 500 networking switches worth 5 crore rupees. They receive bids from 10 different vendors, and someone in their procurement team has to manually read through hundreds of pages of PDFs — checking PAN cards, balance sheets, experience certificates, technical datasheets — for *each* vendor. That is slow, error-prone, and completely manual.
>
> I built **Argus Bid AI** to automate this entire compliance audit using AI.
>
> Here is how it works at a high level:
> - The procurement officer uploads the **Master Tender document** — this is the NIT, the Notice Inviting Tender. The system reads it and extracts all the rules: minimum turnover of 21 crore, at least 5 years of experience, a MAF from the OEM, specific technical specs like 48 ports or Layer-3 switching.
> - Then the officer uploads each **vendor's bid package** — PDFs of their documents.
> - The system runs a full compliance audit — checking every rule against every vendor — and produces a ranked scorecard, showing who qualifies, who is disqualified and exactly why, with page-level citations.
>
> The output is a **Comparative Statement Matrix** — the same format a government officer would produce manually, but generated in minutes instead of days."

---

### ❓ Q2 — Tech Stack Deep Dive
> **"What was your tech stack and why did you choose each piece?"**

#### 📝 Script:
| Component | Technology | Why |
|---|---|---|
| **Frontend / App** | Streamlit | Fastest way to build an interactive data dashboard in pure Python. No HTML/JS needed. |
| **Document Parsing** | pdfplumber, pypdf, Tesseract, EasyOCR | Government documents are mixed — some are digitally generated PDFs, some are scanned images. Needed a 4-layer waterfall stack to handle both. |
| **Vector Store** | ChromaDB | Lightweight, runs fully in-memory, no external server needed. Perfect for a single-session audit run. |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` / Gemini Embeddings | HuggingFace runs locally on CPU at zero API cost. Gemini is the cloud upgrade for higher quality. |
| **LLM** | Groq (Llama 3.1), Gemini, Ollama | Groq is very fast for short structured queries. Gemini for long complex reasoning. Ollama for fully offline deployments. |
| **Retrieval** | LangChain + BM25 (rank-bm25) | LangChain manages the retrieval pipeline. BM25 gives keyword/sparse search that vectors miss for exact terms like PAN numbers or Tender IDs. |
| **Orchestration** | Python `concurrent.futures.ThreadPoolExecutor` | Runs 4 vendor file parsers simultaneously to cut down parsing time. |

> *"I chose Python because the entire ML/NLP ecosystem lives there. I chose Streamlit over Flask/React because procurement officers are not developers — they need a simple upload-and-click interface, not a complex web app."*

---

### ❓ Q3 — Why Regex First?
> **"Why regex first? Why not just use the LLM for everything?"**

#### 📝 Script:
> *"Think of it like a hospital triage system. When someone walks in with a broken arm that's visibly sticking out, you don't need a specialist — a nurse can handle that immediately. But when someone says 'my chest hurts', you escalate to a doctor.*
>
> *Similarly, a PAN number always looks like ABCDE1234F — 10 characters, a fixed pattern. A regex finds it in 0 milliseconds with 100% accuracy. If I send that to an LLM, it takes 2 seconds, costs API credits, and has a tiny chance of hallucinating.*
>
> *But when the question is 'does this vendor have 5 years of relevant experience in government networking projects' — that requires reading, reasoning, and understanding context. That is when the LLM earns its place.*
>
> *Regex for what is deterministic. LLM for what requires judgment."*

---

## 🔵 SECTION 2: END-TO-END PIPELINE FLOWS

### 1. The Query Processing Flow
```
RAW QUERY
"check financial turnover"
        │
        ▼
┌─────────────────────────────────────┐
│  Step 1: Query Enrichment           │
│                                     │
│  IF doc_type = Balance Sheet/PQC    │
│  → HyDE: LLM generates a           │
│    hypothetical vendor paragraph    │
│                                     │
│  IF doc_type = MAF/Datasheet        │
│  → Expansion: LLM rewrites with     │
│    synonyms + abbreviations         │
│                                     │
│  OUTPUT: enriched search_query      │
│  (still plain TEXT, not yet vector) │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 2: EMBED the search_query     │
│                                     │
│  search_query → embed_query()       │
│  → [0.23, -0.14, 0.87, ...]        │
│                                     │
│  The query gets embedded and cached │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: DUAL SEARCH on pre-built Chroma vector DB  │
│                                                     │
│  Dense: query_vector ↔ chunk_vectors (cosine sim)   │
│  Sparse: tokenize(search_query) → BM25 scores       │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 4: RRF Fusion                 │
│  Merges the two ranked lists        │
│  → Pool of ~15 best candidates      │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 5: Cross-Encoder Reranking    │
│                                     │
│  Uses ORIGINAL RAW QUERY (Not HyDE) │
│  Scores each of the 15 candidates   │
│  by reading [raw_query + chunk]     │
│  together to return top k results   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 6: Parent Resolution          │
│  Small child chunk → swapped for    │
│  its larger parent chunk            │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Step 7: Token-Aware Assembly       │
│  Pack top chunks into context       │
│  up to 6,000 token budget           │
└───────────────┬─────────────────────┘
                │
                ▼
              LLM
         (reads context
          + gives verdict)
```

### 2. The Document Processing Flow (Conveyor Belt)
```
UPLOADED PDF FILES (vendor documents)
e.g., "balance_sheet.pdf", "maf_letter.pdf"
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Step 1: 4-Layer Waterfall OCR                  │
│  pypdf → pdfplumber → Tesseract → EasyOCR       │
│  OUTPUT: raw text string per page               │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 2: Document Classification (LLM)          │
│  Assigns doc_type label (e.g. PAN Card)         │
│  Stops search cross-contamination               │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 3: Page-Level Document Objects            │
│  Text split by "--- PAGE X ---" markers         │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 4: Parent-Child Splitting                 │
│  Parent size = 1000 chars (Stored in RAM)       │
│  Child size = 300 chars (Embedded + Stored DB)  │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 5: Embedding (with Cache)                 │
│  Checks SHA256("model_name:chunk_text") hash    │
│  Cache Hit? → Load vector instantly            │
│  Cache Miss? → Call API & save                 │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 6: ChromaDB Storage                       │
│  One collection per vendor for strict isolation │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  Step 7: BM25 Index Pre-Build                   │
│  Tokenize all texts and cache the BM25 index    │
└─────────────────────────────────────────────────┘
```

---

## 🟡 SECTION 3: SYSTEM DEVIATIONS & SCALE

### ❓ Q4: "If 10 procurement officers used this simultaneously, what would break?"
📝 **Script:**
> *"Almost everything, honestly — and I'll be upfront about it. The current architecture is single-user. All state is stored in Streamlit's session state — which is per-browser-tab. But the underlying Python process is shared. If two users upload files simultaneously, they would be writing to the same cache files and could corrupt each other's session.
>
> The correct production architecture would be: FastAPI backend with a Redis job queue, Celery workers handling one audit job per worker, each user gets a UUID-namespaced audit job, results stored in PostgreSQL, and the Streamlit frontend becomes a status-polling dashboard. The audit engine itself is stateless, so it is horizontally scalable; it is the infrastructure around it that needs hardening."*

### ❓ Q5: "ChromaDB stores vectors in memory. What happens if the system crashes mid-audit?"
📝 **Script:**
> *"This is a genuine architectural weakness of the current prototype. The Chroma vector collection is in-memory and deleted at the end of the run. However, we mitigate this by persisting the raw parsed document text in `.sentinel_cache.pkl` and the computed embeddings in `.embeddings_cache.pkl`. On a restart, the vector DB can be rebuilt almost instantly without calling the API again. In production, we would use a persistent vector store service."*

---

## ✍️ SECTION 4: PEN-AND-PAPER CODE SNIPPETS

### 1. Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    scores = {}

    # Score from dense vector results
    for rank, doc in enumerate(vector_results, start=1):
        key = doc.page_content          # unique identifier
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    # Score from BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    # Sort by combined score, highest first
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return sorted_keys
```

### 2. Embedding Cache with SHA-256 Hashing
```python
import hashlib, pickle, os

class CachedEmbeddings:
    def __init__(self, model, cache_file=".embeddings_cache.pkl"):
        self.model = model
        self.cache_file = cache_file
        self.cache = pickle.load(open(cache_file,"rb")) \
                     if os.path.exists(cache_file) else {}

    def _hash(self, text):
        key = f"{self.model.__class__.__name__}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

    def embed_documents(self, texts):
        results = []
        to_embed = []
        indices = []

        for i, text in enumerate(texts):
            h = self._hash(text)
            if h in self.cache:
                results.append(self.cache[h])
            else:
                results.append(None)
                to_embed.append(text)
                indices.append(i)

        if to_embed:
            new_vectors = self.model.embed_documents(to_embed)
            for i, vec in zip(indices, new_vectors):
                h = self._hash(texts[i])
                self.cache[h] = vec
                results[i] = vec
            pickle.dump(self.cache, open(self.cache_file,"wb"))

        return results
```

### 3. Parent-Child Chunking Resolution
```python
# ChromaDB yields child chunks
child_docs = vector_store.similarity_search(query, k=5)

# Extract doc_ids representing parents
doc_ids = [d.metadata.get("doc_id") for d in child_docs if d.metadata.get("doc_id")]

# Fetch parents from store
parent_docs = docstore.mget(doc_ids)
parent_map = {doc_id: p.page_content for doc_id, p in zip(doc_ids, parent_docs) if p}

# Swap child content for parent content
for child in child_docs:
    pid = child.metadata.get("doc_id")
    if pid in parent_map:
        child.metadata["parent_content"] = parent_map[pid]
```

### 4. Two-Stage Cross-Encoder Reranking
```python
from sentence_transformers import CrossEncoder

def rerank(query, candidate_docs, top_k=3):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Create [query, chunk] pairs
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked][:top_k]
```

### 5. HyDE (Hypothetical Document Embeddings)
```python
def generate_hyde(query, llm):
    prompt = f"Write a hypothetical paragraph from a vendor government bid that answers: '{query}'"
    hypothetical_doc = llm.invoke(prompt)
    return hypothetical_doc.content.strip()
```

---

## 💜 SECTION 5: RESUME MISMATCH STRATEGY

> **"Your resume mentions the previous version of the project. How do you handle questions about this?"**

#### 📝 Script:
> *"My resume represents a snapshot of the project at the time of submission. The core architecture — Parent-Child chunking, Hybrid Dense + BM25, and Regex-First fallbacks — is exactly as described.
>
> However, after submitting, I kept developing the system because I identified real production weaknesses: redundant embedding API calls, Groq rate-limiting failures, and vocabulary mismatches in search. I iteratively solved these with embedding caching, routing failover models, HyDE, and Cross-Encoder reranking. This represents my commitment to building complete, robust software rather than just resume items."*
