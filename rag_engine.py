"""
================================================================================
 Argus Bid AI — Local Llama RAG Auditing & Compliance Module
 Built for PSU procurement workflows (IOCL-style NIT / BID evaluation)
================================================================================
 This module contains the local RAG audit engine, which uses LangChain, Chroma,
 and Ollama to perform semantic vector chunk matches and LLM-based verification.
================================================================================
"""

from __future__ import annotations

import re
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Import core audit engine classes & constants
from audit_engine import (
    AuditEngine,
    VendorResult,
    SpecResult,
    PQCResult,
    MAFResult,
    InventoryItem,
    Violation,
    STATUS_RESPONSIVE,
    STATUS_DISQUALIFIED,
    MAF_VALID,
    MAF_INVALID,
    MAF_MISSING,
    READ_PASS,
    READ_LOW,
    READ_CORRUPT,
    write_engine_log,
)

# LangChain and vector store imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger("rag_engine")
logger.setLevel(logging.INFO)

import urllib.request
import hashlib
import pickle
import os
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage

class CachedEmbeddings(Embeddings):
    def __init__(self, inner_embeddings: Embeddings, cache_file: str = ".embeddings_cache.pkl"):
        self.inner = inner_embeddings
        self.cache_file = cache_file
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load embedding cache: {e}")
                self.cache = {}

    def _get_hash(self, text: str) -> str:
        model_name = getattr(self.inner, "model_name", getattr(self.inner, "model", self.inner.__class__.__name__))
        key = f"{model_name}:{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        hashes = [self._get_hash(t) for t in texts]
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, h in enumerate(hashes):
            if h in self.cache:
                results[idx] = self.cache[h]
            else:
                missing_indices.append(idx)
                missing_texts.append(texts[idx])

        if missing_texts:
            try:
                embedded = self.inner.embed_documents(missing_texts)
                for idx, emb in zip(missing_indices, embedded):
                    results[idx] = emb
                    self.cache[hashes[idx]] = emb
                self._save_cache()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise e

        return results

    def embed_query(self, text: str) -> List[float]:
        h = self._get_hash(text)
        if h in self.cache:
            return self.cache[h]
        emb = self.inner.embed_query(text)
        self.cache[h] = emb
        self._save_cache()
        return emb


class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        import urllib.parse
        
        batch_size = 50
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            requests_payload = []
            for text in batch_texts:
                requests_payload.append({
                    "model": f"models/{self.model_name}",
                    "content": {
                        "parts": [{"text": text}]
                    }
                })
            
            body = {
                "requests": requests_payload
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key.startswith("AQ."):
                headers["Authorization"] = f"Bearer {self.api_key}"
                url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:batchEmbedContents"
            else:
                headers["x-goog-api-key"] = self.api_key
                encoded_key = urllib.parse.quote(self.api_key)
                url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:batchEmbedContents?key={encoded_key}"
            
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    batch_embeddings = [emb["values"] for emb in res["embeddings"]]
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Gemini batch embedding generation failed: {e}")
                logger.info("Falling back to single embedding requests with backoff...")
                for text in batch_texts:
                    for attempt in range(3):
                        try:
                            embeddings.append(self.embed_query(text))
                            break
                        except Exception as e_single:
                            if "429" in str(e_single) and attempt < 2:
                                time.sleep(2.0 * (attempt + 1))
                            else:
                                raise e_single
            
            if i + batch_size < len(texts):
                time.sleep(2.5)
                
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key.startswith("AQ."):
            headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:embedContent"
        else:
            headers["x-goog-api-key"] = self.api_key
            encoded_key = urllib.parse.quote(self.api_key)
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:embedContent?key={encoded_key}"
            
        body = {
            "model": f"models/{self.model_name}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res["embedding"]["values"]
        except Exception as e:
            logger.error(f"Gemini embedding generation failed: {e}")
            raise Exception(f"Gemini embedding generation failed: {e}")

class GeminiChatModel(SimpleChatModel):
    api_key: str
    model_name: str = "gemini-1.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        system_instruction = ""
        contents = []
        for m in messages:
            if m.type == "system":
                system_instruction = m.content
            elif m.type == "human" or m.type == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.type == "ai" or m.type == "assistant" or m.type == "model":
                contents.append({"role": "model", "parts": [{"text": m.content}]})
                
        import urllib.parse
        headers = {"Content-Type": "application/json"}
        if self.api_key.startswith("AQ."):
            headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        else:
            headers["x-goog-api-key"] = self.api_key
            encoded_key = urllib.parse.quote(self.api_key)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={encoded_key}"
            
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096
            }
        }
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        import time
        import random
        max_retries = 8
        backoff = 3.0
        
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    text = res["candidates"][0]["content"]["parts"][0]["text"]
                    return text
            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                if status_code == 429 and attempt < max_retries - 1:
                    wait_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(f"Gemini API 429 Rate Limit hit. Attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Gemini API HTTP Error {status_code}: {http_err.read().decode('utf-8', errors='ignore')}")
                    raise http_err
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(f"Gemini API 429 Rate Limit hit (Exception). Attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s before retrying...")
                    time.sleep(wait_time)
                    continue
                if attempt == max_retries - 1:
                    logger.error(f"Gemini API generation failed after {max_retries} attempts: {e}")
                    raise Exception(f"Gemini API generation failed: {e}")
                wait_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait_time)


class GroqChatModel(SimpleChatModel):
    api_key: str
    model_name: str = "llama-3.3-70b-versatile"

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        system_instruction = ""
        contents = []
        for m in messages:
            if m.type == "system":
                system_instruction = m.content
            elif m.type == "human" or m.type == "user":
                contents.append({"role": "user", "content": m.content})
            elif m.type == "ai" or m.type == "assistant" or m.type == "model":
                contents.append({"role": "assistant", "content": m.content})
                
        payload_messages = []
        if system_instruction:
            payload_messages.append({"role": "system", "content": system_instruction})
        payload_messages.extend(contents)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        body = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": 0.0,
            "max_tokens": 4096
        }
            
        import time
        max_retries = 8
        backoff = 3.0
        
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    text = res["choices"][0]["message"]["content"]
                    return text
            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                if status_code == 429:
                    import random
                    retry_after = http_err.headers.get("Retry-After")
                    raw_wait = float(retry_after) if retry_after and retry_after.replace('.','',1).isdigit() else ((backoff ** attempt) + random.uniform(0.5, 1.5))
                    MAX_WAIT = 120.0
                    if raw_wait > MAX_WAIT:
                        raw_wait = MAX_WAIT
                    wait_time = raw_wait
                    logger.warning(f"Groq API 429 Rate Limit hit. Attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.1f}s before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Groq API HTTP Error {status_code}: {http_err.read().decode('utf-8', errors='ignore')}")
                    raise http_err
            except Exception as e:
                import random
                if attempt == max_retries - 1:
                    logger.error(f"Groq API generation failed after {max_retries} attempts: {e}")
                    raise Exception(f"Groq API generation failed: {e}")
                wait_time = (backoff ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
                
        raise Exception("Groq API rate limit retries exhausted.")


class RoutingChatModel(SimpleChatModel):
    primary_llm: Any
    fallback_llms: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "routing-chat"

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        try:
            response = self.primary_llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}. Attempting fallbacks...")
            for fallback in self.fallback_llms:
                try:
                    write_engine_log(f"[Routing Engine] Primary LLM failed/rate-limited. Failover: routing call to fallback model...")
                    response = fallback.invoke(messages, **kwargs)
                    return response.content
                except Exception as fb_err:
                    logger.error(f"Fallback LLM failed: {fb_err}")
            raise e



def parse_json_safely(text: str) -> Any:
    """Robust JSON extraction and parsing to handle conversational prefixes or minor LLM malformations."""
    text_clean = text.strip()
    # Remove markdown code fences if present
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\n", "", text_clean)
        text_clean = re.sub(r"\n```$", "", text_clean)
        text_clean = text_clean.strip()
        
    # Find start of JSON structure
    start_idx = -1
    for i, char in enumerate(text_clean):
        if char in ("{", "["):
            start_idx = i
            break
            
    if start_idx != -1:
        # Find end of JSON structure
        end_idx = -1
        target_char = "}" if text_clean[start_idx] == "{" else "]"
        bracket_count = 0
        open_char = text_clean[start_idx]
        
        for i in range(start_idx, len(text_clean)):
            if text_clean[i] == open_char:
                bracket_count += 1
            elif text_clean[i] == target_char:
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i
                    break
        if end_idx != -1:
            json_str = text_clean[start_idx:end_idx+1]
            try:
                return json.loads(json_str)
            except Exception:
                # If standard json fails, try fixing minor issues like trailing commas or unquoted arithmetic/boolean expressions
                try:
                    # Clean up unquoted expressions like 1+1 or 1+1 PSU
                    json_str_cleaned = re.sub(r":\s*([0-9]+\+[0-9]+)\s*(,|$)", r': "\1"\2', json_str)
                    return json.loads(json_str_cleaned)
                except Exception:
                    pass
    
    # Fallback to direct json.loads
    try:
        return json.loads(text_clean)
    except Exception:
        raise ValueError(f"Could not parse JSON from output: {text}")


def create_langchain_documents(files: Dict[str, str], text_splitter: Optional[Any] = None, file_types: Optional[Dict[str, str]] = None) -> List[Document]:
    """Converts the raw file dictionary into page-level LangChain Documents with source, page, and doc_type metadata."""
    documents = []
    for filename, text in files.items():
        doc_type = file_types.get(filename, "Unclassified Document") if file_types else "Unclassified Document"
        # Split text into pages using page markers: "--- PAGE X ---"
        parts = re.split(r"--- PAGE (\d+) ---", text)
        if not parts:
            continue
            
        # The first part is any text preceding the first PAGE marker
        first_part = parts[0].strip()
        if first_part:
            documents.append(Document(
                page_content=first_part,
                metadata={"source": filename, "page": 1, "doc_type": doc_type}
            ))
        
        # Subsequent parts alternate between page number (as string) and page content
        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
            except ValueError:
                page_num = 1
            
            page_content = parts[i+1].strip() if i+1 < len(parts) else ""
            if page_content:
                documents.append(Document(
                    page_content=page_content,
                    metadata={"source": filename, "page": page_num, "doc_type": doc_type}
                ))
                    
    return documents


class LocalRAGAuditEngine(AuditEngine):
    """
    RAG-based Compliance Audit Engine.
    Supports local inference (via Ollama) or Cloud inference (via direct Google Gemini API integration).
    Performs semantic vector searches and structured compliance checks.
    """

    def __init__(self, model_name: str = "llama3", embedding_model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434", mode: str = "local", api_key: str = "",
                 gemini_api_key: str = "", embedding_provider: str = "Local Embeddings (HuggingFace CPU)"):
        super().__init__()
        self.mode = mode
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        
        # Initialize primary embeddings
        if mode == "gemini":
            if not api_key:
                raise ValueError("Gemini API Key is required for Cloud RAG mode.")
            if embedding_provider == "Local Embeddings (HuggingFace CPU)":
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                except Exception as e:
                    logger.warning(f"Failed to initialize HuggingFaceEmbeddings: {e}. Falling back to Gemini.")
                    self.embeddings = GeminiEmbeddings(api_key=api_key)
            else:
                self.embeddings = GeminiEmbeddings(api_key=api_key)
        elif mode == "groq":
            if not api_key:
                raise ValueError("Groq API Key is required for Cloud RAG (Groq) mode.")
            if embedding_provider == "Cloud Gemini Embeddings" and gemini_api_key:
                self.embeddings = GeminiEmbeddings(api_key=gemini_api_key)
            elif embedding_provider == "Local Ollama Embeddings":
                self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
            else:
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                except Exception as e:
                    logger.warning(f"Failed to initialize HuggingFaceEmbeddings: {e}. Falling back to Ollama.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        else:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to initialize HuggingFaceEmbeddings: {e}. Falling back to Ollama.")
                self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

        # Wrap in CachedEmbeddings
        self.embeddings = CachedEmbeddings(self.embeddings)

        # Initialize LLM with Routing and Failover
        primary_llm = None
        fallbacks = []

        try:
            if mode == "gemini":
                primary_llm = GeminiChatModel(api_key=api_key, model_name=model_name)
                
                groq_env_key = os.environ.get("GROQ_API_KEY", "")
                if groq_env_key:
                    fallbacks.append(GroqChatModel(api_key=groq_env_key, model_name="llama-3.3-70b-versatile"))
                fallbacks.append(ChatOllama(model="qwen2.5:7b", temperature=0.0, base_url=base_url))
                self.has_rag_backend = True
            elif mode == "groq":
                primary_llm = GroqChatModel(api_key=api_key, model_name=model_name)
                
                g_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
                if g_key:
                    fallbacks.append(GeminiChatModel(api_key=g_key, model_name="gemini-1.5-flash"))
                fallbacks.append(ChatOllama(model="qwen2.5:7b", temperature=0.0, base_url=base_url))
                self.has_rag_backend = True
            else:
                primary_llm = ChatOllama(model=model_name, temperature=0.0, base_url=base_url)
                
                g_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
                if g_key:
                    fallbacks.append(GeminiChatModel(api_key=g_key, model_name="gemini-1.5-flash"))
                groq_env_key = os.environ.get("GROQ_API_KEY", "")
                if groq_env_key:
                    fallbacks.append(GroqChatModel(api_key=groq_env_key, model_name="llama-3.3-70b-versatile"))
                self.has_rag_backend = True
                
            self.llm = RoutingChatModel(primary_llm=primary_llm, fallback_llms=fallbacks)
        except Exception as e:
            logger.error(f"Failed to initialize RAG LLM components: {e}")
            self.has_rag_backend = False

        # Define splitters for ParentDocumentRetriever (Phase 4)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        self.json_parser = JsonOutputParser()

    def classify_document(self, filename: str, text: str, bid: Optional[Dict[str, Any]] = None) -> str:
        """Classifies the document content using the local LLM (Llama 3) via structured JSON prompt."""
        from audit_engine import DOC_TYPES
        
        # Prepare a clean preview of the text (first 3000 chars)
        preview = (text or "").strip()
        if not preview:
            return "Unclassified Document"
            
        categories = list(DOC_TYPES)
        if bid and bid.get("mandatory_docs"):
            for d in bid["mandatory_docs"]:
                if d not in categories:
                    categories.append(d)
            
        system_prompt = (
            "You are a professional document classifier for procurement audits.\n"
            "Your task is to classify the uploaded document into exactly ONE of the following categories:\n"
            + "\n".join(f"- {t}" for t in categories) + "\n\n"
            "Return a JSON object with a single key 'category' containing the exact matching category name.\n"
            "Do not include any other keys, comments, markdown tags, or explanation. "
            "Response must be valid, parseable JSON."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("user", "Filename: {filename}\n\nContent (first 2500 characters):\n{preview}")
        ])
        
        chain = prompt | self.llm
        
        try:
            resp = chain.invoke({"filename": filename, "preview": preview[:2500]})
            res = parse_json_safely(resp.content)
            if isinstance(res, dict) and "category" in res:
                category = res["category"].strip()
                # Verify match
                for t in categories:
                    if category.lower() == t.lower():
                        return t
                # Fuzzy match
                for t in categories:
                    if t.lower() in category.lower() or category.lower() in t.lower():
                        return t
        except Exception as e:
            logger.error(f"LLM classification failed for {filename}: {e}")
            
        # Fallback to simple keyword check if LLM fails completely
        low = preview.lower()
        if "pan" in low and ("card" in low or "permanent account" in low):
            return "PAN Card"
        if "gstin" in low or "goods and services tax" in low or "gst" in low:
            return "GST Registration"
        if "balance sheet" in low or "profit and loss" in low:
            return "Audited Balance Sheet"
        if "authorization" in low and ("manufacturer" in low or "oem" in low):
            return "Manufacturer's Authorization Form (MAF)"
            
        return "Unclassified Document"

    def classify_all_documents(self, files: Dict[str, str], bid: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Classifies all documents in a single LLM call to save time, falling back to individual classification if it fails."""
        if not files:
            return {}
            
        from audit_engine import DOC_TYPES
        categories = list(DOC_TYPES)
        if bid and bid.get("mandatory_docs"):
            for d in bid["mandatory_docs"]:
                if d not in categories:
                    categories.append(d)
                    
        system_prompt = (
            "You are a professional document classifier for PSU procurement audits.\n"
            "Your task is to classify each of the uploaded documents into exactly ONE of the following categories:\n"
            + "\n".join(f"- {t}" for t in categories) + "\n\n"
            "Return a JSON object where keys are the exact filenames and values are the exact category names.\n"
            "Do not include any other keys, comments, markdown tags, or explanation. Response must be valid, parseable JSON."
        )
        
        documents_content = ""
        for fname, text in files.items():
            preview = (text or "").strip()[:1000]
            documents_content += f"--- FILENAME: {fname} ---\n{preview}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("user", "Please classify these documents:\n\n{documents_content}")
        ])
        
        chain = prompt | self.llm
        
        classified = {}
        try:
            resp = chain.invoke({"documents_content": documents_content})
            res = parse_json_safely(resp.content)
            if isinstance(res, dict):
                for fname in files.keys():
                    val = res.get(fname)
                    if val:
                        val_str = str(val).strip()
                        # Match category fuzzy
                        matched = None
                        for t in categories:
                            if val_str.lower() == t.lower():
                                matched = t
                                break
                        if not matched:
                            for t in categories:
                                if t.lower() in val_str.lower() or val_str.lower() in t.lower():
                                    matched = t
                                    break
                        if matched:
                            classified[fname] = matched
        except Exception as e:
            logger.error(f"Batch classification failed: {e}. Falling back to individual classification.")
            
        # Fill in missing files using individual classifier
        for fname, text in files.items():
            if fname not in classified:
                classified[fname] = self.classify_document(fname, text, bid)
                
        return classified

    def parse_master_bid(self, text: str) -> Dict[str, Any]:
        """Dynamically parses the Master Tender document (NIT) using a single contiguous LLM call with a base engine fallback."""
        self.bid_text = text
        
        if not self.has_rag_backend:
            raise Exception("RAG backend is not initialized. Please ensure Ollama is running or switch to Cloud RAG (Gemini).")
            
        write_engine_log("[RAG Engine] Extracting rules from Master Tender using contiguous context...")
        
        # Take the first 25000 characters of the text (covers ~10 pages where PQC, checklists, and specs are)
        tender_content = (text or "").strip()[:25000]
        
        system_prompt = (
            "You are an expert PSU procurement auditor. Your task is to analyze the Master Tender / NIT document "
            "and dynamically extract all requirements into a structured JSON object.\n\n"
            "Extract the following fields:\n"
            "1. 'tender_id': The unique Tender Number/ID (e.g. 'IOCL/HR/IT/2026/NW-4471').\n"
            "2. 'pqc': List of Pre-Qualification Criteria. Each item must have:\n"
            "   - 'key': 'experience' (for years of experience), 'turnover' (for financial turnover), or a short unique key for any other requirement.\n"
            "   - 'label': Human-readable label (e.g. 'Minimum Experience').\n"
            "   - 'threshold': Numerical value (e.g. 3 or 5).\n"
            "   - 'unit': Unit of measurement (e.g. 'years', 'INR Crore').\n"
            "   - 'section': Section number where this requirement appears.\n"
            "   - 'evidence': The exact text quote or sentence from the document containing this requirement.\n"
            "3. 'mandatory_docs': List of documents required from the bidder (e.g. ['Manufacturer\\'s Authorization Form (MAF)', 'PAN Card', 'GST Registration', 'Audited Balance Sheet']).\n"
            "4. 'mandatory_docs_evidence': A dictionary mapping each mandatory document name to the exact text quote/sentence from the document that mentions it (e.g. {'PAN Card': 'The bidder must submit a copy of PAN Card.'}).\n"
            "5. 'mandatory_specs': List of technical specifications listed under mandatory requirements. Each spec must have:\n"
            "   - 'key': A short unique slug (e.g. 'architecture').\n"
            "   - 'label': Parameter name (e.g. 'Switch Architecture (Layer-3)').\n"
            "   - 'op': Comparison operator: 'gte' (greater than or equal to), 'lte' (less than or equal to), or 'bool' (must be present/compliant).\n"
            "   - 'required_value': The target threshold value (e.g. 48 or 'Managed Layer-3 switch').\n"
            "   - 'unit': Unit of measurement (e.g. 'ports', 'Gbps').\n"
            "   - 'evidence': The exact text quote or sentence from the document containing this requirement.\n"
            "6. 'preferred_specs': List of technical specifications listed under preferred/desirable features. Format same as mandatory_specs.\n\n"
            "Respond with a JSON object containing these keys. Reply with ONLY valid JSON, no explanations, no markdown styling."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("user", "Relevant Tender Content extracted from full document:\n{tender_content}")
        ])
        
        chain = prompt | self.llm
        
        try:
            resp = chain.invoke({"tender_content": tender_content})
            res = parse_json_safely(resp.content)
            
            # Formulate the response
            return {
                "tender_id": str(res.get("tender_id", "")).strip(),
                "pqc": res.get("pqc", []),
                "mandatory_docs": res.get("mandatory_docs", []),
                "mandatory_docs_evidence": res.get("mandatory_docs_evidence", {}),
                "mandatory_specs": res.get("mandatory_specs", []),
                "preferred_specs": res.get("preferred_specs", []),
                "raw": text
            }
        except Exception as e:
            logger.error(f"Dynamic master bid parsing using LLM failed: {e}. Falling back to base regex parser...")
            write_engine_log("[RAG Engine] LLM Master Tender parsing failed. Executing deterministic base regex fallback.")
            # Fall back to base deterministic parser
            return super().parse_master_bid(text)

    def build_vector_store(self, files: Dict[str, str], file_types: Optional[Dict[str, str]] = None) -> Optional[Chroma]:
        """Creates an in-memory Chroma vector database for a vendor's documents."""
        if not self.has_rag_backend:
            return None
        
        page_docs = create_langchain_documents(files, file_types=file_types)
        if not page_docs:
            return None
            
        try:
            # Create an in-memory Chroma collection
            vector_store = Chroma(
                collection_name=f"vendor_{uuid.uuid4().hex}",
                embedding_function=self.embeddings
            )
            store = InMemoryStore()
            
            # Setup ParentDocumentRetriever (Phase 4)
            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )
            
            retriever.add_documents(page_docs)
            
            # Attach the store to the vector_store so we can resolve parent texts during search
            vector_store.docstore = store
            
            # Pre-build and cache the BM25 index on all child documents (Phase 3 Optimization)
            res = vector_store.get()
            contents = res.get("documents", []) or []
            metadatas = res.get("metadatas", []) or []
            all_docs = [Document(page_content=c, metadata=m) for c, m in zip(contents, metadatas)]
            
            import re
            def tokenize(text):
                return re.findall(r'\w+', text.lower())
            
            vector_store.all_docs = all_docs
            vector_store.tokenize_fn = tokenize
            
            if all_docs:
                from rank_bm25 import BM25Okapi
                corpus = [tokenize(d.page_content) for d in all_docs]
                vector_store.bm25 = BM25Okapi(corpus)
            else:
                vector_store.bm25 = None
                
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create Chroma vector store and ParentDocumentRetriever: {e}")
            write_engine_log(f"[ERROR] Failed to create Chroma vector store: {e}")
            return None

    def _expand_query(self, query: str) -> str:
        """Uses the LLM to rewrite/expand the query for better search coverage."""
        prompt = ChatPromptTemplate.from_template(
            "You are a search query expansion assistant. Expand the following query with relevant procurement terms, "
            "synonyms, abbreviations (like MSE, OEM, GFR, EMD), and typical phrases found in Indian PSU tenders.\n"
            "Query: {query}\n"
            "Return only the expanded query text without any introductory text, prefix, or code block."
        )
        chain = prompt | self.llm
        try:
            expanded = chain.invoke({"query": query})
            content = expanded.content.strip()
            content = content.replace("`", "").strip("'\"")
            return f"{query} | {content}"
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return query

    def _generate_hyde(self, query: str) -> str:
        """Generates a hypothetical document snippet answering the query (HyDE)."""
        prompt = ChatPromptTemplate.from_template(
            "Write a hypothetical paragraph from a vendor's bid document or certificate that would perfectly answer and satisfy the following search query:\n"
            "'{query}'\n"
            "Return only the hypothetical paragraph text. Do not include any explanations, formatting, or commentary."
        )
        chain = prompt | self.llm
        try:
            hyde_res = chain.invoke({"query": query})
            content = hyde_res.content.strip().replace("`", "").strip("'\"")
            return content
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Returning original query.")
            return query

    def _rerank_documents(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Reranks the retrieved documents using a local Cross-Encoder if available, otherwise returns top k."""
        if not docs:
            return []
        
        try:
            from sentence_transformers import CrossEncoder
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, doc.page_content] for doc in docs]
            scores = reranker.predict(pairs)
            
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            write_engine_log(f"[RAG Engine] Cross-Encoder reranked {len(docs)} documents. Top score: {scored_docs[0][1]:.4f}")
            return [doc for doc, score in scored_docs][:k]
        except Exception as e:
            logger.warning(f"Cross-Encoder reranking failed: {e}. Falling back to standard order.")
            return docs[:k]

    def _compress_text_by_sentence(self, text: str) -> str:
        """Compresses long passages by retaining only sentences that contain key content-bearing words."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 10:
            return text
            
        keywords = {
            "turnover", "experience", "audit", "certificate", "pan", "gst", "maf", "oem", "crore", "lakh", "rs",
            "year", "years", "authorized", "compliance", "specification", "deviation", "partner", "model", "make"
        }
        
        retained = []
        for i, sent in enumerate(sentences):
            low = sent.lower()
            has_kw = any(kw in low for kw in keywords)
            has_num = any(c.isdigit() for c in sent)
            has_symbol = any(sym in sent for sym in ["$", "₹", "INR", "%", "GEM"])
            
            if has_kw or has_num or has_symbol:
                retained.append(sent)
            elif i > 0 and sentences[i-1] in retained:
                retained.append(sent)
                
        if len(retained) < len(sentences):
            return " ".join(retained) + "\n... [Filler text compressed] ..."
        return text

    def _similarity_search_filtered(self, vector_store: Chroma, query: str, k: int = 3, doc_type: Optional[str] = None,
                                    use_expansion: bool = False, use_hyde: bool = False) -> List[Document]:
        """Performs a hybrid search combining dense similarity search and sparse BM25 search via Reciprocal Rank Fusion, followed by Cross-Encoder reranking."""
        # Determine defaults based on query contents/doc_type if not explicitly set
        if not use_expansion and not use_hyde:
            if doc_type in ("Manufacturer's Authorization Form (MAF)", "Technical Datasheet / Bid", "Deviation Statement"):
                use_expansion = True
            elif doc_type in ("Experience / Past Performance Certificate", "Audited Balance Sheet") or "experience" in query.lower() or "turnover" in query.lower():
                use_hyde = True
            else:
                use_expansion = True

        search_query = query
        if use_expansion:
            search_query = self._expand_query(query)
        elif use_hyde:
            search_query = self._generate_hyde(query)
            
        all_docs = getattr(vector_store, "all_docs", [])
        
        if doc_type:
            try:
                res = vector_store.get(where={"doc_type": doc_type})
                contents = res.get("documents", []) or []
                metadatas = res.get("metadatas", []) or []
                filtered_docs = [Document(page_content=c, metadata=m) for c, m in zip(contents, metadatas)]
            except Exception as e:
                logger.error(f"Failed to get filtered docs from Chroma: {e}")
                filtered_docs = [d for d in all_docs if d.metadata.get("doc_type") == doc_type]
        else:
            filtered_docs = all_docs

        if not filtered_docs:
            filtered_docs = all_docs

        candidate_k = max(15, k * 3)

        vector_results = []
        try:
            filter_dict = {"doc_type": doc_type} if doc_type else None
            vector_results = vector_store.similarity_search(search_query, k=candidate_k, filter=filter_dict)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Falling back to sparse search only.")

        bm25_results = []
        try:
            bm25 = getattr(vector_store, "bm25", None)
            tokenize = getattr(vector_store, "tokenize_fn", None)
            
            if bm25 and tokenize:
                tokenized_query = tokenize(search_query)
                scores = bm25.get_scores(tokenized_query)
                
                doc_scores = zip(all_docs, scores)
                if doc_type:
                    doc_scores = [(d, s) for d, s in doc_scores if d.metadata.get("doc_type") == doc_type]
                else:
                    doc_scores = list(doc_scores)
                    
                sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
                bm25_results = [d for d, score in sorted_docs if score > 0][:candidate_k]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}. Falling back to dense search only.")
            
        candidates = []
        if not vector_results and not bm25_results:
            candidates = filtered_docs[:candidate_k]
        elif not vector_results:
            candidates = bm25_results
        elif not bm25_results:
            candidates = vector_results
        else:
            rrf_scores = {}
            rrf_constant = 60
            
            def doc_id(doc):
                return (doc.metadata.get("source", ""), doc.metadata.get("page", 1), doc.page_content)

            for rank, doc in enumerate(vector_results, start=1):
                key = doc_id(doc)
                rrf_scores[key] = (1 / (rrf_constant + rank), doc)

            for rank, doc in enumerate(bm25_results, start=1):
                key = doc_id(doc)
                if key in rrf_scores:
                    prev_score, _ = rrf_scores[key]
                    rrf_scores[key] = (prev_score + (1 / (rrf_constant + rank)), doc)
                else:
                    rrf_scores[key] = (1 / (rrf_constant + rank), doc)

            sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
            candidates = [doc for score, doc in sorted_rrf][:candidate_k]

        retrieved_docs = self._rerank_documents(query, candidates, k)

        store = getattr(vector_store, "docstore", None)
        if store:
            doc_ids = [d.metadata.get("doc_id") for d in retrieved_docs if d.metadata.get("doc_id")]
            if doc_ids:
                try:
                    parent_docs = store.mget(doc_ids)
                    parent_map = {doc_id: p.page_content for doc_id, p in zip(doc_ids, parent_docs) if p}
                    for d in retrieved_docs:
                        d_id = d.metadata.get("doc_id")
                        if d_id and d_id in parent_map:
                            d.metadata["parent_content"] = parent_map[d_id]
                except Exception as e:
                    logger.error(f"Failed to retrieve parent documents from docstore: {e}")

        return retrieved_docs

    def _build_context_from_docs(self, docs: List[Document], max_tokens: int = 6000) -> str:
        """Constructs, deduplicates, and compresses context up to a strict token budget."""
        seen = set()
        context_parts = []
        total_tokens = 0
        
        for d in docs:
            src = d.metadata.get("source", "")
            pg = d.metadata.get("page", 1)
            key = (src, pg)
            if key not in seen:
                seen.add(key)
                
                parent_text = d.metadata.get("parent_content", d.page_content)
                if len(parent_text) > 4000:
                    parent_text = self._compress_text_by_sentence(parent_text)
                    
                tokens = len(parent_text) // 4
                if total_tokens + tokens > max_tokens:
                    remaining_tokens = max_tokens - total_tokens
                    if remaining_tokens > 200:
                        truncated_text = parent_text[:remaining_tokens * 4] + "\n... [Truncated due to token limit] ..."
                        context_parts.append(f"[File: {src}, Page: {pg}]\n{truncated_text}")
                        total_tokens += remaining_tokens
                    break
                    
                context_parts.append(f"[File: {src}, Page: {pg}]\n{parent_text}")
                total_tokens += tokens
                
        return "\n\n".join(context_parts)

    def validate_maf_rag(self, vector_store: Chroma, tender_id: str,
                         files: Optional[Dict[str, str]] = None,
                         file_types: Optional[Dict[str, str]] = None) -> MAFResult:
        """Audits the Manufacturer's Authorization Form (MAF) requirement using local RAG."""
        query = (
            f"Manufacturer's Authorization Form (MAF) OEM letterhead signature "
            f"authorizes bidder for Tender No. {tender_id}"
        )
        
        # Build context: Use pre-classified MAF documents if available to prevent chunking/retrieval omissions
        maf_files = []
        if file_types and files:
            maf_files = [fname for fname, dtype in file_types.items() if "MAF" in dtype or "Authorization" in dtype]
            
        if maf_files:
            # Bypass vector search: pass the entire content of the MAF files directly
            context = "\n\n".join([f"Source File: {f}\nContent:\n{files[f]}" for f in maf_files if f in files])
        else:
            # Retrieve context from vector store filtered by MAF doc_type, using k=3 for more context
            docs = self._similarity_search_filtered(vector_store, query, k=3, doc_type="Manufacturer's Authorization Form (MAF)")
            context = self._build_context_from_docs(docs)
            if not context:
                # Absolute fallback: search the raw files
                if files:
                    context = "\n\n".join([f"Source File: {n}\n{t[:5000]}" for n, t in files.items()])
                    
        if len(context) > 15000:
            context = context[:15000] + "\n... [Context truncated for brevity] ..."

        prompt = ChatPromptTemplate.from_template("""
        You are a strict procurement auditor. Review these document excerpts to verify if there is a valid Manufacturer's Authorization Form (MAF).
        
        A valid MAF must satisfy these conditions:
        1. It must be on the original equipment manufacturer's (OEM) letterhead. IMPORTANT: Affiliates, subsidiaries, and regional branches of the OEM (e.g., 'Samsung India' instead of 'Samsung Corp') are perfectly VALID OEM letterheads. Accept them.
        2. It must authorize the vendor to bid for this tender. NOTE: It is perfectly VALID if the MAF references the tender ID: "{tender_id}", OR a GeM Bid Number (e.g., GEM/2026/B/...), OR simply uses a generic phrase like "the above-mentioned tender" or "the respective NIT".
        
        Excerpts:
        ---
        {context}
        ---
        
        Determine if a valid MAF is found. Return a JSON object with:
        {{
            "status": "Found (Valid)" or "Invalid / Non-Compliant" or "MISSING / NOT FOUND",
            "evidence": "Detailed explanation of why it is valid or invalid, quoting the exact passage proving authorization.",
            "source_file": "Name of the file containing the MAF",
            "page": 1 (integer page number of the MAF)
        }}
        
        Important: Output ONLY the raw JSON block. No markdown wrapper, no extra text.
        """)

        chain = prompt | self.llm
        
        try:
            try:
                resp = chain.invoke({"tender_id": tender_id, "context": context})
            except Exception as api_err:
                # Self-healing retry for 413 Payload Too Large on MAF audit
                if "413" in str(api_err) or "too large" in str(api_err).lower():
                    logger.warning("Groq/LLM returned 413 Payload Too Large on MAF audit. Self-healing: retrying with reduced context...")
                    reduced_context = context[:12000]
                    resp = chain.invoke({"tender_id": tender_id, "context": reduced_context})
                else:
                    raise api_err
                    
            result = parse_json_safely(resp.content)
            
            status = result.get("status", MAF_MISSING)
            status_lower = str(status).lower()
            if "valid" in status_lower:
                status = "Found (Valid)"
            elif "invalid" in status_lower or "non-compliant" in status_lower:
                status = "Invalid / Non-Compliant"
            else:
                status = "MISSING / NOT FOUND"
                
            return MAFResult(
                status=status,
                evidence=result.get("evidence", "No valid MAF evidence returned by local RAG engine."),
                source_file=result.get("source_file", ""),
                page=result.get("page", 1)
            )
        except Exception as e:
            logger.error(f"MAF RAG audit failed: {e}")
            # Reconstruct inventory and fall back to base regex MAF check
            inventory = []
            if files and file_types:
                for fname in files.keys():
                    dtype = file_types.get(fname, "Unclassified Document")
                    inventory.append(InventoryItem(fname, dtype, "Readable"))
            return super().validate_maf(inventory, files or {}, tender_id)

    def extract_make_model_rag(self, vector_store: Chroma) -> str:
        """Extracts the proposed make and model of the hardware/equipment semantically using local RAG."""
        query = "proposed hardware equipment desktop PC switch make model manufacturer brand name model number"
        docs = self._similarity_search_filtered(vector_store, query, k=1, doc_type="Technical Datasheet / Bid")
        if not docs:
            docs = self._similarity_search_filtered(vector_store, query, k=1)
        
        if not docs:
            return "Not Given"
            
        context = self._build_context_from_docs(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a technical procurement auditor. Identify the manufacturer brand (make) and model name of the hardware/equipment proposed by the vendor from the context below.
        
        Review the context below:
        ---
        {context}
        ---
        
        Respond with ONLY the Make and Model (e.g., 'HP ProDesk 2 G10 Tower' or 'CYNEX GenX (IQ3030 TOWER DESKTOP)'). 
        Do not include any introductions, reasoning, explanations, or quotes. If not mentioned, respond with 'Not Given'.
        """)
        
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"context": context})
            val = resp.content.strip().strip("'\"")
            if len(val) > 100 or "here is" in val.lower() or "context" in val.lower():
                return "Given (As per Datasheet)"
            return val if val else "Not Given"
        except Exception as e:
            logger.error(f"Make and model extraction failed: {e}")
            return "Not Given"

    def extract_commercial_details_rag(self, vector_store: Chroma) -> str:
        """Extracts completed GeM contracts and work order details (numbers, values, dates) using local RAG."""
        query = "completed contracts work orders GeM contract number completion certificate invoice value date"
        docs = self._similarity_search_filtered(vector_store, query, k=2, doc_type="Experience / Past Performance Certificate")
        if not docs:
            docs = self._similarity_search_filtered(vector_store, query, k=2)
            
        if not docs:
            return "No Experience Documents found"
            
        context = self._build_context_from_docs(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a commercial procurement auditor. Analyze the experience certificates/contracts below and extract details of the completed contracts.
        For each contract, list:
        1. Contract/Order Number (e.g., GEMC-511687706445555)
        2. Date (e.g., 29-Sep-2023)
        3. Value (e.g., INR 10.67 Lakhs or 1.54 Crore)
        
        Review the context below:
        ---
        {context}
        ---
        
        Format each contract on a new line (e.g., '1. GEMC-511687706445555 dtd 29-Sep-2023 of INR 10.67 Lakhs').
        If multiple contracts are found, list them. If no completion certificates or work orders are present, respond with 'No completed work orders found'.
        Respond with ONLY the listed contract details. No intro, no conversational text.
        """)
        
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"context": context})
            return resp.content.strip()
        except Exception as e:
            logger.error(f"Commercial details extraction failed: {e}")
            return "Not Given"

    def evaluate_pqc_rag(self, vector_store: Chroma, pqc_reqs: List[Dict[str, Any]]) -> List[PQCResult]:
        """Audits Pre-Qualification Criteria (PQC) using local RAG (batched in a single LLM call to save CPU time)."""
        if not pqc_reqs:
            return []

        # 1. Gather context docs for all PQC requirements
        combined_docs = []
        params_str = ""
        req_mapping = {}

        for req in pqc_reqs:
            key = req["key"]
            label = req["label"]
            threshold = req.get("threshold")
            unit = req.get("unit", "")
            
            # Formulate queries and map keys
            if key == "experience":
                query = "years of experience supplying installing networking equipment PSU Government completion certificate"
                requirement_str = f"≥ {int(threshold)} {unit}"
                doc_type = "Experience / Past Performance Certificate"
            elif key == "turnover":
                query = "audited balance sheet annual financial turnover profit and loss statement Crore"
                requirement_str = f"≥ INR {threshold:g} Crore"
                doc_type = "Audited Balance Sheet"
            else:
                query = f"{label} compliance requirement documentation verification certificate"
                requirement_str = f"{threshold} {unit}" if threshold is not None else "Required"
                doc_type = None

            req_mapping[key] = {
                "req": req,
                "requirement_str": requirement_str
            }
            params_str += f"- Parameter Key: \"{key}\", Label: \"{label}\", Required: {requirement_str}\n"

            # Search across relevant doc_type chunks to save CPU cycles
            docs = self._similarity_search_filtered(vector_store, query, k=2, doc_type=doc_type)
            combined_docs.extend(docs)

        # 2. Build joint context
        context = self._build_context_from_docs(combined_docs)
        if len(context) > 15000:
            context = context[:15000] + "\n... [Context truncated for brevity] ..."

        prompt = ChatPromptTemplate.from_template("""
        You are a PSU procurement auditor. Your job is to verify if the vendor satisfies the Pre-Qualification Criteria (PQC) requirements.
        
        Here are the parameters to check:
        {parameters}
        
        Review the context below:
        ---
        {context}
        ---
        
        For each parameter in the input list, evaluate if the vendor satisfies the requirement.
        Determine the actual value/status offered by the vendor and compare it to the required threshold.
        
        Return a JSON object containing a single key 'pqc' which maps to a list. Each item in the list must represent one PQC parameter and have the following keys:
        - "key": (exact key from the input list, e.g. "experience" or "turnover")
        - "provided": (string description of what they actually provided, e.g. '14 years of experience' or 'INR 1,240 Crore average turnover')
        - "passed": (true/false, whether the provided value satisfies the requirement)
        - "source_file": (string filename of the certificate/balance sheet)
        - "page": (integer page number of the certificate/balance sheet)
        
        Important: Output ONLY the raw JSON block. No explanation.
        """)

        try:
            chain = prompt | self.llm
            resp = chain.invoke({
                "parameters": params_str,
                "context": context
            })
            res = parse_json_safely(resp.content)

            pqc_map = {}
            for item in res.get("pqc", []):
                if isinstance(item, dict) and "key" in item:
                    pqc_map[str(item["key"]).lower().strip()] = item

            results = []
            for req in pqc_reqs:
                key = req["key"]
                label = req["label"]
                section = req.get("section", "")
                map_info = req_mapping[key]
                requirement_str = map_info["requirement_str"]

                # Setup target bid positions
                bid_file, bid_page = "", 1
                bid_pos = self._find_bid_pqc_pos(key)
                if bid_pos is not None:
                    bid_file, bid_page = self._find_file_and_page_for_bid_match(bid_pos)

                llm_item = pqc_map.get(key.lower().strip())
                if llm_item:
                    results.append(PQCResult(
                        label=label,
                        required=requirement_str,
                        provided=llm_item.get("provided", "[NOT FOUND]"),
                        passed=bool(llm_item.get("passed", False)),
                        section=section,
                        file=llm_item.get("source_file", ""),
                        page=llm_item.get("page", 1),
                        bid_file=bid_file,
                        bid_page=bid_page
                    ))
                else:
                    # Fallback to individual check if missing from batch response
                    results.append(self._evaluate_single_pqc_rag(vector_store, req))
            return results

        except Exception as e:
            logger.error(f"Batch PQC extraction failed: {e}. Falling back to individual extraction.")
            # Fall back to individual checks
            results = []
            for req in pqc_reqs:
                results.append(self._evaluate_single_pqc_rag(vector_store, req))
            return results

    def _evaluate_single_pqc_rag(self, vector_store: Chroma, req: Dict[str, Any]) -> PQCResult:
        """Audits a single Pre-Qualification Criteria (PQC) using local RAG (internal fallback helper)."""
        key = req["key"]
        label = req["label"]
        threshold = req.get("threshold")
        unit = req.get("unit", "")
        section = req.get("section", "")
        
        if key == "experience":
            query = "years of experience supplying installing networking equipment PSU Government completion certificate"
            requirement_str = f"≥ {int(threshold)} {unit}"
            doc_type = "Experience / Past Performance Certificate"
        elif key == "turnover":
            query = "audited balance sheet annual financial turnover profit and loss statement Crore"
            requirement_str = f"≥ INR {threshold:g} Crore"
            doc_type = "Audited Balance Sheet"
        else:
            query = f"{label} compliance requirement documentation verification certificate"
            requirement_str = f"{threshold} {unit}" if threshold is not None else "Required"
            doc_type = None

        docs = []
        if doc_type:
            # Try to retrieve all pages of this specific credential document
            try:
                res = vector_store.get(where={"doc_type": doc_type})
                contents = res.get("documents", []) or []
                metadatas = res.get("metadatas", []) or []
                docs = [Document(page_content=c, metadata=m) for c, m in zip(contents, metadatas)]
            except Exception:
                pass
                
        if not docs:
            # Fall back to similarity search with a high k to capture all pages
            docs = self._similarity_search_filtered(vector_store, query, k=8, doc_type=doc_type)
            
        context = self._build_context_from_docs(docs)
        if len(context) > 15000:
            context = context[:15000] + "\n... [Context truncated for brevity] ..."

        prompt = ChatPromptTemplate.from_template("""
        You are a PSU procurement auditor. Auditing PQC Parameter: "{label}" (Requirement: {requirement_str}).
        
        Review the context below:
        ---
        {context}
        ---
        
        Evaluate if the vendor satisfies this requirement.
        Find the actual value/status offered by the vendor for the parameter "{label}" and compare it to the required "{requirement_str}".
        
        Return a JSON object with:
        {{
            "provided": "Description of what they actually provided (e.g. '14 years of experience' or 'INR 1,240 Crore average turnover')",
            "passed": true/false (whether provided satisfies the threshold/requirement),
            "source_file": "Filename of the certificate/balance sheet",
            "page": 1 (integer page number of the certificate/balance sheet)
        }}
        
        Important: Output ONLY the raw JSON block. No markdown, no extra text.
        """)

        chain = prompt | self.llm
        
        try:
            try:
                resp = chain.invoke({
                    "label": label,
                    "requirement_str": requirement_str,
                    "context": context
                })
            except Exception as api_err:
                # Self-healing retry for 413 Payload Too Large on PQC audit
                if "413" in str(api_err) or "too large" in str(api_err).lower():
                    logger.warning("Groq/LLM returned 413 Payload Too Large on PQC audit. Self-healing: retrying with reduced context...")
                    reduced_context = context[:12000]
                    resp = chain.invoke({
                        "label": label,
                        "requirement_str": requirement_str,
                        "context": reduced_context
                    })
                else:
                    raise api_err
                    
            res = parse_json_safely(resp.content)
            
            bid_file, bid_page = "", 1
            bid_pos = self._find_bid_pqc_pos(key)
            if bid_pos is not None:
                bid_file, bid_page = self._find_file_and_page_for_bid_match(bid_pos)

            return PQCResult(
                label=label,
                required=requirement_str,
                provided=res.get("provided", "[NOT FOUND]"),
                passed=bool(res.get("passed", False)),
                section=section,
                file=res.get("source_file", ""),
                page=res.get("page", 1),
                bid_file=bid_file,
                bid_page=bid_page
            )
        except Exception as e:
            logger.error(f"Single PQC RAG audit failed for {key}: {e}", exc_info=True)
            # Fall back to base regex checker
            try:
                raw_docs = vector_store.get()
                doc_text = "\n\n".join(raw_docs["documents"]) if raw_docs.get("documents") else ""
            except Exception:
                doc_text = ""
            
            fallback_results = super().evaluate_pqc([req], doc_text, True)
            if fallback_results:
                return fallback_results[0]
                
            # If the base checker returned nothing (e.g. for custom parameters), safely construct a default PQCResult
            bid_file, bid_page = "", 1
            bid_pos = self._find_bid_pqc_pos(key)
            if bid_pos is not None:
                bid_file, bid_page = self._find_file_and_page_for_bid_match(bid_pos)
                
            # Simple keyword search fallback
            passed = False
            provided = "[NOT FOUND]"
            file_name, page_num = "", 1
            if doc_text and key:
                pos = doc_text.lower().find(key.lower().replace("_", " "))
                if pos == -1:
                    pos = doc_text.lower().find(label.lower())
                if pos != -1:
                    file_name, page_num = find_file_and_page_for_match(doc_text, pos)
                    provided = "Found in submission"
                    passed = True
                    
            return PQCResult(
                label=label,
                required=requirement_str,
                provided=provided,
                passed=passed,
                section=section,
                file=file_name,
                page=page_num,
                bid_file=bid_file,
                bid_page=bid_page
            )

    def extract_spec_rag(self, vector_store: Chroma, spec: Dict[str, Any], mandatory: bool) -> SpecResult:
        """Extracts and audits a specific technical parameter using local RAG."""
        label = spec["label"]
        op = spec["op"]
        required = spec.get("required_value", spec.get("bid_value", True))
        unit = spec.get("unit", "")
        
        # Search queries
        query = f"Technical specifications proposed model parameter value: {label}"
        docs = self._similarity_search_filtered(vector_store, query, k=2, doc_type="Technical Datasheet / Bid")
        context = self._build_context_from_docs(docs)

        prompt = ChatPromptTemplate.from_template("""
        You are a PSU procurement auditor auditing technical parameter compliance:
        Parameter: "{label}"
        Required Value: {required} {unit}
        
        Review the context below:
        ---
        {context}
        ---
        
        Find the value offered by the vendor for this parameter.
        Grade compliance:
        - "match": The vendor's value fully meets or exceeds the requirement.
        - "fail": The vendor's value fails to meet the requirement.
        - "lacking": The vendor context does not contain information about this parameter.
        
        Return a JSON object with:
        {{
            "provided": "Description of what they offer (e.g. '1000 Gbps aggregate' or 'Generic L3 Switch')",
            "status": "match" or "fail" or "lacking",
            "source_file": "Filename containing the spec",
            "page": 1 (integer page number of the spec)
        }}
        
        Important: Output ONLY the raw JSON block. No markdown, no extra text.
        """)

        chain = prompt | self.llm
        
        # Setup target bid positions
        bid_file, bid_page = "", 1
        bid_pos = self._find_bid_pos(spec)
        if bid_pos is not None:
            bid_file, bid_page = self._find_file_and_page_for_bid_match(bid_pos)

        try:
            resp = chain.invoke({
                "label": label,
                "required": required,
                "unit": unit,
                "context": context
            })
            res = parse_json_safely(resp.content)
            
            provided_val = res.get("provided", "[DATA LACKING]")
            status = res.get("status", "lacking")
            page_info = f" (Pg {res.get('page', 1)})" if status != "lacking" else ""
            
            return SpecResult(
                param=label,
                required=self._fmt(required, unit) if op != "bool" else "Required",
                provided=provided_val + page_info,
                status=status,
                mandatory=mandatory,
                file=res.get("source_file", ""),
                page=res.get("page", 1),
                bid_file=bid_file,
                bid_page=bid_page
            )
        except Exception as e:
            logger.error(f"Spec RAG audit failed for {label}: {e}")
            return SpecResult(
                param=label,
                required=str(required),
                provided="[RAG ERROR]",
                status="lacking",
                mandatory=mandatory,
                bid_file=bid_file,
                bid_page=bid_page
            )

    def extract_all_specs_rag(self, vector_store: Chroma, specs: List[Dict[str, Any]], mandatory: bool) -> List[SpecResult]:
        """Extracts and audits a list of technical parameters in a single LLM call to save time."""
        if not specs:
            return []
            
        labels = [s["label"] for s in specs]
        query = "Technical datasheet specifications: " + ", ".join(labels[:5])
        
        # Retrieve all pages of Technical Datasheet / Bid if possible to ensure we don't miss specs
        docs = []
        try:
            res = vector_store.get(where={"doc_type": "Technical Datasheet / Bid"})
            contents = res.get("documents", []) or []
            metadatas = res.get("metadatas", []) or []
            docs = [Document(page_content=c, metadata=m) for c, m in zip(contents, metadatas)]
        except Exception:
            pass
            
        if not docs:
            # Fall back to similarity search with a high k to capture all pages
            docs = self._similarity_search_filtered(vector_store, query, k=8, doc_type="Technical Datasheet / Bid")
            
        context = self._build_context_from_docs(docs)
        if len(context) > 16000:
            context = context[:16000] + "\n... [Context truncated for brevity] ..."
        
        prompt = ChatPromptTemplate.from_template("""
        You are a technical compliance auditor. Your job is to verify if the vendor's technical datasheet complies with the required technical specifications.
        
        Here are the parameters to check:
        {parameters}
        
        Review the technical datasheet context below:
        ---
        {context}
        ---
        
        For each parameter in the input list, extract:
        1. What value the vendor provided for this parameter.
        2. The compliance status:
           - "match": if the vendor's value meets or exceeds the requirement.
           - "fail": if the vendor's value fails to meet the requirement.
           - "lacking": if the context doesn't mention this parameter.
        
        Return a JSON object containing a single key 'specs' which maps to a list. Each item in the list must represent one parameter and have the following keys:
        - "label": (exact parameter name from the input list)
        - "provided": (string description of the vendor's value)
        - "status": ("match", "fail", or "lacking")
        - "source_file": (string filename where you found this information)
        - "page": (integer page number where you found this information)
        
        Important: Output ONLY the raw JSON block. No explanation.
        """)
        
        # Format parameters for the prompt
        params_str = ""
        for s in specs:
            req_val = s.get("required_value", s.get("bid_value", True))
            params_str += f"- Parameter: \"{s['label']}\", Operator: \"{s['op']}\", Required: {req_val} {s.get('unit', '')}\n"
            
        try:
            chain = prompt | self.llm
            try:
                resp = chain.invoke({
                    "parameters": params_str,
                    "context": context
                })
            except Exception as api_err:
                # Self-healing retry for 413 Payload Too Large on specs audit
                if "413" in str(api_err) or "too large" in str(api_err).lower():
                    logger.warning("Groq/LLM returned 413 Payload Too Large on specs audit. Self-healing: retrying with reduced context...")
                    reduced_context = context[:12000]
                    resp = chain.invoke({
                        "parameters": params_str,
                        "context": reduced_context
                    })
                else:
                    raise api_err
                    
            res = parse_json_safely(resp.content)
            
            # Map LLM results back to SpecResult objects
            spec_map = {}
            for item in res.get("specs", []):
                if isinstance(item, dict) and "label" in item:
                    spec_map[str(item["label"]).lower().strip()] = item
                    
            results = []
            for s in specs:
                label = s["label"]
                op = s["op"]
                required = s.get("required_value", s.get("bid_value", True))
                unit = s.get("unit", "")
                
                # Setup target bid positions
                bid_file, bid_page = "", 1
                bid_pos = self._find_bid_pos(s)
                if bid_pos is not None:
                    bid_file, bid_page = self._find_file_and_page_for_bid_match(bid_pos)
                
                llm_item = spec_map.get(label.lower().strip())
                if llm_item:
                    provided_val = llm_item.get("provided", "[DATA LACKING]")
                    status = llm_item.get("status", "lacking")
                    page_info = f" (Pg {llm_item.get('page', 1)})" if status != "lacking" else ""
                    
                    results.append(SpecResult(
                        param=label,
                        required=self._fmt(required, unit) if op != "bool" else "Required",
                        provided=provided_val + page_info,
                        status=status,
                        mandatory=mandatory,
                        file=llm_item.get("source_file", ""),
                        page=llm_item.get("page", 1),
                        bid_file=bid_file,
                        bid_page=bid_page
                    ))
                else:
                    # Fallback to individual check if missing from batch response
                    results.append(self.extract_spec_rag(vector_store, s, mandatory))
            return results
        except Exception as e:
            logger.error(f"Batch spec extraction failed: {e}. Falling back to individual extraction.")
            results = []
            for s in specs:
                results.append(self.extract_spec_rag(vector_store, s, mandatory))
            return results

    def detect_deviations_rag(self, vector_store: Chroma) -> List[str]:
        """Identifies vendor deviations semantically using local RAG."""
        query = "deviation statement cannot supply instead we will provide not supported alternative exception"
        docs = self._similarity_search_filtered(vector_store, query, k=2, doc_type="Deviation Statement")
        context = self._build_context_from_docs(docs)

        prompt = ChatPromptTemplate.from_template("""
        You are a technical compliance auditor. Identify any explicit deviations or limitations proposed by the vendor.
        Look for statements where they specify they cannot meet a parameter, offer an alternative, or mention a "deviation".
        
        Review the context below:
        ---
        {context}
        ---
        
        Return a JSON list of deviation descriptions. If none are found, return an empty list.
        Each item in the list must be a string containing the deviation description, the filename, and the page number.
        Example item format: "We cannot supply 60 degC rated hardware; instead we will provide 45 degC. (tech_offer.pdf - Pg 2)"
        
        Return ONLY a JSON list (e.g. ["Deviation 1...", "Deviation 2..."]). No markdown wrapper, no extra text.
        """)

        chain = prompt | self.llm
        
        try:
            resp = chain.invoke({"context": context})
            deviations = parse_json_safely(resp.content)
            if isinstance(deviations, list):
                return [str(d) for d in deviations[:6]]
            return []
        except Exception as e:
            logger.error(f"Deviations RAG audit failed: {e}")
            return []

    def _validate_document_checklist_rag(self, vector_store: Chroma, mandatory_docs: List[str], inventory: List[InventoryItem], combined_text: str) -> Dict[str, Dict[str, Any]]:
        checklist_results = {}
        present_docs = {item.doc_type: item for item in inventory}
        import re
        from audit_engine import resolve_actual_document_page
        
        for doc in mandatory_docs:
            if doc in present_docs:
                item = present_docs[doc]
                filename = item.filename
                
                # Check for presence and details of the document
                docs = self._similarity_search_filtered(vector_store, f"Check for presence and details of required document: {doc}", k=1, doc_type=doc)
                page = resolve_actual_document_page(doc, combined_text, filename, mandatory_docs)
                finding = "Present and verified."
                status = "Compliant"
                if docs:
                    low_content = docs[0].page_content.lower()
                    if "insolvency" in doc.lower() and "undergoing" in low_content and "bankruptcy" in low_content:
                        status = "Non-Compliant"
                        finding = "Undergoing insolvency or bankruptcy proceedings."
                    elif "black listing" in doc.lower() and "blacklist" in low_content:
                        status = "Non-Compliant"
                        finding = "Declares history of blacklisting or holiday listing."
                    else:
                        finding = f"Document matches proforma requirements."
                
                checklist_results[doc] = {
                    "status": status,
                    "page": page,
                    "file": filename,
                    "finding": finding
                }
            else:
                # Dynamic Check: Search vector store to see if this document is embedded in another file (consolidated booklet)
                docs = self._similarity_search_filtered(vector_store, f"mandatory document: {doc}", k=1)
                
                is_embedded = False
                page = 1
                filename = ""
                finding = f"Mandatory document {doc} was not submitted."
                status = "Missing"
                
                if docs:
                    context = docs[0].page_content
                    low_context = context.lower()
                    
                    # Resilient PAN card pattern and keywords matching for noisy OCR
                    resilient_pan_pattern = re.compile(r'\b[A-Z]{4}[A-Z0-9][0-9]{4}[A-Z0-9]\b')
                    is_pan_page = (
                        resilient_pan_pattern.search(context) or
                        any(kw in low_context for kw in ["permanent account", "pan card", "income tax", "govt of india", "tax department", "par anenl", "acaunilnber"])
                    )
                    if doc == "PAN Card" and is_pan_page:
                        is_embedded = True
                        pan_match = resilient_pan_pattern.search(context)
                        finding = f"Present and verified. PAN Number: {pan_match.group(0)}." if pan_match else "Present and verified."
                    elif doc in ("GST Registration", "GST Certificate", "GSTIN") and (re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b', context) or "gstin" in low_context or "gst registration" in low_context):
                        is_embedded = True
                        gst_match = re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b', context)
                        finding = f"Present and verified. Extracted GSTIN: {gst_match.group(0)}." if gst_match else "Present and verified."
                    elif doc == "Audited Balance Sheet" and any(k in low_context for k in ["balance sheet", "profit & loss", "audited", "financial statements", "chartered accountant"]):
                        is_embedded = True
                        finding = "Present and verified. Financial statements audited."
                    elif "gem registration" in doc.lower() or "gem seller profile" in doc.lower() or "gem profile" in doc.lower():
                        gem_kws = ["gem registration", "gem seller", "gem profile", "gem portal", "marketplace", "gemc-"]
                        if any(kw in low_context for kw in gem_kws):
                            is_embedded = True
                            finding = "Present and verified. GeM registration profile active."
                    else:
                        # Ask the LLM to verify if it is present in the retrieved page
                        prompt = ChatPromptTemplate.from_template(
                            "Does the following document excerpt confirm the presence or submission of the required document '{doc}'?\n"
                            "Answer with a JSON object: {{\"present\": true/false, \"evidence\": \"exact quote\"}}\n\n"
                            "Excerpt:\n{context}"
                        )
                        chain = prompt | self.llm
                        try:
                            resp = chain.invoke({"doc": doc, "context": context})
                            res = parse_json_safely(resp.content)
                            if res.get("present"):
                                is_embedded = True
                                finding = res.get("evidence", "Present and verified.")
                        except Exception:
                            pass
                            
                if is_embedded:
                    status = "Compliant"
                    filename = docs[0].metadata.get("source", docs[0].metadata.get("file", ""))
                    if not filename:
                        # Fallback filename search
                        for item in inventory:
                            filename = item.filename
                            break
                    page = resolve_actual_document_page(doc, combined_text, filename, mandatory_docs)
                    
                checklist_results[doc] = {
                    "status": status,
                    "page": page,
                    "file": filename if filename else None,
                    "finding": finding
                }
        return checklist_results

    def analyze_vendor(self, name: str, files: Dict[str, str],
                       errors: Dict[str, Optional[str]], bid: Dict[str, Any]) -> VendorResult:
        """Runs the entire vendor audit using a fast Regex-First, RAG-Fallback engine."""
        if not self.has_rag_backend:
            raise Exception("RAG backend is not initialized. Please ensure Ollama is running or switch to Cloud RAG (Gemini).")

        write_engine_log(f"[RAG Engine] Starting Regex-First compliance audit for: {name}")
        result = VendorResult(name=name)

        # 1. Inventory & classification
        write_engine_log(f"[RAG Engine] [{name}] Step 1: Classifying all vendor documents...")
        file_types = self.classify_all_documents(files, bid)
        for fname, text in files.items():
            doc_type = file_types.get(fname, "Unclassified Document")
            readability = self.assess_readability(text, errors.get(fname))
            result.inventory.append(InventoryItem(fname, doc_type, readability))
            write_engine_log(f"  - Document: {fname} -> Classified as: {doc_type} (Readability: {readability})")

        present_types = {i.doc_type for i in result.inventory}
        has_gem_doc = "GeM Registration" in present_types

        # Build combined text for regex check
        combined_text_parts = []
        for fname, text in files.items():
            combined_text_parts.append(f"--- FILE {fname} ---\n{text}")
        combined_text = "\n\n".join(combined_text_parts)

        # 2. Build local Chroma vector database in memory
        write_engine_log(f"[RAG Engine] [{name}] Step 2: Building local vector store in memory...")
        vector_store = self.build_vector_store(files, file_types=file_types)
        if not vector_store:
            if self.mode == "gemini":
                raise Exception("Failed to build local Chroma vector store for vendor. This usually happens when the Gemini API key is invalid or there is a network issue.")
            elif self.mode == "groq":
                raise Exception("Failed to build local Chroma vector store for vendor. If using Gemini Embeddings, check your Gemini API key. If using Local Embeddings, ensure Ollama is running.")
            else:
                raise Exception("Failed to build local Chroma vector store for vendor. This usually happens when Ollama runs out of memory or crashes.")

        try:
            # Step 3: MAF Gate (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 3: Evaluating MAF (Manufacturer's Authorization Form)...")
            regex_maf = super().validate_maf(result.inventory, files, bid.get("tender_id", ""))
            if regex_maf.status == "Found (Valid)":
                result.maf = regex_maf
                write_engine_log(f"  - MAF verified compliant via deterministic base rules.")
            else:
                write_engine_log(f"  - MAF base check: {regex_maf.evidence}. Falling back to semantic RAG verification...")
                try:
                    result.maf = self.validate_maf_rag(vector_store, bid.get("tender_id", ""), files, file_types)
                except Exception as maf_err:
                    logger.error(f"MAF RAG validation failed: {maf_err}. Using base regex verdict.")
                    result.maf = regex_maf
            write_engine_log(f"  - MAF Status: {result.maf.status} (Details: {result.maf.evidence})")

            # Step 10: Document checklist validation (run early to detect embedded docs like GeM Registration!)
            write_engine_log(f"[RAG Engine] [{name}] Step 10: Validating mandatory documents checklist...")
            result.document_checklist = self._validate_document_checklist_rag(vector_store, bid.get("mandatory_docs", []), result.inventory, combined_text)

            # Calculate has_gem_doc dynamically (checking both inventory and document checklist)
            has_gem_doc = ("GeM Registration" in present_types or result.document_checklist.get("GeM Registration", {}).get("status") == "Compliant")

            # Step 4: PQC (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 4: Evaluating Pre-Qualification Criteria (PQC)...")
            regex_pqcs = super().evaluate_pqc(bid.get("pqc", []), combined_text, has_gem_doc)
            
            unresolved_pqcs = []
            pqc_results_map = {}
            for req, p_res in zip(bid.get("pqc", []), regex_pqcs):
                if p_res.passed:
                    pqc_results_map[req["key"]] = p_res
                else:
                    unresolved_pqcs.append(req)

            if unresolved_pqcs:
                write_engine_log(f"  - PQC: {len(unresolved_pqcs)} parameters unresolved by regex. Querying semantic RAG fallback...")
                try:
                    rag_pqcs = self.evaluate_pqc_rag(vector_store, unresolved_pqcs)
                    for r_res, req in zip(rag_pqcs, unresolved_pqcs):
                        pqc_results_map[req["key"]] = r_res
                except Exception as pqc_err:
                    logger.error(f"Batch PQC RAG validation failed: {pqc_err}. Using base regex verdicts.")
                    for req, p_res in zip(bid.get("pqc", []), regex_pqcs):
                        if req["key"] not in pqc_results_map:
                            pqc_results_map[req["key"]] = p_res

            # Safety fallback: Ensure every single PQC requirement key is populated in the map.
            # If any key is missing (e.g. LLM failed to return a value or returned mismatched keys), fall back to the base regex result.
            for req, p_res in zip(bid.get("pqc", []), regex_pqcs):
                if req["key"] not in pqc_results_map:
                    pqc_results_map[req["key"]] = p_res

            # Keep original bid order for PQC results
            result.pqc = [pqc_results_map[req["key"]] for req in bid.get("pqc", [])]
            for item in result.pqc:
                write_engine_log(f"  - PQC [{item.label}] Passed: {item.passed} (Required: {item.required}, Provided: {item.provided})")

            # Step 5: Mandatory specs (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 5: Extracting and auditing mandatory technical specifications...")
            mand_specs_results = []
            unresolved_mand_specs = []
            unresolved_mand_indices = []

            for i, spec in enumerate(bid.get("mandatory_specs", [])):
                res_spec = super().extract_spec(spec, combined_text, True)
                mand_specs_results.append(res_spec)
                if res_spec.status in ("lacking", "fail"):
                    unresolved_mand_specs.append(spec)
                    unresolved_mand_indices.append(i)

            if unresolved_mand_specs:
                write_engine_log(f"  - Specs: {len(unresolved_mand_specs)} mandatory parameters unresolved by regex. Querying semantic RAG fallback...")
                try:
                    rag_specs = self.extract_all_specs_rag(vector_store, unresolved_mand_specs, True)
                    for r_spec, idx in zip(rag_specs, unresolved_mand_indices):
                        mand_specs_results[idx] = r_spec
                except Exception as spec_err:
                    logger.error(f"Mandatory specs RAG extraction failed: {spec_err}.")

            result.mandatory_specs = mand_specs_results
            for item in result.mandatory_specs:
                write_engine_log(f"  - Spec [{item.param}] Status: {item.status} (Required: {item.required}, Provided: {item.provided})")

            # Step 6: Preferred specs (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 6: Extracting and auditing preferred technical specifications...")
            pref_specs_results = []
            unresolved_pref_specs = []
            unresolved_pref_indices = []

            for i, spec in enumerate(bid.get("preferred_specs", [])):
                res_spec = super().extract_spec(spec, combined_text, False)
                pref_specs_results.append(res_spec)
                if res_spec.status in ("lacking", "fail"):
                    unresolved_pref_specs.append(spec)
                    unresolved_pref_indices.append(i)

            if unresolved_pref_specs:
                write_engine_log(f"  - Specs: {len(unresolved_pref_specs)} preferred parameters unresolved by regex. Querying semantic RAG fallback...")
                try:
                    rag_specs = self.extract_all_specs_rag(vector_store, unresolved_pref_specs, False)
                    for r_spec, idx in zip(rag_specs, unresolved_pref_indices):
                        pref_specs_results[idx] = r_spec
                except Exception as spec_err:
                    logger.error(f"Preferred specs RAG extraction failed: {spec_err}.")

            result.preferred_specs = pref_specs_results

            # Step 7: Technical deviations (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 7: Detecting technical deviations...")
            regex_devs = super().detect_deviations(combined_text)
            if regex_devs:
                result.deviations = regex_devs
            else:
                try:
                    result.deviations = self.detect_deviations_rag(vector_store)
                except Exception as dev_err:
                    logger.error(f"Deviations RAG check failed: {dev_err}")
                    result.deviations = []
            if result.deviations:
                write_engine_log(f"  - Found {len(result.deviations)} deviations/violations.")

            # Step 8: Make/Model (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 8: Extracting Make/Model details...")
            make_model = "Not Given"
            model_match = re.search(r"(?:make|model|manufacturer|brand)\s*[:\-]\s*([a-z0-9][a-z0-9\s\-_]{3,25})", combined_text, re.I)
            if model_match:
                make_model = model_match.group(1).strip()
            else:
                if result.maf and result.maf.status == "Found (Valid)":
                    m = re.search(r"authorization\s*from\s*([a-z0-9\s\-]+)", result.maf.evidence, re.I)
                    if m:
                        make_model = f"Given ({m.group(1).strip()})"
                    else:
                        make_model = "Given (As per MAF)"
                else:
                    try:
                        make_model = self.extract_make_model_rag(vector_store)
                    except Exception:
                        make_model = "Given (As per Datasheet)"
            result.make_model = make_model
            write_engine_log(f"  - Make/Model: {result.make_model}")

            # Step 9: Commercial details (Regex-First, RAG-Fallback)
            write_engine_log(f"[RAG Engine] [{name}] Step 9: Extracting commercial details...")
            contracts = []
            gemc_matches = re.findall(r"(GEMC-[A-Z0-9\-]{10,20})", combined_text, re.I)
            for i, contract in enumerate(list(set(gemc_matches))[:4], start=1):
                pos = combined_text.find(contract)
                snippet = combined_text[max(0, pos-100):min(len(combined_text), pos+300)]
                date_match = re.search(r"(\d{1,2}[/\-\.][a-z0-9]{3,9}[/\-\.]\d{2,4})", snippet, re.I)
                val_match = re.search(r"(?:inr|rs\.?)\s*([\d,\.]+)\s*(?:lakh|crore|crores|lakhs)", snippet, re.I)
                date_str = f" dtd {date_match.group(1)}" if date_match else ""
                val_str = f" of {val_match.group(0)}" if val_match else ""
                contracts.append(f"{i}. {contract}{date_str}{val_str}")
            
            if contracts:
                result.commercial_details = "\n".join(contracts)
            else:
                try:
                    result.commercial_details = self.extract_commercial_details_rag(vector_store)
                except Exception:
                    result.commercial_details = "No completed work orders found"

            # Step 11: Missing mandatory documents check
            for doc in bid.get("mandatory_docs", []):
                checklist_entry = result.document_checklist.get(doc, {})
                if checklist_entry.get("status", "Missing") == "Missing":
                    result.missing_documents.append(doc)

            # Cross-document dependency and exemption audits (Partnership deed, consortium JV, EMD exemption, MII, etc.)
            self.evaluate_document_dependencies_and_exemptions(result, bid, combined_text, vector_store)

            # Step 12: Disqualification checks
            self._apply_disqualification_gate(result, bid)
            write_engine_log(f"[RAG Engine] [{name}] Vendor audit completed. Status: {result.status}")

            # Step 13: Scoring and summaries
            self._score(result)
            result.summary = self._summarize(result)

        finally:
            # Chroma deletes vector collections automatically in-memory,
            # but we explicitly clear references to prevent leaks.
            del vector_store

        return result

    def _find_file_and_page_for_match(self, pos: int) -> Tuple[str, int]:
        """Finds source file and page for a specific string match position in master text."""
        return find_file_and_page_for_match(self.bid_text, pos)

    def _find_file_and_page_for_bid_match(self, pos: int) -> Tuple[str, int]:
        """Wrapper helper to call top-level file page function."""
        return find_file_and_page_for_match(self.bid_text, pos)

    def evaluate_document_dependencies_and_exemptions(self, r: VendorResult, bid: Dict[str, Any], combined_text: str, vector_store: Any = None) -> None:
        if not self.has_rag_backend:
            super().evaluate_document_dependencies_and_exemptions(r, bid, combined_text, vector_store)
            return

        write_engine_log(f"[RAG Engine] [{r.name}] Dynamically checking document dependencies and exemptions via RAG...")
        
        # 1. Retrieve relevant pages/clauses from the entire bidder's documents using vector similarity search
        query = (
            "partnership deed partnership firm partners deed of partnership "
            "power of attorney poa authorized signatory authorized partner "
            "joint venture consortium agreement MoU lead member "
            "emd exemption bid security exemption earnest money deposit "
            "make in india local content Class-I Class-II chartered accountant ca certificate"
        )
        
        retrieved_context = ""
        if vector_store:
            try:
                # Retrieve the top 6 most relevant pages/sections from anywhere in the vendor's files
                docs = self._similarity_search_filtered(vector_store, query, k=6)
                retrieved_context = self._build_context_from_docs(docs)
            except Exception as search_err:
                logger.warning(f"RAG search for document dependencies failed: {search_err}")

        # Build prompt context: bidder's checklist, inventory, first page info, and retrieved RAG context
        checklist_summary = json.dumps(r.document_checklist, indent=2)
        inventory_summary = ", ".join([f"{i.filename} ({i.doc_type})" for i in r.inventory])
        
        # We take the first 8000 characters as the general preview, and append the RAG context
        preview = combined_text[:8000]
        context = f"--- General Bid Information (First few pages) ---\n{preview}\n\n--- Retrieved Relevant Document Sections (Search results from anywhere in files) ---\n{retrieved_context}"
        
        prompt = ChatPromptTemplate.from_template("""
        You are a senior government procurement officer auditing a bidder's tender submission.
        Your task is to dynamically analyze document dependencies and exemptions based on:
        1. Explicit rules or mandatory documents in the tender: {mandatory_docs}
        2. Standard Indian Government Public Procurement Guidelines (GFR 2017, MSME Procurement Policy, GeM Guidelines).
        
        Standard Government Document Dependency Rules:
        - EMD Exemption: Micro & Small Enterprises (MSEs) claiming EMD exemption must submit a valid Udyam Registration (MSME) or NSIC certificate.
        - Partnership Firm: Partnership firms must submit the Partnership Deed and a Power of Attorney (PoA) for the signing partner.
        - Consortium / JV: Consortium bids must submit a JV Agreement/MoU and a Power of Attorney for the Lead Member.
        - Make in India (MII): Local content preference claims for tenders > 10 Crores require a CA/Auditor certificate (self-certification is rejected).
        
        Exemptions:
        - A valid Udyam Registration Certificate exempts Micro and Small Enterprises from submitting a Manufacturer's Authorization Form (MAF) or EMD in government/PSU tenders.
        
        Bidder Name: {bidder_name}
        Tender ID: {tender_id}
        
        Bidder Submission Context:
        ---
        - Submitted Files: {inventory}
        - Current Document Checklist Status:
        {checklist}
        
        - First few pages of Bidder's Documents (for verification of signatures, PAN card status, or consortium/MII claims) plus relevant retrieved sections:
        {context}
        ---
        
        Identify:
        1. Bidder Profile: Are they a partnership firm, consortium, claiming MSME/Udyam status, claiming EMD exemption, or claiming MII local content preference?
        2. Exemptions: Do they qualify for any exemptions (e.g. Udyam waiving EMD or MAF)?
        3. Violations: Are there any missing supporting documents required for their profile under GFR/Tender rules?
        
        Return a JSON object with:
        {{
            "profile": "Short summary of bidder's profile",
            "exemptions": [
                {{
                    "doc": "Manufacturer's Authorization Form (MAF)" or "EMD",
                    "status": "Exempted" or "Required",
                    "reason": "Exemption explanation"
                }}
            ],
            "violations": [
                {{
                    "title": "Violation Title",
                    "rule": "The tender or government rule violated",
                    "evidence": "Detail of the missing or non-compliant document"
                }}
            ]
        }}
        
        Respond with ONLY the raw JSON block. No explanation.
        """)
        
        chain = prompt | self.llm
        
        try:
            resp = chain.invoke({
                "mandatory_docs": ", ".join(bid.get("mandatory_docs", [])),
                "bidder_name": r.name,
                "tender_id": bid.get("tender_id", ""),
                "inventory": inventory_summary,
                "checklist": checklist_summary,
                "context": context
            })
            
            res = parse_json_safely(resp.content)
            
            # Apply exemptions dynamically from LLM response
            for ex in res.get("exemptions", []):
                doc = ex.get("doc")
                status = ex.get("status")
                reason = ex.get("reason", "")
                
                if doc == "Manufacturer's Authorization Form (MAF)" and status == "Exempted":
                    write_engine_log(f"  - Dynamic Exemption: Waiving MAF requirement for {r.name}. Reason: {reason}")
                    if r.maf:
                        r.maf.status = MAF_VALID
                        r.maf.evidence = f"Exempted via RAG audit: {reason}"
                    # Remove from missing documents list if present
                    if "Manufacturer's Authorization Form (MAF)" in r.missing_documents:
                        r.missing_documents.remove("Manufacturer's Authorization Form (MAF)")
                        
                elif (doc == "EMD" or "earnest money" in str(doc).lower()) and status == "Exempted":
                    write_engine_log(f"  - Dynamic Exemption: Waiving EMD requirement for {r.name}. Reason: {reason}")
                    # Remove from missing documents list if present
                    emd_names = [d for d in r.missing_documents if "emd" in d.lower() or "earnest money" in d.lower() or "bid security" in d.lower()]
                    for ed in emd_names:
                        r.missing_documents.remove(ed)
            
            # Apply violations dynamically from LLM response
            for viol in res.get("violations", []):
                title = viol.get("title", "")
                rule = viol.get("rule", "")
                evidence = viol.get("evidence", "")
                r.violations.append(Violation(title, rule, evidence))
                write_engine_log(f"  - Dynamic Violation Detected: {title} ({rule})")
                
        except Exception as e:
            logger.warning(f"Dynamic RAG document dependency check failed: {e}. Falling back to base rules.")
            # Fall back to base deterministic python rules
            super().evaluate_document_dependencies_and_exemptions(r, bid, combined_text)

    def narrate(self, bid: Dict[str, Any], results: List[VendorResult]) -> Optional[str]:
        """Generates a plain-language executive narrative of the evaluation outcome using local Llama."""
        if not self.has_rag_backend:
            return None
        ranked = sorted([r for r in results if not r.disqualified],
                        key=lambda x: x.score, reverse=True)
        dq = [r for r in results if r.disqualified]
        ctx = {
            "tender_id": bid.get("tender_id"),
            "responsive": [{"name": r.name, "rank": r.rank, "score": r.score,
                            "summary": r.summary} for r in ranked],
            "disqualified": [{"name": r.name,
                              "reasons": [v.title for v in r.violations]} for r in dq],
        }
        
        prompt = ChatPromptTemplate.from_template("""
        You are a PSU tender evaluation officer. Write a concise, formal executive summary (max 120 words) of the evaluation outcome.
        Be factual, cite ranks, scores, and the decisive reasons for disqualification or ranking.
        Do not invent data beyond what is provided in the JSON below.
        
        JSON evaluation data:
        ---
        {{ctx_json}}
        ---
        
        Return ONLY the summary text, no markdown styling, no introduction.
        """)
        
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"ctx_json": json.dumps(ctx, indent=2)})
            return resp.content.strip()
        except Exception as e:
            logger.error(f"Narration generation failed: {e}")
            return None



# Helper function from audit_engine duplicated for scope
def find_file_and_page_for_match(text: str, match_index: int) -> Tuple[str, int]:
    file_markers = list(re.finditer(r"--- FILE (.*?) ---", text[:match_index]))
    filename = ""
    if file_markers:
        filename = file_markers[-1].group(1).strip()
        start_idx = file_markers[-1].end()
        page_markers = list(re.finditer(r"--- PAGE (\d+) ---", text[start_idx:match_index]))
        if page_markers:
            return filename, int(page_markers[-1].group(1))
        return filename, 1
    page_markers = list(re.finditer(r"--- PAGE (\d+) ---", text[:match_index]))
    if page_markers:
        return "", int(page_markers[-1].group(1))
    return "", 1
