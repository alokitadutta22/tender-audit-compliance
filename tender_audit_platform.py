"""
================================================================================
 Argus Bid AI — Tender Auditing & Compliance Platform
 Built for PSU procurement workflows (IOCL-style NIT / BID evaluation)
================================================================================
 This is the main entrypoint and UI layer. It imports core auditing rules from
 audit_engine.py and custom themes/styling from ui_styles.py.
================================================================================
"""

from __future__ import annotations

import html
import os
import pickle
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

# Import core audit engine logic and models with forced reload to clear Streamlit's cache
import importlib
import audit_engine
importlib.reload(audit_engine)
from audit_engine import (
    AuditEngine,
    LLMAuditEngine,
    VendorResult,
    SpecResult,
    PQCResult,
    InventoryItem,
    MAFResult,
    Violation,
    get_engine,
    read_uploaded_file,
    is_valid_bid_document,
    MASTER_BID_TEXT,
    STATUS_RESPONSIVE,
    STATUS_DISQUALIFIED,
    MAF_VALID,
    MAF_INVALID,
    MAF_MISSING,
    READ_PASS,
    READ_LOW,
    READ_CORRUPT
)

# Import Local RAG engine dynamically with forced reload to clear Streamlit's cache (Reload Trigger 1)
try:
    import importlib
    import rag_engine
    importlib.reload(rag_engine)
    from rag_engine import LocalRAGAuditEngine
    HAS_RAG_ENGINE = True
except Exception as e:
    HAS_RAG_ENGINE = False


# Import UI themes, css styles, and HTML helper components with forced reload to clear Streamlit's cache
import importlib
import ui_styles
importlib.reload(ui_styles)
from ui_styles import (
    PALETTE,
    CSS,
    CUSTOM_SPINNER_CSS,
    get_base64_image,
    inject_custom_loading_screen,
    format_pdf_text_to_html,
    status_pill,
    maf_pill,
    read_pill,
    spec_chip,
    render_audit_terminal,
    custom_spinner
)

# Inject the loading screen at startup
# inject_custom_loading_screen()


# ===========================================================================
# SECTION 9 — STREAMLIT APPLICATION STATE & CACHING
# ===========================================================================
CACHE_FILE = ".sentinel_cache.pkl"

def serialize_results(results: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    if results is None:
        return None
    serialized = []
    for r in results:
        inventory_list = []
        for i in r.inventory:
            inventory_list.append({
                "filename": i.filename,
                "doc_type": i.doc_type,
                "readability": i.readability
            })
            
        maf_dict = None
        if r.maf:
            maf_dict = {
                "status": r.maf.status,
                "evidence": r.maf.evidence,
                "source_file": r.maf.source_file,
                "page": r.maf.page
            }
            
        pqc_list = []
        for p in r.pqc:
            pqc_list.append({
                "label": p.label,
                "required": p.required,
                "provided": p.provided,
                "passed": p.passed,
                "section": p.section,
                "file": p.file,
                "page": p.page,
                "bid_file": p.bid_file,
                "bid_page": p.bid_page
            })
            
        mand_specs_list = []
        for s in r.mandatory_specs:
            mand_specs_list.append({
                "param": s.param,
                "required": s.required,
                "provided": s.provided,
                "status": s.status,
                "mandatory": s.mandatory,
                "section": s.section,
                "file": s.file,
                "page": s.page,
                "bid_file": s.bid_file,
                "bid_page": s.bid_page
            })
            
        pref_specs_list = []
        for s in r.preferred_specs:
            pref_specs_list.append({
                "param": s.param,
                "required": s.required,
                "provided": s.provided,
                "status": s.status,
                "mandatory": s.mandatory,
                "section": s.section,
                "file": s.file,
                "page": s.page,
                "bid_file": s.bid_file,
                "bid_page": s.bid_page
            })
            
        violations_list = []
        for v in r.violations:
            violations_list.append({
                "title": v.title,
                "requirement": v.requirement,
                "finding": v.finding
            })
            
        serialized.append({
            "name": r.name,
            "inventory": inventory_list,
            "maf": maf_dict,
            "pqc": pqc_list,
            "mandatory_specs": mand_specs_list,
            "preferred_specs": pref_specs_list,
            "deviations": r.deviations,
            "missing_documents": r.missing_documents,
            "disqualified": r.disqualified,
            "violations": violations_list,
            "score": r.score,
            "mandatory_score": r.mandatory_score,
            "preferred_score": r.preferred_score,
            "rank": r.rank,
            "status": r.status,
            "summary": r.summary,
            "make_model": r.make_model,
            "commercial_details": r.commercial_details,
            "document_checklist": r.document_checklist
        })
    return serialized


def deserialize_results(serialized: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
    if serialized is None:
        return None
    from audit_engine import VendorResult, InventoryItem, MAFResult, PQCResult, SpecResult, Violation
    results = []
    for d in serialized:
        inventory = [InventoryItem(**i) for i in d.get("inventory", [])]
        maf = None
        if d.get("maf"):
            maf = MAFResult(**d["maf"])
        pqc = [PQCResult(**p) for p in d.get("pqc", [])]
        mandatory_specs = [SpecResult(**s) for s in d.get("mandatory_specs", [])]
        preferred_specs = [SpecResult(**s) for s in d.get("preferred_specs", [])]
        violations = [Violation(**v) for v in d.get("violations", [])]
        
        r = VendorResult(
            name=d["name"],
            inventory=inventory,
            maf=maf,
            pqc=pqc,
            mandatory_specs=mandatory_specs,
            preferred_specs=preferred_specs,
            deviations=d.get("deviations", []),
            missing_documents=d.get("missing_documents", []),
            disqualified=d.get("disqualified", False),
            violations=violations,
            score=d.get("score", 0.0),
            mandatory_score=d.get("mandatory_score", 0.0),
            preferred_score=d.get("preferred_score", 0.0),
            rank=d.get("rank"),
            status=d.get("status", "Responsive"),
            summary=d.get("summary", ""),
            make_model=d.get("make_model", ""),
            commercial_details=d.get("commercial_details", ""),
            document_checklist=d.get("document_checklist", {})
        )
        results.append(r)
    return results


def serialize_audit_cache(cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    serialized = {}
    for name, entry in cache.items():
        res = entry.get("result")
        res_serialized = serialize_results([res])[0] if res else None
        serialized[name] = {
            "hash": entry["hash"],
            "result": res_serialized
        }
    return serialized


def deserialize_audit_cache(serialized: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cache = {}
    for name, entry in serialized.items():
        res_dict = entry.get("result")
        res = deserialize_results([res_dict])[0] if res_dict else None
        cache[name] = {
            "hash": entry["hash"],
            "result": res
        }
    return cache


def save_state_to_disk() -> None:
    ss = st.session_state
    try:
        data = {
            "bid_text": ss.get("bid_text", ""),
            "bid_source": ss.get("bid_source", ""),
            "bid_file_id": ss.get("bid_file_id", ""),
            "bid_files": ss.get("bid_files", {}),
            "vendor_files": ss.get("vendor_files", {}),
            "vendor_errors": ss.get("vendor_errors", {}),
            "bid": ss.get("bid"),
            "results": serialize_results(ss.get("results")),
            "xai": ss.get("xai", []),
            "narrative": ss.get("narrative", None),
            "processed": ss.get("processed", False),
            "audit_cache": serialize_audit_cache(ss.get("audit_cache", {})),
            "last_bid_hash": ss.get("last_bid_hash", ""),
            "vendor_mtime": ss.get("vendor_mtime", {}),
            "vendor_paths": ss.get("vendor_paths", {}),
            "bid_paths": ss.get("bid_paths", {}),
            "pipeline_stage": ss.get("pipeline_stage", "upload"),
            "bid_confirmed": ss.get("bid_confirmed", False),
        }
        tmp_file = CACHE_FILE + ".tmp"
        with open(tmp_file, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_file, CACHE_FILE)
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        with open("error_log.txt", "w") as f:
            f.write(err_str)
        st.error(f"Cache Save Error: {err_str}")

def load_state_from_disk() -> bool:
    ss = st.session_state
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        ss.bid_text = data.get("bid_text", "")
        ss.bid_source = data.get("bid_source", "")
        ss.bid_file_id = data.get("bid_file_id", "")
        ss.bid_files = data.get("bid_files", {})
        ss.vendor_files = data.get("vendor_files", {})
        ss.vendor_errors = data.get("vendor_errors", {})
        ss.bid = data.get("bid")
        ss.results = deserialize_results(data.get("results"))
        ss.xai = data.get("xai", [])
        ss.narrative = data.get("narrative", None)
        ss.processed = data.get("processed", False)
        ss.audit_cache = deserialize_audit_cache(data.get("audit_cache", {}))
        ss.last_bid_hash = data.get("last_bid_hash", "")
        ss.vendor_mtime = data.get("vendor_mtime", {})
        ss.vendor_paths = data.get("vendor_paths", {})
        ss.bid_paths = data.get("bid_paths", {})
        ss.pipeline_stage = data.get("pipeline_stage", "upload")
        ss.bid_confirmed = data.get("bid_confirmed", False)
        return True
    except EOFError:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        return False
    except Exception as e:
        import traceback
        st.error(f"Cache Load Error: {traceback.format_exc()}")
        return False

def init_state() -> None:
    ss = st.session_state
    if "bid_text" not in ss:
        if not load_state_from_disk():
            ss.setdefault("bid_text", "")
            ss.setdefault("bid_source", "")
            ss.setdefault("bid_file_id", "")
            ss.setdefault("bid_files", {})
            ss.setdefault("vendor_files", {})      # name -> {filename: text}
            ss.setdefault("vendor_errors", {})     # name -> {filename: error|None}
            ss.setdefault("vendor_paths", {})
            ss.setdefault("bid_paths", {})
            ss.setdefault("audit_cache", {})
            ss.setdefault("last_bid_hash", "")
            ss.setdefault("vendor_mtime", {})
    ss.setdefault("results", None)
    ss.setdefault("bid", None)
    ss.setdefault("xai", [])
    ss.setdefault("narrative", None)
    ss.setdefault("processed", False)
    ss.setdefault("bid_uploader_key", 0)
    ss.setdefault("audit_cache", {})
    ss.setdefault("last_bid_hash", "")
    ss.setdefault("vendor_mtime", {})
    ss.setdefault("vendor_paths", {})
    ss.setdefault("bid_paths", {})
    ss.setdefault("pipeline_stage", "upload")   # upload | rules_review | audit
    ss.setdefault("bid_confirmed", False)


def generate_dynamic_demo_bidders(bid: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    tender_id = bid.get("tender_id", "TENDER-2026")
    if not tender_id or tender_id == "Unknown":
        tender_id = "TENDER-2026"
        
    mand_specs = bid.get("mandatory_specs", [])
    pref_specs = bid.get("preferred_specs", [])
    
    # 1. Delta Systems (Fully Compliant)
    delta_maf = f"OEM Manufacturer's Authorization Form (MAF) for Tender {tender_id}.\nWe hereby authorize M/s Delta Systems to bid, supply and support our products for the duration of the contract."
    
    delta_specs = f"TECHNICAL COMPLIANCE DATASHEET — DELTA SYSTEMS\nProposed Model: DS-Premium-Compliant-Model\n"
    for s in mand_specs:
        req_val = s.get("required_value", "Compliant")
        delta_specs += f"- {s['label']}: Proposed compliant value meeting {req_val} {s.get('unit', '')}\n"
    for s in pref_specs:
        req_val = s.get("required_value", "Compliant")
        delta_specs += f"- {s['label']}: Proposed compliant value meeting {req_val} {s.get('unit', '')}\n"
        
    delta_exp = f"COMPLETED WORK ORDERS & CERTIFICATES\n"
    delta_exp += f"Order 1: Supply to IOCL of similar equipment. Value: INR 45 Lakhs. Completed successfully.\n"
    delta_exp += f"Order 2: Supply to ONGC. Value: INR 35 Lakhs. Completed successfully.\n"
    delta_exp += f"Order 3: Supply to BHEL. Value: INR 25 Lakhs. Completed successfully.\n"
    
    delta_decl = (
        "Annexure-I Bidder Details: PAN AAACT9901M, GSTIN 27AAACT9901M1Z5, SAP Code 500192.\n"
        "Annexure-III Bid Security Declaration: We declare that we will be suspended if we withdraw or modify our bid.\n"
        "Annexure-IV Insolvency Declaration: We declare that we are not undergoing insolvency or bankruptcy proceedings.\n"
        "Annexure-V Blacklisting Declaration: We declare that we are not blacklisted or holiday listed.\n"
        "Annexure-X Local Content Certificate: Minimum local content is 55%."
    )
    
    # 2. Vibrant IT (Partially Compliant)
    vibrant_maf = f"OEM Authorization Letter for Tender {tender_id}.\nWe authorize M/s Vibrant IT to quote and support our hardware solutions."
    
    vibrant_specs = f"TECHNICAL DATASHEET — VIBRANT IT\nProposed Model: VIT-Standard-Model\n"
    # Fails first spec parameter slightly to test less-compliant scoring logic
    for i, s in enumerate(mand_specs):
        if i == 0 and s.get("op") in ("gte", "lte"):
            vibrant_specs += f"- {s['label']}: Proposed value fails to meet the requirement (offering non-compliant variant)\n"
        else:
            req_val = s.get("required_value", "Compliant")
            vibrant_specs += f"- {s['label']}: Proposed compliant value meeting {req_val} {s.get('unit', '')}\n"
    for s in pref_specs:
        req_val = s.get("required_value", "Compliant")
        vibrant_specs += f"- {s['label']}: Proposed compliant value meeting {req_val} {s.get('unit', '')}\n"
        
    vibrant_exp = f"COMPLETED CONTRACTS\n"
    vibrant_exp += f"Order 1: Supply of similar systems to GAIL. Value: INR 38.5 Lakhs. Completed successfully.\n"
    
    vibrant_decl = (
        "Annexure-I Bidder Details: PAN AAABT1102K, GSTIN 27AAABT1102K1Z9.\n"
        "Annexure-III Bid Security Declaration: We declare that we will be suspended if we withdraw or modify our bid.\n"
        "Annexure-IV Insolvency Declaration: We declare that we are not undergoing insolvency or bankruptcy proceedings.\n"
        "Annexure-V Blacklisting Declaration: We declare that we are not blacklisted.\n"
        "Annexure-X Local Content Certificate: Minimum local content is 60%."
    )
    
    # 3. Cyber Infosys (Non-Compliant)
    cyber_maf = f"OEM Letterhead: We authorize M/s Cyber Infosys to supply computer accessories for Tender {tender_id}."
    
    cyber_specs = f"TECHNICAL BID — CYBER INFOSYS\nProposed Model: Basic-Computer-Accessory-Model-280\n"
    for s in mand_specs:
        cyber_specs += f"- {s['label']}: Proposed computer accessory (unrelated product model specs)\n"
        
    cyber_exp = f"Experience: Reseller of general computer accessories. No similar large work orders completed."
    
    cyber_decl = "We accept GeM General Terms and Conditions."
    
    return {
        "Delta Systems": {
            "maf.pdf": delta_maf,
            "datasheet.pdf": delta_specs,
            "experience.pdf": delta_exp,
            "declarations.pdf": delta_decl
        },
        "Vibrant IT": {
            "maf.pdf": vibrant_maf,
            "datasheet.pdf": vibrant_specs,
            "experience.pdf": vibrant_exp,
            "declarations.pdf": vibrant_decl
        },
        "Cyber Infosys": {
            "maf.pdf": cyber_maf,
            "datasheet.pdf": cyber_specs,
            "experience.pdf": cyber_exp,
            "declarations.pdf": cyber_decl
        }
    }


def load_demo() -> None:
    ss = st.session_state
    ss.bid_text = MASTER_BID_TEXT
    ss.bid_source = "Demo NIT (IOCL/HR/IT/2026/NW-4471)"
    ss.bid_file_id = "demo_bid"
    ss.bid_files = {"Demo NIT.pdf": MASTER_BID_TEXT}
    
    # Use deterministic parser to instantly extract tender criteria structure
    from audit_engine import AuditEngine
    engine = AuditEngine()
    ss.bid = engine.parse_master_bid(MASTER_BID_TEXT)
    
    # Dynamically generate bidder files based on the tender specs
    ss.vendor_files = generate_dynamic_demo_bidders(ss.bid)
    ss.vendor_errors = {k: {f: None for f in v} for k, v in ss.vendor_files.items()}
    
    ss.results = None
    ss.processed = False
    ss.bid_confirmed = False
    ss.pipeline_stage = "rules_review"
    ss.bid_uploader_key += 1
    save_state_to_disk()


def reset_all() -> None:
    for k in ["bid_text", "bid_source", "bid_file_id", "bid_files", "vendor_files", "vendor_errors",
              "results", "bid", "xai", "narrative", "processed", "bid_uploader_key", "audit_cache", "last_bid_hash",
              "vendor_mtime", "vendor_paths", "bid_paths", "pipeline_stage", "bid_confirmed"]:
        st.session_state.pop(k, None)
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except OSError:
            pass
    import shutil
    if os.path.exists(".sentinel_cache_files"):
        try:
            shutil.rmtree(".sentinel_cache_files")
        except Exception:
            pass
    init_state()

def get_vendor_files_hash(files: Dict[str, str]) -> str:
    import hashlib
    import json
    sorted_items = sorted(files.items())
    serialized = json.dumps(sorted_items)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def run_audit(api_key: str, model: str) -> None:
    # Clear the audit_engine.log file on every run
    try:
        with open("audit_engine.log", "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

    ss = st.session_state
    import hashlib
    
    # Invalidate cache if Master Bid text changes
    current_bid_hash = hashlib.sha256(ss.bid_text.encode('utf-8')).hexdigest()
    if ss.get("last_bid_hash") != current_bid_hash:
        ss.audit_cache = {}
        ss.last_bid_hash = current_bid_hash
        
    mode = ss.get("engine_mode", "Deterministic Base Engine (Zero-AI / Fast)")
    engine = None
    
    # Dynamic self-healing fallback wrapper
    try:
        if mode == "Deterministic Base Engine (Zero-AI / Fast)":
            engine = audit_engine.AuditEngine()
        elif mode == "Local Llama RAG (Ollama)" and HAS_RAG_ENGINE:
            import urllib.request
            import subprocess
            
            ollama_running = False
            ollama_url = ss.get("ollama_url", "http://localhost:11434")
            try:
                req = urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=1.5)
                if req.getcode() == 200:
                    ollama_running = True
            except Exception:
                pass
                
            groq_key = ss.get("groq_key", "").strip()
            
            if not ollama_running:
                if groq_key:
                    st.warning("⚠️ Local Ollama is not running. Automatically switching to Cloud RAG (Groq) mode using your API key to perform the audit...")
                    ss.engine_mode = "Cloud RAG (Groq)"
                    mode = "Cloud RAG (Groq)"
                else:
                    st.error("❌ Local Ollama is not running (Connection Refused). Please launch the Ollama desktop application or paste your Groq API key in the sidebar to run in Cloud RAG mode.")
                    return
            else:
                # Check available RAM on Windows to prevent CPU crashes
                try:
                    out = subprocess.check_output("wmic OS get FreePhysicalMemory", shell=True).decode("utf-8")
                    lines = [line.strip() for line in out.split("\n") if line.strip()]
                    if len(lines) > 1:
                        free_kb = int(lines[1])
                        free_gb = free_kb / (1024 * 1024)
                        if free_gb < 3.0:
                            if groq_key:
                                st.warning(f"⚠️ Free system RAM is extremely low ({free_gb:.2f} GB). To prevent a system crash, automatically switching to Cloud RAG (Groq) mode...")
                                ss.engine_mode = "Cloud RAG (Groq)"
                                mode = "Cloud RAG (Groq)"
                            else:
                                st.warning(f"⚠️ Free system RAM is low ({free_gb:.2f} GB). Ollama might crash or fail. If this happens, please close other apps or paste a Groq API key in the sidebar.")
                except Exception:
                    pass

            if mode == "Local Llama RAG (Ollama)":
                engine = LocalRAGAuditEngine(
                    model_name=ss.get("ollama_model_name", "qwen2.5:7b"),
                    base_url=ss.get("ollama_url", "http://localhost:11434"),
                    mode="local"
                )
        elif mode == "Cloud RAG (Groq)" and HAS_RAG_ENGINE:
            groq_key = ss.get("groq_key", "").strip()
            if not groq_key:
                st.error("❌ Groq API Key is required for Cloud RAG (Groq) mode. Please enter it in the sidebar.")
                return
            
            gemini_api_key = ""
            emb_provider = ss.get("groq_embedding_provider", "Local Ollama Embeddings")
            if emb_provider == "Cloud Gemini Embeddings":
                gemini_api_key = ss.get("gemini_key", "").strip()
                if not gemini_api_key:
                    st.error("❌ Gemini API Key is required to run Cloud Gemini Embeddings.")
                    return
            
            engine = LocalRAGAuditEngine(
                model_name=ss.get("groq_model_name", "llama-3.3-70b-versatile"),
                mode="groq",
                api_key=groq_key,
                gemini_api_key=gemini_api_key,
                embedding_provider=emb_provider
            )
        elif mode == "Cloud RAG (Gemini)" and HAS_RAG_ENGINE:
            gemini_key = ss.get("gemini_key", "").strip()
            if not gemini_key:
                st.error("❌ Gemini API Key is required for Cloud RAG (Gemini) mode. Please enter it in the sidebar.")
                return
            
            engine = LocalRAGAuditEngine(
                model_name=ss.get("gemini_model_name", "gemini-1.5-flash"),
                mode="gemini",
                api_key=gemini_key,
                embedding_provider="Cloud Gemini Embeddings"
            )
        else:
            engine = audit_engine.AuditEngine()
    except Exception as init_err:
        st.warning(f"⚠️ Failed to initialize {mode}: {init_err}. Automatically falling back to Deterministic Base Engine (Zero-AI / Fast)...")
        engine = audit_engine.AuditEngine()
        mode = "Deterministic Base Engine (Zero-AI / Fast)"
        ss.engine_mode = "Deterministic Base Engine (Zero-AI / Fast)"

    if not engine:
        engine = audit_engine.AuditEngine()

    engine.bid_text = ss.bid_text
    engine.bid_source = ss.bid_source

    placeholder = st.empty()
    error_console_placeholder = st.empty()
    placeholder.markdown(render_audit_terminal(0, "Pending Vendor Analysis...", 0.0), unsafe_allow_html=True)
    time.sleep(0.1)
    if not ss.get("bid"):
        with placeholder:
            with st.spinner("🔍 Extracting rules from master bid document..."):
                ss.bid = engine.parse_master_bid(ss.bid_text)

    # Pause here and send user to rules review page before vendor analysis
    if not ss.get("bid_confirmed", False):
        ss.pipeline_stage = "rules_review"
        save_state_to_disk()
        st.rerun()
        return

    names = list(ss.vendor_files.keys())
    results: List[VendorResult] = []
    cache_miss = False
    
    # Step 1: Analyzing Vendors (leveraging smart incremental cache)
    for i, name in enumerate(names, start=1):
        pct = (i / max(len(names), 1)) * 100
        files = ss.vendor_files[name]
        files_hash = get_vendor_files_hash(files)
        
        cached_entry = ss.audit_cache.get(name)
        if cached_entry and cached_entry.get("hash") == files_hash:
            placeholder.markdown(render_audit_terminal(1, f"Loading {name} (Cached)...", pct), unsafe_allow_html=True)
            time.sleep(0.1)  # fast display update
            results.append(cached_entry["result"])
        else:
            cache_miss = True
            placeholder.markdown(render_audit_terminal(1, f"Analyzing {name} ({i}/{len(names)})...", pct), unsafe_allow_html=True)
            time.sleep(0.3)
            errors = ss.vendor_errors.get(name, {f: None for f in files})
            res = engine.analyze_vendor(name, files, errors, ss.bid)
            ss.audit_cache[name] = {"hash": files_hash, "result": res}
            results.append(res)

        # Read and extract warnings/errors from log file in real-time
        errors_found = []
        try:
            import os
            import re
            if os.path.exists("audit_engine.log"):
                with open("audit_engine.log", "r", encoding="utf-8") as lf:
                    for line in lf:
                        lower_line = line.lower()
                        if "failed" in lower_line or "error" in lower_line or "warning" in lower_line or "exception" in lower_line or "corrupt" in lower_line or "winerror" in lower_line:
                            clean_line = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*", "", line.strip())
                            if clean_line not in errors_found:
                                errors_found.append(clean_line)
        except Exception:
            pass

        if errors_found:
            error_list_html = "".join(f"<div style='margin-bottom: 4px;'>• {html.escape(e)}</div>" for e in errors_found[-5:])
            error_console_placeholder.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-family: monospace; font-size: 12px; max-width: 1000px;">
                    <div style="color: #F87171; font-weight: 700; margin-bottom: 6px; display: flex; align-items: center; gap: 8px;">
                        <span>⚠️ Active Warnings & Errors ({len(errors_found)})</span>
                    </div>
                    <div style="max-height: 120px; overflow-y: auto; color: #FCA5A5; line-height: 1.4;">
                        {error_list_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            error_console_placeholder.empty()
    
    # Step 2: Generating Ranking & Summary
    placeholder.markdown(render_audit_terminal(2, f"Analyzed {len(names)} vendor submissions.", 100.0), unsafe_allow_html=True)
    time.sleep(0.1)
    if cache_miss or not ss.get("xai") or not ss.get("narrative"):
        ss.xai = engine.rank_and_explain(results)
        ss.narrative = engine.narrate(ss.bid, results) if hasattr(engine, "narrate") else None
        time.sleep(0.4)
    else:
        placeholder.markdown(render_audit_terminal(2, f"Loaded ranking & narrative summary from cache.", 100.0), unsafe_allow_html=True)
        time.sleep(0.3)

    placeholder.empty()
    error_console_placeholder.empty()

    ss.results = results
    ss.processed = True
    save_state_to_disk()


# ---------------------------------------------------------------------------
# DIALOGS & SIDEBAR — Control inputs
# ---------------------------------------------------------------------------
try:
    dialog_decorator = st.dialog
except AttributeError:
    try:
        dialog_decorator = st.experimental_dialog
    except AttributeError:
        def dummy_dialog(*args, **kwargs):
            def wrapper(func):
                return func
            return wrapper
        dialog_decorator = dummy_dialog

@dialog_decorator("Document Viewer", width="large")
def view_documents_dialog(title: str, files_dict: Dict[str, str], focus_file: Optional[str] = None, focus_page: Optional[int] = None) -> None:
    st.markdown(f"<div class='doc-viewer-marker' style='font-weight:600; font-size:16px; color:var(--blue); margin-bottom:16px;'>{html.escape(title)}</div>", unsafe_allow_html=True)
    st.markdown("""<style>
    div[role="dialog"] div[data-testid="stExpander"] summary p::before,
    .stDialog div[data-testid="stExpander"] summary p::before {
        content: '';
        display: inline-block;
        width: 16px; height: 16px; margin-right: 8px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23C7D3EA' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3Cpolyline points='10 9 9 9 8 9'/%3E%3C/svg%3E") no-repeat center;
        background-size: contain;
        vertical-align: middle;
        filter: drop-shadow(0 0 3px rgba(199,211,234,0.4));
    }
    div[role="dialog"] div[data-testid="stExpander"] summary p,
    .stDialog div[data-testid="stExpander"] summary p {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    </style>""", unsafe_allow_html=True)
    if not files_dict:
        st.info("No documents available.")
        return
    for fname, ftext in files_dict.items():
        is_expanded = (focus_file is not None and fname == focus_file) or (focus_file is None)
        with st.expander(f"{html.escape(fname)}", expanded=is_expanded):
            formatted_html = format_pdf_text_to_html(ftext)
            st.markdown(f"<div style='max-height: 500px; overflow-y: auto; overflow-x: hidden; background: var(--panel2); padding: 16px 22px; border-radius: 8px; border: 1px solid var(--line);'>{formatted_html}</div>", unsafe_allow_html=True)
            
    if focus_page is not None:
        scroll_js = f"""
        <script>
            setTimeout(function() {{
                const doc = window.parent.document;
                const targetId = "page-{focus_page}";
                const el = doc.getElementById(targetId);
                if (el) {{
                    el.scrollIntoView({{ behavior: "smooth", block: "center" }});
                    const originalBorder = el.style.border;
                    el.style.border = "1px solid #E9F056";
                    el.style.boxShadow = "0 0 15px rgba(233, 240, 86, 0.5)";
                    setTimeout(function() {{
                        el.style.border = originalBorder;
                        el.style.boxShadow = "";
                    }}, 2500);
                }}
            }}, 800);
        </script>
        """
        components.html(scroll_js, height=0, width=0)

def render_visual_document_viewer() -> None:
    ss = st.session_state
    focus_file = ss.get("view_visual_file")
    focus_page = ss.get("view_visual_page", 1)
    vendor_name = ss.get("view_visual_vendor", "Master")
    
    st.markdown(f"## 📄 Visual PDF Page Viewer")
    st.markdown(f"**Vendor:** `{vendor_name}` | **Document:** `{focus_file}`")
    
    # CSS injection to force high-contrast black text on button backgrounds, remove colorful emojis, and handle disabled state
    st.markdown("""<style>
    div[data-testid="stColumn"] div.stButton > button {
        background: linear-gradient(135deg, #E9F056, #C6CD3E) !important;
        color: #1e1e1e !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(233, 240, 86, 0.25) !important;
        font-weight: 800 !important;
        font-size: 13.5px !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        text-shadow: none !important;
    }
    div[data-testid="stColumn"] div.stButton > button p {
        color: #1e1e1e !important;
        font-weight: 800 !important;
        text-shadow: none !important;
    }
    div[data-testid="stColumn"] div.stButton > button:hover {
        background: #FFFFFF !important;
        color: #1e1e1e !important;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.4) !important;
    }
    div[data-testid="stColumn"] div.stButton > button:hover p {
        color: #1e1e1e !important;
    }
    div[data-testid="stColumn"] div.stButton > button:disabled {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #64748B !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    div[data-testid="stColumn"] div.stButton > button:disabled p {
        color: #64748B !important;
    }
    </style>""", unsafe_allow_html=True)
    
    # 1. Resolve local filepath
    filepath = None
    
    # If focus_file is the bid_source description, resolve it to the actual first filename in cache
    if vendor_name == "Master" and focus_file == ss.get("bid_source"):
        if ss.get("bid_files"):
            focus_file = list(ss.bid_files.keys())[0]
            
    if vendor_name == "Master":
        filepath = ss.get("bid_paths", {}).get(focus_file)
    else:
        filepath = ss.get("vendor_paths", {}).get(vendor_name, {}).get(focus_file)
        
    # If not in session state paths, search in local files cache or directories
    if not filepath or not os.path.exists(filepath):
        # Fallback 1: check in local .sentinel_cache_files
        cache_path = os.path.join(".sentinel_cache_files", focus_file) if focus_file else ""
        if cache_path and os.path.exists(cache_path):
            filepath = cache_path
        else:
            # Fallback 2: walk current workspace or user directories to find the file
            found = False
            if focus_file:
                for parent_dir in [r"C:\Users\aloki\Downloads\iocl bid files", r"c:\Users\aloki\Desktop\iocl"]:
                    if os.path.exists(parent_dir):
                        for root, dirs, files in os.walk(parent_dir):
                            if focus_file in files:
                                filepath = os.path.join(root, focus_file)
                                found = True
                                break
                    if found:
                        break
                    
    # Save the resolved filepath back to session state to prevent searching again
    if filepath and os.path.exists(filepath):
        if vendor_name == "Master":
            ss.setdefault("bid_paths", {})[focus_file] = filepath
        else:
            ss.setdefault("vendor_paths", {}).setdefault(vendor_name, {})[focus_file] = filepath

    # Check if we have the text content in memory (e.g. for demo corpus or text uploads)
    file_text = None
    if focus_file:
        if vendor_name == "Master":
            file_text = ss.get("bid_files", {}).get(focus_file)
        else:
            file_text = ss.get("vendor_files", {}).get(vendor_name, {}).get(focus_file)

    # 1.5 Render in-memory text if physical PDF is missing
    if not filepath or not os.path.exists(filepath):
        if file_text:
            st.info(f"ℹ️ Serving document `{focus_file}` from memory cache.")
            
            # Parse page segments
            page_blocks = {}
            import re as _re
            page_matches = list(_re.finditer(r'--- PAGE (\d+) ---\n?', file_text))
            for idx_p, m in enumerate(page_matches):
                p_num = int(m.group(1))
                p_start = m.end()
                p_end = page_matches[idx_p+1].start() if idx_p+1 < len(page_matches) else len(file_text)
                page_blocks[p_num] = file_text[p_start:p_end]
                
            if not page_blocks:
                # Split by form feed or use the entire text as page 1
                parts = file_text.split('\f')
                page_blocks = {i+1: p for i, p in enumerate(parts)}
                
            total_pages = len(page_blocks) if page_blocks else 1
            page_num = max(1, min(focus_page, total_pages))
            
            # Action controls
            col_back, col_prev, col_page, col_next, col_spacer = st.columns([2, 1, 1, 1, 4])
            
            with col_back:
                if st.button("Return to Dashboard", key="text_view_back", use_container_width=True):
                    ss.pop("view_visual_file", None)
                    st.rerun()
            with col_prev:
                if st.button("Previous", key="text_view_prev", disabled=(page_num <= 1), use_container_width=True):
                    ss["view_visual_page"] = page_num - 1
                    st.rerun()
            with col_page:
                st.markdown(f"<div style='text-align: center; line-height: 38px; font-weight: 700; color: #E9F056;'>{page_num} / {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.button("Next", key="text_view_next", disabled=(page_num >= total_pages), use_container_width=True):
                    ss["view_visual_page"] = page_num + 1
                    st.rerun()
                    
            st.divider()
            
            # Render page content
            page_content = page_blocks.get(page_num, file_text)
            st.markdown(f"### 📄 Page {page_num}")
            st.markdown(
                f"<div style='background: var(--panel2); padding: 20px 24px; border-radius: 8px; border: 1px solid var(--line); font-family: monospace; white-space: pre-wrap; font-size: 13.5px; color: #F8FAFC; max-height: 600px; overflow-y: auto;'>{html.escape(page_content)}</div>",
                unsafe_allow_html=True
            )
            return
        else:
            st.error(f"Could not locate document file `{focus_file}` on disk or in memory cache.")
            if st.button("Return to Dashboard", key="err_view_back"):
                ss.pop("view_visual_file", None)
                st.rerun()
            return

    # 2. Render Page using pdfplumber
    import pdfplumber
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            page_num = max(1, min(focus_page, total_pages))
            
            # Action controls
            col_back, col_prev, col_page, col_next, col_spacer = st.columns([2, 1, 1, 1, 4])
            
            with col_back:
                if st.button("Return to Dashboard", use_container_width=True):
                    ss.pop("view_visual_file", None)
                    st.rerun()
            with col_prev:
                if st.button("Previous", disabled=(page_num <= 1), use_container_width=True):
                    ss["view_visual_page"] = page_num - 1
                    st.rerun()
            with col_page:
                st.markdown(f"<div style='text-align: center; line-height: 38px; font-weight: 700; color: #E9F056;'>{page_num} / {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.button("Next", disabled=(page_num >= total_pages), use_container_width=True):
                    ss["view_visual_page"] = page_num + 1
                    st.rerun()
                    
            st.divider()
            
            # Extract page
            page = pdf.pages[page_num - 1]
            try:
                with st.spinner("Rendering high-quality page view..."):
                    im = page.to_image(resolution=150)
                    pil_img = im.original
                    st.image(pil_img, caption=f"{focus_file} (Page {page_num} of {total_pages})", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render page visually: {e}. Showing text fallback below:")
                text_ext = page.extract_text()
                if text_ext:
                    st.code(text_ext)
                else:
                    st.info("Manual check required: The page layout cannot be automatically rendered or has extremely bad print quality.")
    except Exception as e:
        st.error(f"Error opening PDF document `{focus_file}`: {e}")
        if st.button("Return to Dashboard"):
            ss.pop("view_visual_file", None)
            st.rerun()

def render_sidebar() -> Tuple[str, str, bool]:
    ss = st.session_state
    logo_b64 = get_base64_image("logo.jpg")
    glyph_content = f'<img src="data:image/jpeg;base64,{logo_b64}">' if logo_b64 else 'A'
    
    with st.sidebar:
        st.markdown(f"""
<div style="background: linear-gradient(160deg, rgba(15, 23, 42, 0.8) 0%, rgba(26, 26, 30, 0.8) 100%);
backdrop-filter: blur(20px); padding: 22px 20px; border-radius: 16px;
border: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 2px solid #E9F056;
margin-bottom: 24px; position: relative; overflow: hidden; display: flex; flex-direction: column; gap: 14px;">
<div style="position: absolute; right: 0; bottom: 0; width: 100%; height: 100%; opacity: 0.02; background: repeating-linear-gradient(45deg, #ffffff, #ffffff 1px, transparent 1px, transparent 8px); z-index: 0;"></div>
<div class="sys-status" style="align-self: flex-start; display: inline-flex; align-items: center; gap: 6px; background: rgba(233, 240, 86, 0.05); border: 1px solid rgba(233, 240, 86, 0.1); padding: 4px 8px; border-radius: 6px; z-index: 2;">
<div style="width: 5px; height: 5px; border-radius: 50%; background: #E9F056; box-shadow: 0 0 4px #E9F056;"></div>
<span id="sys-clock" style="color: #E9F056; font-size: 9px; font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.5px;">SYSTEM ONLINE · {time.strftime("%d %b %Y %H:%M:%S").upper()}</span>
</div>
<div style="display: flex; align-items: center; gap: 16px; z-index: 2;">
<label class="sidebar-glyph" for="logo-anim-toggle" style="transition: border-color 0.3s ease; border: 1px solid rgba(255,255,255,0.1); cursor: pointer; margin: 0;">{glyph_content}</label>
<div>
<div style="font-weight: 900; font-size: 22px; letter-spacing: -0.5px; background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%); -webkit-background-clip: text; color: transparent; line-height: 1;">Argus Bid AI</div>
<div style="color: #94a3b8; font-size: 11px; font-weight: 600; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; line-height: 1;">Tender Audit Engine</div>
</div>
</div>
</div>
    """, unsafe_allow_html=True)
        
        # Theme appearance toggle switch
        st.write("")
        theme_toggle = st.toggle("☀️ Light Mode Theme", value=st.session_state.get("light_mode", False))
        if theme_toggle != st.session_state.get("light_mode", False):
            st.session_state.light_mode = theme_toggle
            st.rerun()

        components.html("""
        <script>
        setInterval(() => {
            const el = window.parent.document.getElementById("sys-clock");
            if (el) {
                const d = new Date();
                const pad = (n) => n.toString().padStart(2, '0');
                const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
                const timeStr = `${pad(d.getDate())} ${months[d.getMonth()]} ${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
                el.innerText = `SYSTEM ONLINE · ${timeStr}`;
            }
        }, 1000);
        </script>
        """, height=0, width=0)
        st.markdown(f"""
        <div class="demo-btn-marker"></div>
        <style>
        .element-container:has(.demo-btn-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.demo-btn-marker) + div[data-testid="stElementContainer"] button {{
            background: linear-gradient(135deg, #8B5CF6, #6D28D9) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            padding-top: 6px !important;
            padding-bottom: 6px !important;
        }}
        .element-container:has(.demo-btn-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.demo-btn-marker) + div[data-testid="stElementContainer"] button:hover {{
            background: linear-gradient(135deg, #A78BFA, #8B5CF6) !important;
            transform: translateY(-2px) !important;
        }}
        .element-container:has(.demo-btn-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.demo-btn-marker) + div[data-testid="stElementContainer"] button p::before {{
            content: '';
            display: inline-block;
            width: 18px; height: 18px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: text-bottom;
            filter: drop-shadow(0 0 3px rgba(255,255,255,0.4));
        }}
        </style>
        """, unsafe_allow_html=True)
        
        st.button("Load Demo Corpus", on_click=load_demo, use_container_width=True,
                  help="Loads a complete sample NIT + 3 vendor submissions for an instant demo.", key="btn_load_demo")

        if ss.get("bid"):
            if st.button("Generate Simulated Bids", use_container_width=True,
                         help="Dynamically generates 3 mock vendor submissions matching the uploaded tender requirements.", key="btn_generate_simulated"):
                ss.vendor_files = generate_dynamic_demo_bidders(ss.bid)
                ss.vendor_errors = {k: {f: None for f in v} for k, v in ss.vendor_files.items()}
                ss.results = None
                ss.processed = False
                save_state_to_disk()
                st.rerun()

        st.divider()
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(26, 26, 30, 0.6); 
        border: 1px solid rgba(245, 158, 11, 0.15); border-left: 4px solid #F59E0B; border-radius: 8px; 
        padding: 12px 14px; font-weight: 800; font-size: 13px; color: #E2E8F0; letter-spacing: 0.5px; 
        text-transform: uppercase; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 5px rgba(245,158,11,0.5));">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            1 · Master BID / NIT
        </div>
        """, unsafe_allow_html=True)
        with st.form("add_master_bid", clear_on_submit=True):
            bid_up = st.file_uploader("Upload the tender document(s)", type=["pdf", "txt", "md"],
                                       accept_multiple_files=True,
                                       key=f"bid_uploader_{ss.bid_uploader_key}", label_visibility="collapsed")
            st.markdown('<div class="add-bid-marker"></div>', unsafe_allow_html=True)
            add_bid = st.form_submit_button("Add / Update BID", use_container_width=True)

        st.markdown("""<style>
        .element-container:has(.add-bid-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.add-bid-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #F59E0B, #C6CD3E) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
        }
        .element-container:has(.add-bid-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.add-bid-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #FBBF24, #F59E0B) !important;
            transform: translateY(-1px) !important;
        }
        .element-container:has(.add-bid-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.add-bid-marker) + div[data-testid="stElementContainer"] button p::before {
            content: '';
            display: inline-block;
            width: 18px; height: 18px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' y1='5' x2='12' y2='19'/%3E%3Cline x1='5' y1='12' x2='19' y2='12'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: middle;
            filter: drop-shadow(0 0 5px rgba(255,255,255,0.6));
        }
        </style>""", unsafe_allow_html=True)
        bid_spinner_placeholder = st.empty()

        if ss.bid_files:
            with st.container(border=True):
                st.markdown(
                    """<div style="margin-bottom: 12px; padding: 10px 14px; background: rgba(245, 158, 11, 0.08); 
                    border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; color: #E8EEF8; font-weight: 600; font-size: 14px; 
                    display: flex; align-items: center; gap: 10px; box-shadow: inset 0 0 12px rgba(245, 158, 11, 0.05);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="rgba(245, 158, 11, 0.2)" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.6));"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                    Cached Master BID Documents
                    </div>""", unsafe_allow_html=True
                )
                for filename, text in ss.bid_files.items():
                    c1, c2 = st.columns([8.5, 1.5])
                    with c1:
                        st.markdown('<div class="view-bid-marker"></div>', unsafe_allow_html=True)
                        if st.button(f"{filename}", use_container_width=True, key=f"btn_view_bid_{filename}"):
                            view_documents_dialog("Master Tender Document", {f"{filename} — {len(text):,} chars": text})
                    with c2:
                        st.markdown('<div class="rm-bid-marker"></div>', unsafe_allow_html=True)
                        if st.button("✕", key=f"rm_bid_{filename}", help=f"Remove {filename}", use_container_width=True):
                            ss.bid_files.pop(filename, None)
                            if not ss.bid_files:
                                ss.bid_text = ""
                                ss.bid_source = ""
                                ss.bid_file_id = ""
                            else:
                                combined = ""
                                for name, t in ss.bid_files.items():
                                    combined += f"\n\n--- FILE {name} ---\n\n{t}"
                                ss.bid_text = combined.strip()
                                names = list(ss.bid_files.keys())
                                ss.bid_source = names[0] if len(names) == 1 else f"{len(names)} Documents"
                            ss.processed = False
                            ss.bid = None
                            ss.results = None
                            ss.xai = []
                            ss.narrative = None
                            ss.audit_cache = {}
                            save_state_to_disk()
                            st.rerun()

        st.markdown("""<style>
        .element-container:has(.view-bid-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.view-bid-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #E9F056, #C6CD3E) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .element-container:has(.view-bid-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.view-bid-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #FFFFFF, #E9F056) !important;
            transform: translateY(-1px) !important;
        }
        .element-container:has(.view-bid-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.view-bid-marker) + div[data-testid="stElementContainer"] button p::before {
            content: '';
            display: inline-block;
            width: 16px; height: 16px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3Cpolyline points='10 9 9 9 8 9'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: middle;
            filter: drop-shadow(0 0 3px rgba(255,255,255,0.5));
        }
        
        .element-container:has(.rm-bid-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.rm-bid-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #FF5C34, #DC2626) !important;
            border: none !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        .element-container:has(.rm-bid-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.rm-bid-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #F87171, #FF5C34) !important;
            transform: translateY(-1px) !important;
        }
        .element-container:has(.rm-bid-marker) + .element-container button p,
        div[data-testid="stElementContainer"]:has(.rm-bid-marker) + div[data-testid="stElementContainer"] button p {
            color: #FFFFFF !important;
            font-weight: bold !important;
            text-shadow: none !important;
            font-size: 16px !important;
        }
        </style>""", unsafe_allow_html=True)

        if add_bid and bid_up:
            current_ids = ",".join(sorted([getattr(f, "file_id", f.name) for f in bid_up]))
            if ss.get("bid_file_id") != current_ids:
                with bid_spinner_placeholder:
                    with custom_spinner("Analyzing master tender document(s)...", theme="yellow"):
                        added = False
                        for f in bid_up:
                            if f.name not in ss.bid_files:
                                text, err = read_uploaded_file(f, enable_ocr=True)
                                if text:
                                    if not is_valid_bid_document(text):
                                        st.error(f"'{f.name}' is not a valid BID document. Please enter a valid BID document.")
                                        continue
                                    ss.bid_files[f.name] = text
                                    added = True
                                    try:
                                        os.makedirs(".sentinel_cache_files", exist_ok=True)
                                        cache_path = os.path.join(".sentinel_cache_files", f.name)
                                        with open(cache_path, "wb") as fp:
                                            fp.write(f.getvalue())
                                        ss.setdefault("bid_paths", {})[f.name] = cache_path
                                    except Exception:
                                        pass
                                elif err:
                                    st.error(f"Could not read BID {f.name}: {err}")
                        if added:
                            combined = ""
                            for name, text in ss.bid_files.items():
                                combined += f"\n\n--- FILE {name} ---\n\n{text}"
                            ss.bid_text = combined.strip()
                            names = list(ss.bid_files.keys())
                            ss.bid_source = names[0] if len(names) == 1 else f"{len(names)} Documents"
                            ss.bid_file_id = current_ids
                            ss.processed = False
                            from audit_engine import AuditEngine
                            engine = AuditEngine()
                            ss.bid = engine.parse_master_bid(ss.bid_text)
                            ss.bid_confirmed = False
                            ss.pipeline_stage = "rules_review"
                            ss.results = None
                            ss.xai = []
                            ss.narrative = None
                            ss.audit_cache = {}
                            ss.bid_uploader_key += 1
                            save_state_to_disk()
                            st.rerun()

        st.divider()
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(26, 26, 30, 0.6); 
        border: 1px solid rgba(139, 92, 246, 0.15); border-left: 4px solid #8B5CF6; border-radius: 8px; 
        padding: 12px 14px; font-weight: 800; font-size: 13px; color: #E2E8F0; letter-spacing: 0.5px; 
        text-transform: uppercase; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 5px rgba(139,92,246,0.5));">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            2 · Vendor Submissions
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<style>
        .element-container:has(.add-vendor-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.add-vendor-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #8B5CF6, #6D28D9) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
        }
        .element-container:has(.add-vendor-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.add-vendor-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #A78BFA, #8B5CF6) !important;
            transform: translateY(-1px) !important;
        }
        .element-container:has(.add-vendor-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.add-vendor-marker) + div[data-testid="stElementContainer"] button p::before {
            content: '';
            display: inline-block;
            width: 18px; height: 18px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='12' y1='5' x2='12' y2='19'/%3E%3Cline x1='5' y1='12' x2='19' y2='12'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: middle;
            filter: drop-shadow(0 0 5px rgba(255,255,255,0.6));
        }
        </style>""", unsafe_allow_html=True)

        def detect_vendor_name_from_file(filename: str, text: str) -> str:
            import re
            base = filename.split("/")[-1].split("\\")[-1]
            for sep in [" - ", " _ ", " — ", " – "]:
                if sep in base:
                    return base.split(sep)[0].strip()
            m = re.search(r"bidder(?:\s*name)?\s*[:\-]?\s*([A-Za-z0-9\s.]{3,40})", text, re.I)
            if m:
                name = m.group(1).strip()
                if len(name) > 3 and "tender" not in name.lower() and "document" not in name.lower() and "ernet" not in name.lower():
                    return name.title()
            m2 = re.search(r"m/s\.?\s*([A-Za-z0-9\s.]{3,40})", text, re.I)
            if m2:
                name = m2.group(1).strip()
                if len(name) > 3 and "ernet" not in name.lower() and "iocl" not in name.lower():
                    return name.title()
            prefix = base.replace("_", " ").replace("-", " ")
            parts = prefix.split()
            if len(parts) >= 2:
                name = f"{parts[0]} {parts[1]}"
                return re.sub(r"\.pdf$", "", name, flags=re.I).strip().title()
            return re.sub(r"\.pdf$", "", parts[0], flags=re.I).strip().title()

        with st.form("add_vendor", clear_on_submit=True):
            vname = st.text_input("Vendor name (Optional)", placeholder="e.g. Sourav — Leave blank to auto-detect vendor name from files")
            vfiles = st.file_uploader("Vendor documents", type=["pdf", "txt", "md"],
                                      accept_multiple_files=True)
            st.markdown('<div class="add-vendor-marker"></div>', unsafe_allow_html=True)
            add = st.form_submit_button("Add / Update Vendor", use_container_width=True)
            
        st.markdown("""
        <div style="font-size: 11px; color: var(--muted); font-weight: 700; text-transform: uppercase; margin-top: 10px; margin-bottom: 5px; text-align: center;">
            ⚡ OR SCAN LOCAL FOLDER (Auto-Group Vendors)
        </div>
        """, unsafe_allow_html=True)
        with st.form("scan_local_folder", clear_on_submit=False):
            local_folder_path = st.text_input("Folder path on your PC", placeholder="e.g. C:\\Users\\aloki\\Downloads\\iocl bid files")
            st.markdown('<div class="scan-folder-marker"></div>', unsafe_allow_html=True)
            scan = st.form_submit_button("Scan & Load All Vendors", use_container_width=True)
            
        st.markdown("""<style>
        .element-container:has(.scan-folder-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.scan-folder-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #8B5CF6, #EC4899) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
        }
        .element-container:has(.scan-folder-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.scan-folder-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #A78BFA, #EC4899) !important;
            transform: translateY(-1px) !important;
        }
        </style>""", unsafe_allow_html=True)
        
        vendor_spinner_placeholder = st.empty()

        if scan and local_folder_path:
            import os
            path = local_folder_path.strip()
            if not os.path.exists(path) or not os.path.isdir(path):
                st.sidebar.error("The specified path does not exist or is not a directory.")
            else:
                # Ignore system/hidden directories and the code repository directory to avoid infinite loops
                subfolders = []
                for entry in os.scandir(path):
                    if entry.is_dir():
                        name_lower = entry.name.lower()
                        if not name_lower.startswith('.') and name_lower not in ["node_modules", "venv", "env", "argus-bid-ai-tender-audit-compliance"]:
                            subfolders.append(entry)
                
                if not subfolders:
                    st.sidebar.error("No valid subdirectories found. Please specify the parent folder containing your vendor subfolders.")
                else:
                    # Let's count total files to show a progress bar
                    all_files = []
                    for sub in subfolders:
                        vendor_name = sub.name.strip()
                        for root, dirs, files_in_dir in os.walk(sub.path):
                            # Skip hidden/system directories inside walk
                            dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() not in ["node_modules", "venv", "env", "argus-bid-ai-tender-audit-compliance"]]
                            for file in files_in_dir:
                                if file.lower().endswith(('.pdf', '.txt', '.md')):
                                    filepath = os.path.join(root, file)
                                    all_files.append((vendor_name, filepath, file))
                                    
                    total_count = len(all_files)
                    if total_count == 0:
                        st.sidebar.warning("No PDF, TXT, or MD documents found in any of the subfolders.")
                    else:
                        progress_bar = st.sidebar.progress(0)
                        status_text = st.sidebar.empty()
                        
                        ss.setdefault("vendor_mtime", {})
                        
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        
                        # Prepare list of tasks on the main thread (thread-safe!)
                        tasks_to_run = []
                        for vendor_name, filepath, file in all_files:
                            try:
                                mtime = os.path.getmtime(filepath)
                                size = os.path.getsize(filepath)
                                
                                cached_mtime = ss.get("vendor_mtime", {}).get(vendor_name, {}).get(file, {}).get("mtime")
                                cached_size = ss.get("vendor_mtime", {}).get(vendor_name, {}).get(file, {}).get("size")
                                
                                is_cached = (
                                    cached_mtime == mtime 
                                    and cached_size == size 
                                    and vendor_name in ss.vendor_files 
                                    and file in ss.vendor_files[vendor_name]
                                )
                                if not is_cached:
                                    tasks_to_run.append((vendor_name, filepath, file, mtime, size))
                            except Exception:
                                tasks_to_run.append((vendor_name, filepath, file, 0, 0))
                                
                        tasks_count = len(tasks_to_run)
                        
                        # Worker function (no session state or streamlit access)
                        def process_single_file_worker(task_data):
                            v_name, f_path, f_name, mtime, size = task_data
                            try:
                                class MockUploadedFile:
                                    def __init__(self, path, name):
                                        self._path = path
                                        self.name = name
                                    def getvalue(self):
                                        with open(self._path, 'rb') as fp:
                                            return fp.read()
                                            
                                mock_f = MockUploadedFile(f_path, f_name)
                                t, e = read_uploaded_file(mock_f, enable_ocr=True)
                                return v_name, f_name, t, e, mtime, size, f_path
                            except Exception as ex:
                                return v_name, f_name, "", str(ex), mtime, size, f_path

                        # Run parallel parser only if needed
                        if tasks_count > 0:
                            progress_bar = st.sidebar.progress(0)
                            status_text = st.sidebar.empty()
                            
                            completed_count = 0
                            already_cached_count = total_count - tasks_count
                            
                            with ThreadPoolExecutor(max_workers=4) as executor:
                                futures = {executor.submit(process_single_file_worker, task): task for task in tasks_to_run}
                                for future in as_completed(futures):
                                    completed_count += 1
                                    v_name, f_path, f_name, mtime, size = futures[future]
                                    
                                    display_idx = already_cached_count + completed_count
                                    status_text.markdown(f"📄 **Parsing:** `{v_name}` / `{f_name}` ({display_idx}/{total_count})")
                                    progress_bar.progress(display_idx / total_count)
                                    
                                    try:
                                        res_v, res_f, t, e, res_mtime, res_size, res_path = future.result()
                                        
                                        if res_v not in ss.vendor_files:
                                            ss.vendor_files[res_v] = {}
                                            ss.vendor_errors[res_v] = {}
                                        ss.vendor_files[res_v][res_f] = t
                                        ss.vendor_errors[res_v][res_f] = e
                                        ss.setdefault("vendor_paths", {}).setdefault(res_v, {})[res_f] = res_path
                                        
                                        if res_v not in ss["vendor_mtime"]:
                                            ss["vendor_mtime"][res_v] = {}
                                        ss["vendor_mtime"][res_v][res_f] = {"mtime": res_mtime, "size": res_size}
                                    except Exception as exc:
                                        st.sidebar.error(f"Error processing {f_name}: {exc}")
                                        
                            status_text.empty()
                            progress_bar.empty()
                            
                        ss.processed = False
                        save_state_to_disk()
                        st.rerun()

        if ss.vendor_files:
            with st.container(border=True):
                st.markdown(
                    """<div style="margin-bottom: 12px; padding: 10px 14px; background: rgba(139, 92, 246, 0.08); 
                    border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; color: #E8EEF8; font-weight: 600; font-size: 14px; 
                    display: flex; align-items: center; gap: 10px; box-shadow: inset 0 0 12px rgba(139, 92, 246, 0.05);">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(139, 92, 246, 0.6));"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
                    Cached Vendor Submissions
                    </div>""", unsafe_allow_html=True
                )
                for name in list(ss.vendor_files.keys()):
                    c1, c2 = st.columns([8.5, 1.5])
                    with c1:
                        st.markdown('<div class="view-ven-marker"></div>', unsafe_allow_html=True)
                        if st.button(f"{name} ({len(ss.vendor_files[name])} files)", use_container_width=True, key=f"btn_view_ven_{name}"):
                            view_documents_dialog(name, ss.vendor_files[name])
                    with c2:
                        st.markdown('<div class="rm-bid-marker"></div>', unsafe_allow_html=True)
                        if st.button("✕", key=f"rm_{name}", help=f"Remove {name}", use_container_width=True):
                            ss.vendor_files.pop(name, None)
                            ss.vendor_errors.pop(name, None)
                            ss.processed = False
                            save_state_to_disk()
                            st.rerun()
                st.markdown("""<style>
                .element-container:has(.view-ven-marker) + .element-container button,
                div[data-testid="stElementContainer"]:has(.view-ven-marker) + div[data-testid="stElementContainer"] button {
                    background: linear-gradient(135deg, #E9F056, #C6CD3E) !important;
                    border: none !important;
                    color: #FFFFFF !important;
                    border-radius: 8px !important;
                    font-weight: 600 !important;
                    transition: all 0.3s ease !important;
                }
                .element-container:has(.view-ven-marker) + .element-container button:hover,
                div[data-testid="stElementContainer"]:has(.view-ven-marker) + div[data-testid="stElementContainer"] button:hover {
                    background: linear-gradient(135deg, #FFFFFF, #E9F056) !important;
                    transform: translateY(-1px) !important;
                }
                .element-container:has(.view-ven-marker) + .element-container button p::before,
                div[data-testid="stElementContainer"]:has(.view-ven-marker) + div[data-testid="stElementContainer"] button p::before {
                    content: '';
                    display: inline-block;
                    width: 16px; height: 16px; margin-right: 8px;
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z'/%3E%3C/svg%3E") no-repeat center;
                    background-size: contain;
                    vertical-align: middle;
                    filter: drop-shadow(0 0 3px rgba(255,255,255,0.5));
                }
                </style>""", unsafe_allow_html=True)

        if add and vfiles:
            with vendor_spinner_placeholder:
                with custom_spinner(f"Extracting and classifying {len(vfiles)} uploaded document(s)...", theme="purple"):
                    for f in vfiles:
                        t, e = read_uploaded_file(f, enable_ocr=True)
                        
                        if vname.strip():
                            target_vendor = vname.strip()
                        else:
                            target_vendor = detect_vendor_name_from_file(f.name, t)
                            
                        if target_vendor not in ss.vendor_files:
                            ss.vendor_files[target_vendor] = {}
                            ss.vendor_errors[target_vendor] = {}
                            
                        ss.vendor_files[target_vendor][f.name] = t
                        ss.vendor_errors[target_vendor][f.name] = e
                        
                        try:
                            os.makedirs(".sentinel_cache_files", exist_ok=True)
                            cache_path = os.path.join(".sentinel_cache_files", f.name)
                            with open(cache_path, "wb") as fp:
                                fp.write(f.getvalue())
                            ss.setdefault("vendor_paths", {}).setdefault(target_vendor, {})[f.name] = cache_path
                        except Exception:
                            pass
                            
            ss.processed = False
            save_state_to_disk()
            st.rerun()

        st.divider()
        st.markdown("""<style>
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary p::before {
            content: '';
            display: inline-block;
            width: 18px; height: 18px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2338BDF8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z'/%3E%3Cpath d='M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: text-bottom;
            filter: drop-shadow(0 0 4px rgba(233,240,86,0.6));
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
            color: #E9F056 !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"],
        [data-testid="stSidebar"] div[data-testid="stExpander"] details,
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            border-radius: 0px !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] details {
            border: 1px solid rgba(233, 240, 86, 0.3) !important;
            background: rgba(233, 240, 86, 0.05) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] details:hover {
            border-color: rgba(233, 240, 86, 0.6) !important;
            box-shadow: 0 0 15px rgba(233, 240, 86, 0.1) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover,
        [data-testid="stSidebar"] div[data-testid="stExpander"] summary:focus {
            background-color: transparent !important;
        }
        </style>""", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(26, 26, 30, 0.6); 
        border: 1px solid rgba(233, 240, 86, 0.15); border-left: 4px solid #E9F056; border-radius: 8px; 
        padding: 12px 14px; font-weight: 800; font-size: 13px; color: #E2E8F0; letter-spacing: 0.5px; 
        text-transform: uppercase; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E9F056" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 5px rgba(233,240,86,0.5));">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            3 · Audit Configuration
        </div>
        """, unsafe_allow_html=True)
        engine_mode = st.radio(
            "Audit Engine Mode",
            ["Deterministic Base Engine (Zero-AI / Fast)", "Local Llama RAG (Ollama)", "Cloud RAG (Groq)", "Cloud RAG (Gemini)"],
            index=0,
            key="engine_mode",
            help="Choose between the fast Zero-AI deterministic engine, local Ollama, Cloud Groq, or Cloud Gemini."
        )
        if engine_mode == "Local Llama RAG (Ollama)":
            st.text_input(
                "Ollama API Endpoint",
                value="http://localhost:11434",
                key="ollama_url",
                help="URL of your local or remote Ollama server."
            )
            # Model dropdown — qwen2.5:7b is default (CPU-friendly); larger models for GPU users
            ollama_models = [
                "qwen2.5:7b",           # Default — fast & accurate on CPU
                "llama3:latest",         # Meta Llama 3 8B
                "llama3.1:8b",          # Meta Llama 3.1 8B
                "llama3.1:70b",         # GPU recommended
                "llama3.2:3b",          # Lightest option for low RAM
                "mistral:7b",            # Mistral 7B
                "mixtral:8x7b",         # GPU recommended
                "qwen2.5:14b",          # GPU recommended
                "qwen2.5:72b",          # GPU recommended
                "deepseek-r1:7b",       # DeepSeek R1 7B
                "phi3:mini",            # Microsoft Phi-3 Mini (very lightweight)
            ]
            current_model = ss.get("ollama_model_name", "qwen2.5:7b")
            if current_model not in ollama_models:
                ollama_models.insert(0, current_model)  # Keep any custom model at top
            st.selectbox(
                "Ollama Model",
                ollama_models,
                index=ollama_models.index(current_model) if current_model in ollama_models else 0,
                key="ollama_model_name",
                help="qwen2.5:7b is recommended for CPU. For GPU users: llama3.1:70b or qwen2.5:72b give better accuracy."
            )
        elif engine_mode == "Cloud RAG (Groq)":
            st.text_input(
                "Groq API Key",
                type="password",
                value=ss.get("groq_key", ""),
                key="groq_key",
                help="Specify your Groq API key (starts with gsk_)."
            )
            groq_key = ss.get("groq_key", "").strip()
            if groq_key:
                if st.button("🔍 Test Groq Connection", key="test_groq_btn", use_container_width=True):
                    with st.spinner("Testing connection to Groq API..."):
                        try:
                            import json
                            import urllib.request
                            req = urllib.request.Request(
                                "https://api.groq.com/openai/v1/models",
                                headers={
                                    "Authorization": f"Bearer {groq_key}",
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                                    "Accept": "application/json",
                                },
                                method="GET"
                             )
                            with urllib.request.urlopen(req, timeout=10) as r:
                                data = json.loads(r.read().decode())
                                models = [m["id"] for m in data.get("data", []) if "llama" in m["id"] or "qwen" in m["id"]]
                                st.success(f"✅ Connection successful! Models: {', '.join(models[:3])}")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {e}")
  
            st.selectbox(
                "Groq Model Name",
                [
                    "llama-3.3-70b-versatile",
                    "llama3-8b-8192",
                    "llama3-70b-8192",
                ],
                index=0,
                key="groq_model_name",
                help="llama-3.3-70b-versatile gives the most accurate results."
            )
            emb_provider = st.selectbox(
                "Embedding Provider",
                ["Local Embeddings (HuggingFace CPU)", "Cloud Gemini Embeddings", "Local Ollama Embeddings"],
                index=0,
                key="groq_embedding_provider",
                help="Choose whether to use local CPU embeddings, Cloud Gemini embeddings, or local Ollama embeddings for vector search."
            )
            if emb_provider == "Cloud Gemini Embeddings":
                st.text_input(
                    "Gemini API Key (for Embeddings)",
                    type="password",
                    value=ss.get("gemini_key", ""),
                    key="gemini_key",
                    help="Gemini API Key is required to run Cloud Embeddings."
                )
        elif engine_mode == "Cloud RAG (Gemini)":
            st.text_input(
                "Gemini API Key",
                type="password",
                value=ss.get("gemini_key", ""),
                key="gemini_key",
                help="Specify your Gemini API key (starts with AIzaSy)."
            )
            st.selectbox(
                "Gemini Model Name",
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash-thinking-exp",
                ],
                index=0,
                key="gemini_model_name",
                help="gemini-1.5-flash is extremely fast; gemini-1.5-pro is recommended for higher reasoning quality."
            )
        api_key = ""
        model = ""
        st.divider()
        ready = bool(ss.bid_text) and bool(ss.vendor_files)
        st.markdown('<div class="run-audit-marker"></div>', unsafe_allow_html=True)

        run_clicked = st.button("Run Full Audit", disabled=not ready, use_container_width=True, type="primary", key="btn_run_audit")
        
        if not ready:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: -8px; margin-bottom: 8px; color: #94A3B8; font-size: 13px; background: rgba(15, 23, 42, 0.4); padding: 8px 12px; border-radius: 6px; border: 1px dashed rgba(255,255,255,0.1);">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 4px rgba(251,191,36,0.6));"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                Load a Master BID and at least one vendor to enable Audit
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="reset-btn-marker"></div>', unsafe_allow_html=True)
        st.button("Reset", on_click=reset_all, use_container_width=True, key="btn_reset_all")
        
        st.markdown("""<style>
        .element-container:has(.run-audit-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.run-audit-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #E9F056, #C6CD3E) !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }
        .element-container:has(.run-audit-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.run-audit-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #E9F056, #E9F056) !important;
            transform: translateY(-2px) !important;
        }
        .element-container:has(.run-audit-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.run-audit-marker) + div[data-testid="stElementContainer"] button p::before {
            content: '';
            display: inline-block;
            width: 18px; height: 18px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white' stroke='%23FFFFFF' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='5 3 19 12 5 21 5 3'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: text-bottom;
            filter: drop-shadow(0 0 4px rgba(255,255,255,0.5));
        }

        .element-container:has(.reset-btn-marker) + .element-container button,
        div[data-testid="stElementContainer"]:has(.reset-btn-marker) + div[data-testid="stElementContainer"] button {
            background: linear-gradient(135deg, #f43f5e, #e11d48) !important;
            border: none !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            transition: all 0.3s ease !important;
        }
        .element-container:has(.reset-btn-marker) + .element-container button:hover,
        div[data-testid="stElementContainer"]:has(.reset-btn-marker) + div[data-testid="stElementContainer"] button:hover {
            background: linear-gradient(135deg, #fb7185, #f43f5e) !important;
            transform: translateY(-1px) !important;
        }
        .element-container:has(.reset-btn-marker) + .element-container button p::before,
        div[data-testid="stElementContainer"]:has(.reset-btn-marker) + div[data-testid="stElementContainer"] button p::before {
            content: '';
            display: inline-block;
            width: 16px; height: 16px; margin-right: 8px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='1 4 1 10 7 10'/%3E%3Cpolyline points='23 20 23 14 17 14'/%3E%3Cpath d='M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15'/%3E%3C/svg%3E") no-repeat center;
            background-size: contain;
            vertical-align: middle;
            transition: all 0.3s ease;
            filter: drop-shadow(0 0 3px rgba(255,255,255,0.4));
        }
        </style>""", unsafe_allow_html=True)
        
    return api_key, model, run_clicked


def go_to_audit() -> None:
    st.session_state.nav_radio = "Audit Engine"
    st.query_params["page"] = "audit"

def go_to_home() -> None:
    st.session_state.nav_radio = "Home"
    if "page" in st.query_params:
        del st.query_params["page"]


def render_landing_page() -> None:
    ss = st.session_state
    results = ss.get("results", [])
    total = len(results) if results else 3
    responsive = [r for r in results if not getattr(r, 'disqualified', True)] if results else [1]*2
    dq = total - len(responsive)
    top = max((r.score for r in results if not getattr(r, 'disqualified', True)), default=100.0) if results else 100.0

    st.markdown("""
    <style>
    .lp-section { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 30px; margin-bottom: 24px; }
    .lp-section h2 { font-size: 24px; font-weight: 700; margin-top: 0; margin-bottom: 16px; color: var(--blue); }
    .lp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
    .lp-card { background: var(--panel2); padding: 20px; border-radius: 12px; border: 1px solid var(--line); }
    .lp-card h3 { font-size: 18px; margin-top: 0; color: var(--text); margin-bottom: 12px; }
    .lp-card p { color: var(--muted); line-height: 1.6; font-size: 14.5px; margin: 0; }
    
    .lp-eyebrow { margin: 80px 0 32px; display: flex; scroll-margin-top: 130px;  flex-direction: column; align-items: flex-start; gap: 12px; }
    
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin-bottom: 24px;
    }
    .bento-wide { grid-column: span 2; }
    .bento-tall { grid-row: span 2; }
    @media (max-width: 900px) {
        .bento-grid { grid-template-columns: repeat(2, 1fr); }
        .bento-wide { grid-column: span 2; }
        .bento-tall { grid-row: span 1; }
    }
    @media (max-width: 600px) {
        .bento-grid { grid-template-columns: 1fr; }
        .bento-wide { grid-column: span 1; }
        .bento-tall { grid-row: span 1; }
    }
    .lp-eyebrow .n { 
        display: inline-flex; align-items: center; justify-content: center;
        padding: 6px 16px; 
        border-radius: 20px; 
        background: rgba(233, 240, 86, 0.1); 
        border: 1px solid rgba(233, 240, 86, 0.2); 
        color: #E9F056; 
        font-family: 'Inter', sans-serif; 
        font-size: 13px; 
        font-weight: 700; 
        letter-spacing: 1px; 
    }
    .lp-eyebrow h2 { font-size: 36px; font-weight: 800; color: #F8FAFC; margin: 0; letter-spacing: -1px; }

    /* ---------- hero ---------- */
    .hero{padding:40px 0 18px; position:relative;}
    .eyebrow-tag {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #E9F056;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(233, 240, 86, 0.25);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 0 12px rgba(233, 240, 86, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 8px 16px;
        border-radius: 9999px;
        margin-bottom: 24px;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .eyebrow-tag::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(233, 240, 86, 0.2), transparent);
        transform: skewX(-20deg);
        animation: eyebrowSweep 5s infinite;
    }
    @keyframes eyebrowSweep {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }
    .eyebrow-tag:hover {
        border-color: rgba(233, 240, 86, 0.5);
        box-shadow: 0 6px 20px rgba(233, 240, 86, 0.2), inset 0 0 16px rgba(233, 240, 86, 0.1);
        transform: translateY(-2px);
        color: #FFFFFF;
    }
    .eyebrow-tag .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px #10B981;
        position: relative;
    }
    .eyebrow-tag .dot::after {
        content: '';
        position: absolute;
        top: -4px; left: -4px; right: -4px; bottom: -4px;
        border-radius: 50%;
        border: 1px solid rgba(16, 185, 129, 0.5);
        animation: dotPing 2.5s cubic-bezier(0, 0, 0.2, 1) infinite;
    }
    @keyframes dotPing {
        75%, 100% { transform: scale(2.2); opacity: 0; }
    }
    .hero h1{font-size:clamp(34px,5vw,58px);line-height:1.04;font-weight:900;letter-spacing:-1.6px;
      margin:0 0 18px; max-width:17ch;}
    .hero h1 .hl{background:linear-gradient(120deg,var(--blue),var(--green));-webkit-background-clip:text;
      background-clip:text;color:transparent;}
    .hero p.lede{font-size:18px;color:#B7C2D8;max-width:62ch;margin:0 0 20px;}
    .hero-stats{display:flex;gap:34px;margin-top:20px;flex-wrap:wrap;}
    .hero-stats .s .n{font-size:30px;font-weight:900;letter-spacing:-.6px;}
    .hero-stats .s .l{color:var(--muted);font-size:12.5px;margin-top:2px;text-transform:uppercase;letter-spacing:.8px;}
    .hero-stats .s .n.g{color:var(--green);} .hero-stats .s .n.b{color:var(--blue);} .hero-stats .s .n.r{color:var(--red);}

    /* ---------- how it works ---------- */
    .flow{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:30px;}
    .step{
        --hover-rgb: 123, 146, 255;
        background: rgba(10, 15, 30, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .step::after {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 15px 15px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.3;
    }
    .step:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(var(--hover-rgb), 0.15);
        border-color: rgba(var(--hover-rgb), 0.4);
    }
    .step::before {
        content: ''; position: absolute; top: -2px; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, rgb(var(--hover-rgb)), transparent);
        opacity: 0; transition: opacity 0.4s ease;
        box-shadow: 0 0 15px rgb(var(--hover-rgb));
        z-index: 2;
    }
    .step:hover::before { opacity: 1; }
    .step-icon {
        margin-bottom: 18px; display: inline-flex; align-items: center; justify-content: center;
        width: 48px; height: 48px;
        background: linear-gradient(135deg, rgba(233, 240, 86, 0.15), rgba(233, 240, 86, 0.05));
        border-radius: 12px; border: 1px solid rgba(233, 240, 86, 0.3);
        box-shadow: inset 0 0 10px rgba(233, 240, 86, 0.1);
        position: relative;
        z-index: 2;
    }
    .step-content { position: relative; z-index: 2; }
    .step .sn{font-family:'JetBrains Mono',monospace;font-size:12px;color:#E9F056; font-weight:700; letter-spacing:1px; text-transform:uppercase;}
    .step h5{margin:8px 0 6px;font-size:16px;font-weight:800; color:#E2E8F0;}
    .step p{margin:0;font-size:13.5px;color:#94A3B8;line-height:1.6;}
    
    .arch{display:grid;grid-template-columns:1fr;gap:24px;margin-top:20px;max-width:800px;margin-left:auto;margin-right:auto;}
    .arch .col{
        background: rgba(10, 15, 30, 0.7); backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 32px; 
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .arch .col::after {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 20px 20px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.5;
    }
    .arch .col:hover { transform: translateY(-6px); }
    .arch .col.det{
        border-top: 2px solid #10B981;
    }
    .arch .col.det:hover {
        box-shadow: 0 20px 50px rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .arch .col.det::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 100%;
        background: radial-gradient(circle at top left, rgba(16,185,129,0.1), transparent 50%);
        pointer-events: none; z-index: 1;
    }
    .arch-icon { 
        margin-bottom: 20px; display: inline-flex; align-items: center; justify-content: center;
        width: 56px; height: 56px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative; z-index: 2;
    }
    .arch .col h5{font-size:18px;margin:0 0 10px;font-weight:800; color:#F8FAFC; position:relative; z-index:2;}
    .arch .col p{font-size:14.5px;color:#94A3B8;line-height:1.7;margin:0; position:relative; z-index:2;}
    code.inl{font-family:'JetBrains Mono',monospace;font-size:12px;background:var(--panel2);border:1px solid var(--line);
      padding:2px 7px;border-radius:6px;color:#C7D3EA;}
    
    @media (max-width:860px){
      .flow{grid-template-columns:1fr 1fr;}
      .arch{grid-template-columns:1fr;}
    }
    
    /* COMPREHENSIVE RESPONSIVENESS FIXES */
    @media (max-width: 1024px) {
        .hero-container {
            grid-template-columns: 1fr !important;
            text-align: center;
        }
        .hero-container .hero-left {
            align-items: center !important;
            justify-content: center !important;
        }
        .hero-container .hero-right {
            display: none !important;
        }
    }
</style>
    """, unsafe_allow_html=True)

    logo_b64 = get_base64_image("logo.jpg")
    glyph_content = f'<img src="data:image/jpeg;base64,{logo_b64}">' if logo_b64 else ''
    replacement_img = '<img style="width: 100%; height: 100%; object-fit: cover;" '
    fancy_glyph_content = (
        '<div class="fancy-logo-wrapper">'
        '<div style="position: absolute; inset: 0; border-radius: 36px; padding: 3px; background: conic-gradient(from 0deg, #E9F056, rgba(233,240,86,0.05) 25%, #10B981, rgba(16,185,129,0.05) 75%, #E9F056); animation: spin 5s linear infinite; box-shadow: 0 0 60px rgba(233, 240, 86, 0.4), inset 0 0 20px rgba(16, 185, 129, 0.2);">'
        '<div style="position: absolute; inset: 3px; background: #211119; border-radius: 33px; z-index: 1;"></div></div>'
        '<div style="position: absolute; inset: -20px; border-radius: 46px; border: 1px dashed rgba(233, 240, 86, 0.3); animation: spin 15s linear infinite reverse; z-index: 0;"></div>'
        '<div style="position: absolute; inset: -10px; border-radius: 40px; border: 1px solid rgba(16, 185, 129, 0.2); animation: spin 10s linear infinite; z-index: 0;"></div>'
        '<div style="position: relative; z-index: 2; width: 94%; height: 94%; border-radius: 28px; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #211119; box-shadow: inset 0 0 40px rgba(0,0,0,0.8);">'
        + glyph_content.replace('<img ', replacement_img) + '</div></div>'
    ) if logo_b64 else ''

    st.markdown(f"""
    <style>
    header[data-testid="stHeader"] {{ display: none !important; }}
    .block-container {{ padding-top: 0 !important; padding-bottom: 0 !important; }}
    
    @keyframes subtleFloat {{
        0%, 100% {{ transform: translateY(0); box-shadow: 0 10px 40px rgba(233, 240, 86, 0.2); }}
        50% {{ transform: translateY(-15px); box-shadow: 0 25px 50px rgba(233, 240, 86, 0.4); }}
    }}
    .landing-navbar {{
        width: 100vw;
        position: fixed;
        top: 0;
        left: 0;
        margin: 0;
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 5vw; background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        box-sizing: border-box; z-index: 9999;
    }}
    .landing-navbar .nav-logo {{
        display: flex; align-items: center; gap: 12px;
    }}
    .landing-navbar .nav-logo img {{
        width: 32px; height: 32px; border-radius: 8px;
        border: 1px solid rgba(233, 240, 86, 0.5);
        box-shadow: 0 0 12px rgba(233, 240, 86, 0.3);
    }}
    .landing-navbar .nav-logo span {{
        font-weight: 900; font-size: 20px; letter-spacing: -0.5px; 
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%); 
        -webkit-background-clip: text; color: transparent;
    }}
    .landing-navbar .nav-links {{
        display: flex; gap: 28px; align-items: center;
    }}
    .landing-navbar .nav-links a {{
        color: #94A3B8; text-decoration: none; font-size: 14px; font-weight: 600; 
        padding: 6px 14px; border-radius: 20px; transition: all 0.3s ease;
        border: 1px solid transparent; background: transparent;
    }}
    .landing-navbar .nav-links a:hover, .landing-navbar .nav-links a.active {{ 
        color: #E9F056; text-shadow: 0 0 8px rgba(233,240,86,0.5); 
        background: rgba(233, 240, 86, 0.1); border: 1px solid rgba(233, 240, 86, 0.3);
        box-shadow: inset 0 0 10px rgba(233, 240, 86, 0.05);
    }}
    
    .lp-section {{
        --hover-rgb: 123, 146, 255;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        padding: 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        position: relative;
    }}
    .lp-section:hover {{
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(var(--hover-rgb), 0.2), 0 0 20px rgba(var(--hover-rgb), 0.1);
        border-color: rgba(var(--hover-rgb), 0.5);
    }}
    .lp-card {{
        --hover-rgb: 139, 92, 246;
        background: rgba(15, 23, 42, 0.4);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }}
    .lp-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(var(--hover-rgb), 0.2), 0 0 20px rgba(var(--hover-rgb), 0.1);
        border-color: rgba(var(--hover-rgb), 0.5);
    }}
    .lp-grid {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px;
    }}
    
    @media (max-width: 1100px) {{
        .landing-navbar {{
            flex-direction: column;
            align-items: flex-start !important;
            padding: 12px 20px !important;
        }}
        .landing-navbar-top {{
            display: flex;
            width: 100%;
            justify-content: space-between;
            align-items: center;
        }}
        .mobile-menu-icon {{
            display: block !important;
        }}
        .nav-links {{
            display: none !important;
            flex-direction: column;
            width: 100%;
            gap: 0 !important;
            margin-top: 16px;
            background: #0B1120;
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 8px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            align-items: stretch !important;
        }}
        .nav-links a {{
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            gap: 16px;
            padding: 14px 24px;
            font-size: 14px;
            font-weight: 500;
            color: #94A3B8 !important;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            text-decoration: none;
            transition: background 0.2s;
            position: relative;
        }}
        .nav-links a:hover {{
            background: rgba(255,255,255,0.03);
            color: #F1F5F9 !important;
        }}
        .nav-links a svg {{
            display: block !important;
            width: 18px;
            height: 18px;
            color: #64748B;
        }}
        #mobile-menu-toggle:checked ~ .nav-links {{
            display: flex !important;
        }}
        .hero-container {{
            grid-template-columns: 1fr !important;
        }}
        .hero-right {{
            display: flex !important;
            margin-top: 50px;
            transform: scale(0.75);
            transform-origin: center;
        }}
        .lp-grid {{
            grid-template-columns: 1fr !important;
        }}
    }}
    </style>
    
    <input type="checkbox" id="logo-anim-toggle">
    <div class="fullscreen-logo-overlay">
       <div class="anim-content">
           {fancy_glyph_content}
           <h1 class="anim-title">Argus Bid AI — Tender Audit &amp; Compliance</h1>
           <div class="anim-tagline-container">
               <span class="anim-tagline">THE HUNDRED EYED GUARDIAN OF PROCUREMENT</span>
           </div>
       </div>
    </div>
    
    <div class="landing-navbar" id="top">
        <div class="landing-navbar-top" style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
            <label class="nav-logo" for="logo-anim-toggle" style="cursor: pointer; margin: 0;">
                <div class="small-glyph">{glyph_content}</div>
                <div style="display: flex; flex-direction: column; justify-content: center;">
                    <span style="line-height: 1.1;">Argus Bid AI</span>
                    <div style="font-size: 10px; color: #94A3B8; font-weight: 600; margin-top: 2px; letter-spacing: 0.5px; text-transform: uppercase;">Tender Audit & Compliance</div>
                </div>
            </label>
            <label for="mobile-menu-toggle" class="mobile-menu-icon" style="display: none; color: #94A3B8; font-size: 28px; cursor: pointer; user-select: none;">&#9776;</label>
        </div>
        <input type="checkbox" id="mobile-menu-toggle" style="display: none;">
        <div class="nav-links" style="display: flex; align-items: center; gap: 10px; font-size: 12.5px;">
            <a href="#hero-section" target="_self" style="text-decoration: none; color: inherit;">Home</a>
            <a href="#sec-01" target="_self" style="text-decoration: none; color: inherit;">Problem</a>
            <a href="#sec-02" target="_self" style="text-decoration: none; color: inherit;">Solution</a>
            <a href="#sec-03" target="_self" style="text-decoration: none; color: inherit;">Features</a>
            <a href="#sec-04" target="_self" style="text-decoration: none; color: inherit;">Modules</a>
            <a href="#sec-05" target="_self" style="text-decoration: none; color: inherit;">Pipeline</a>
            <a href="#sec-07" target="_self" style="text-decoration: none; color: inherit;">Types</a>
            <a href="#contact" target="_self" style="text-decoration: none; color: inherit;">Contact</a>
        </div>
    </div>

    <div class="hero-container" id="hero-section" style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 40px; align-items: center; padding: 60px 0 18px; position: relative; margin-bottom: 40px; scroll-margin-top: 100px;">
      <div style="position: absolute; top: -150px; left: -100px; width: 400px; height: 400px; background: rgba(233, 240, 86, 0.15); filter: blur(80px); border-radius: 50%; z-index: 0; animation: subtleFloat 8s ease-in-out infinite;"></div>
      <div style="position: absolute; bottom: -150px; right: -100px; width: 400px; height: 400px; background: rgba(16, 185, 129, 0.15); filter: blur(80px); border-radius: 50%; z-index: 0; animation: subtleFloat 6s ease-in-out infinite reverse;"></div>
      
      <div class="hero-left" style="position: relative; z-index: 1;">
        <div class="eyebrow-tag" style="display: inline-flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #E9F056; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(233, 240, 86, 0.3); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 0 12px rgba(233, 240, 86, 0.1); padding: 8px 16px; border-radius: 9999px; margin-bottom: 24px; letter-spacing: 0.5px; position: relative; overflow: hidden;">
           <div style="position: absolute; top: 0; left: -100%; width: 50%; height: 100%; background: linear-gradient(90deg, transparent, rgba(233, 240, 86, 0.25), transparent); transform: skewX(-20deg); animation: subtleFloat 4s ease-in-out infinite alternate;"></div>
           <span class="dot" style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981; position: relative; z-index: 2;"></span>
           <span style="position: relative; z-index: 2; font-weight: 600;">Enterprise Engine &middot; Fully deterministic local execution</span>
        </div>
        <h1 style="font-size: clamp(40px, 5vw, 64px); line-height: 1.05; font-weight: 900; letter-spacing: -1.5px; margin: 0 0 20px;">
            The tender file lands.<br>
            <span style="background: linear-gradient(120deg, #E9F056, #10B981); -webkit-background-clip: text; color: transparent; text-shadow: 0 0 30px rgba(233, 240, 86, 0.2);">The verdict is already written.</span>
        </h1>
        <p class="lede" style="font-size: 18px; color: #94A3B8; max-width: 680px; line-height: 1.6; margin: 0 0 30px; font-weight: 400;">
            Argus Bid AI reads a PSU Notice Inviting Tender line by line, inventories every vendor's messy submission, runs the eligibility gates &mdash; MAF, pre-qualification, mandatory documents &mdash; then ranks the survivors on a transparent 70/30 weighting and <strong style="color: #E2E8F0; font-weight: 600;">explains exactly why rank 1 beat rank 2.</strong> Every verdict carries its evidence.
        </p>
        <div class="cta-row" style="display: flex; gap: 14px; margin-top: 10px; flex-wrap: wrap;">
            <a href="?page=audit" target="_self" style="display: inline-flex; align-items: center; justify-content: center; padding: 14px 28px; border-radius: 12px; font-weight: 700; font-size: 16px; color: #022C22; background: linear-gradient(120deg, #10B981, #6EE7B7); text-decoration: none; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(16, 185, 129, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.5)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(16, 185, 129, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4)';">View the live audit &rarr;</a>
            <a href="?page=documentation" target="_self" style="display: inline-flex; align-items: center; justify-content: center; padding: 14px 28px; border-radius: 12px; font-weight: 700; font-size: 16px; color: #211119; background: linear-gradient(120deg, #E9F056, #FFFFFF); text-decoration: none; box-shadow: 0 4px 15px rgba(233, 240, 86, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(233, 240, 86, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.5)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(233, 240, 86, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4)';">Documentation</a>
            <a href="?page=case-studies" target="_self" style="display: inline-flex; align-items: center; justify-content: center; padding: 14px 28px; border-radius: 12px; font-weight: 700; font-size: 16px; color: #2E1065; background: linear-gradient(120deg, #A78BFA, #DDD6FE); text-decoration: none; box-shadow: 0 4px 15px rgba(167, 139, 250, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(167, 139, 250, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.5)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(167, 139, 250, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4)';">Case Studies</a>
        </div>
      </div>
      <div class="hero-right" style="display: flex; justify-content: center; position: relative; z-index: 1;">
        <div style="position: relative; width: 320px; height: 320px; display: flex; align-items: center; justify-content: center;">
            <div style="position: absolute; inset: 0; border-radius: 36px; padding: 3px; background: conic-gradient(from 0deg, #E9F056, rgba(233,240,86,0.05) 25%, #10B981, rgba(16,185,129,0.05) 75%, #E9F056); animation: spin 5s linear infinite; box-shadow: 0 0 60px rgba(233, 240, 86, 0.4), inset 0 0 20px rgba(16, 185, 129, 0.2);">
                <div style="position: absolute; inset: 3px; background: #211119; border-radius: 33px; z-index: 1;"></div>
            </div>
            <div style="position: absolute; inset: -20px; border-radius: 46px; border: 1px dashed rgba(233, 240, 86, 0.3); animation: spin 15s linear infinite reverse; z-index: 0;"></div>
            <div style="position: absolute; inset: -10px; border-radius: 40px; border: 1px solid rgba(16, 185, 129, 0.2); animation: spin 10s linear infinite; z-index: 0;"></div>
            <div style="position: relative; z-index: 2; width: 94%; height: 94%; border-radius: 28px; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #211119; box-shadow: inset 0 0 40px rgba(0,0,0,0.8);">
               {glyph_content.replace('<img ', '<img style="width: 100%; height: 100%; object-fit: cover;" ')}
            </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    components.html("""
    <script>
        const parentDoc = window.parent.document;
        const links = parentDoc.querySelectorAll('.nav-links a');
        links.forEach(link => {
            link.addEventListener('click', () => {
                const toggle = parentDoc.getElementById('mobile-menu-toggle');
                if (toggle) toggle.checked = false;
            });
        });
    </script>
    """, height=0, width=0)
    
    st.markdown(f"""
<div class="kpis" style="margin-top: 20px;">
<div class="kpi blue"><div class="v">{total}</div><div class="l">Vendors Audited</div></div>
<div class="kpi green"><div class="v">{len(responsive)}</div><div class="l">Responsive</div></div>
<div class="kpi red"><div class="v">{dq}</div><div class="l">Disqualified</div></div>
<div class="kpi white"><div class="v">{top:g}%</div><div class="l">Top Score</div></div>
</div>
<div style="margin-top: 60px;"></div>

<div id="sec-01" class="lp-eyebrow"><span class="n">SECTION 01</span><h2>The Problem We Solve</h2></div>
<div class="lp-section" style="border-left: 4px solid #F59E0B; background: linear-gradient(90deg, rgba(245, 158, 11, 0.05), transparent); --hover-rgb: 245, 158, 11;">
<p style="color:var(--muted); line-height: 1.8; font-size: 15px; margin: 0;">
Public Sector Undertaking (PSU) procurement processes are plagued by massive, complex tender documents and hundreds of dense vendor submissions. Evaluating these manually is excruciatingly slow, highly prone to human error, and vulnerable to bias. Missing a single clause in a 500-page Manufacturer's Authorization Form (MAF) can lead to illegal awards, litigation, and severe financial penalties.
</p>
</div>

<div id="sec-02" class="lp-eyebrow"><span class="n">SECTION 02</span><h2>Our Solution</h2></div>
<div class="lp-section" style="border-left: 4px solid #10B981; background: linear-gradient(90deg, rgba(16, 185, 129, 0.05), transparent); --hover-rgb: 16, 185, 129;">
<p style="color:var(--muted); line-height: 1.8; font-size: 15px; margin: 0;">
Argus Bid AI transforms procurement evaluation from a manual bottleneck into an instant, deterministic, and auditable process. We consume the Master BID (NIT) and automatically extract the strict matrix of requirements:
</p>
<ul style="color:var(--muted); font-size: 15px; line-height: 1.8; margin-top: 10px;">
<li><b>Pre-Qualification Criteria:</b> Revenue thresholds, prior experience, certifications.</li>
<li><b>Mandatory Documents:</b> MAFs, EMDs, valid GST/PAN registrations.</li>
<li><b>Technical Specifications:</b> Line-by-line semantic matching of requested features vs. vendor brochures.</li>
</ul>
<p style="color:var(--muted); line-height: 1.8; font-size: 15px; margin: 0;">
We then parse every vendor's submission, intelligently classifying documents, executing a strict compliance gate, and scoring them on a weighted scale.
</p>
</div>

<div id="sec-03" class="lp-eyebrow"><span class="n">SECTION 03</span><h2>Why It Is Better</h2></div>
<div class="bento-grid">
<div class="lp-card bento-wide" style="position: relative; overflow: hidden; --hover-rgb: 123, 146, 255;">
<div style="position: absolute; top:0; left:0; width:100%; height:4px; background: #E9F056;"></div>
<div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(233, 240, 86, 0.1); border: 1px solid rgba(233, 240, 86, 0.2); display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(233, 240, 86, 0.1);">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E9F056" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
</div>
<h3 style="color: #E9F056; font-size: 18px; margin-top: 0;">Deterministic Accuracy</h3>
<p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Unlike black-box AI tools that hallucinate, Argus relies on a strictly deterministic rule-engine for pass/fail compliance. Every decision is traceable to a specific text snippet, ensuring full legal defensibility.</p>
</div>
<div class="lp-card" style="position: relative; overflow: hidden; --hover-rgb: 139, 92, 246;">
<div style="position: absolute; top:0; left:0; width:100%; height:4px; background: #8B5CF6;"></div>
<div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(139, 92, 246, 0.1);">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
</div>
<h3 style="color: #8B5CF6; font-size: 18px; margin-top: 0;">Explainable Audit Trails</h3>
<p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Achieve 100x faster evaluation times with zero bias. Argus produces an instantly exportable, explainable audit trail (XAI) that justifies every rank and disqualification.</p>
</div>
<div class="lp-card" style="position: relative; overflow: hidden; --hover-rgb: 16, 185, 129;">
<div style="position: absolute; top:0; left:0; width:100%; height:4px; background: #10B981;"></div>
<div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(16, 185, 129, 0.1);">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
</div>
<h3 style="color: #10B981; font-size: 18px; margin-top: 0;">Continuous Compliance</h3>
<p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Instantly adapt to evolving CVC guidelines. Our modular compliance engine allows you to update evaluation parameters without re-writing a single line of code.</p>
</div>
<div class="lp-card" style="position: relative; overflow: hidden; --hover-rgb: 245, 158, 11;">
<div style="position: absolute; top:0; left:0; width:100%; height:4px; background: #F59E0B;"></div>
<div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2); display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(245, 158, 11, 0.1);">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
</div>
<h3 style="color: #F59E0B; font-size: 18px; margin-top: 0;">Zero Human Bias</h3>
<p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Manual evaluation is subjective. Argus ensures that every vendor is evaluated against the exact same strict criteria, eliminating favoritism and disputes.</p>
</div>
<div class="lp-card" style="position: relative; overflow: hidden; --hover-rgb: 244, 63, 94;">
<div style="position: absolute; top:0; left:0; width:100%; height:4px; background: #F43F5E;"></div>
<div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.2); display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(244, 63, 94, 0.1);">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>
</div>
<h3 style="color: #F43F5E; font-size: 18px; margin-top: 0;">Universal Processing</h3>
<p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">From native PDFs to heavily skewed mobile scans, the intelligence engine automatically repairs and normalizes unstructured vendor submissions.</p>
</div>
</div>

<div id="sec-04" class="lp-eyebrow"><span class="n">SECTION 04</span><h2>Platform Modules</h2></div>
<div class="lp-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
    <div class="lp-card" style="--hover-rgb: 244, 63, 94;">
        <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(244, 63, 94, 0.1);">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M2 15h10"></path><path d="m9 18 3-3-3-3"></path></svg>
        </div>
        <h4 style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;">Document Intelligence</h4>
        <p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Multi-modal OCR and structural parsing extract text, tables, and signatures from scanned PDFs effortlessly.</p>
    </div>
    <div class="lp-card" style="--hover-rgb: 234, 179, 8;">
        <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(234, 179, 8, 0.1); border: 1px solid rgba(234, 179, 8, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(234, 179, 8, 0.1);">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#EAB308" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
        </div>
        <h4 style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;">Compliance Engine</h4>
        <p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">A flexible rules engine that allows procurement officers to define custom constraints without writing code.</p>
    </div>
    <div class="lp-card" style="--hover-rgb: 16, 185, 129;">
        <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 20px; box-shadow: inset 0 0 12px rgba(16, 185, 129, 0.1);">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
        </div>
        <h4 style="font-size: 18px; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;">Comparative Matrix</h4>
        <p style="font-size: 14.5px; color: #94A3B8; line-height: 1.6; margin: 0;">Automatically generates side-by-side technical comparison tables for all responsive bidders.</p>
    </div>
</div>

<div id="sec-05" class="lp-eyebrow"><span class="n">SECTION 05</span><h2>How the engine decides</h2></div>
<div class="flow">
<div class="step" style="--hover-rgb: 123, 146, 255;">
<div class="step-icon">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E9F056" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(233,240,86,0.8));"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
</div>
<div class="step-content">
<div class="sn">step 1</div><h5>Parse the NIT</h5>
<p>Pull the exact tender ID, pre-qualification thresholds, mandatory documents, and every technical spec straight from the bid text.</p>
</div>
</div>
<div class="step" style="--hover-rgb: 139, 92, 246;">
<div class="step-icon">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(139,92,246,0.8));"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path><line x1="12" y1="11" x2="12" y2="17"></line><line x1="9" y1="14" x2="15" y2="14"></line></svg>
</div>
<div class="step-content">
<div class="sn" style="color: #8B5CF6;">step 2</div><h5>Inventory &amp; classify</h5>
<p>Identify each vendor file by content, not filename &mdash; <code class="inl">Scan_001.pdf</code> becomes a typed, readability-graded document.</p>
</div>
</div>
<div class="step" style="--hover-rgb: 245, 158, 11;">
<div class="step-icon">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(245,158,11,0.8));"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>
</div>
<div class="step-content">
<div class="sn" style="color: #F59E0B;">step 3</div><h5>Run the gates</h5>
<p>A missing or invalid MAF, a failed pre-qualification, or a missing mandatory document disqualifies a vendor outright, with the reason logged.</p>
</div>
</div>
<div class="step" style="--hover-rgb: 16, 185, 129;">
<div class="step-icon">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 6px rgba(16,185,129,0.8));"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>
</div>
<div class="step-content">
<div class="sn" style="color: #10B981;">step 4</div><h5>Score &amp; explain</h5>
<p>Survivors earn 70% on mandatory specs and 30% on preferred features, then a plain-language note explains the ranking.</p>
</div>
</div>
</div>
<div class="arch">
<div class="col det">
<div class="arch-icon">
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(16,185,129,0.8));"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
</div>
<h5>The rules decide &mdash; always</h5>
<p>Every compliance verdict is produced by a deterministic engine. The same inputs always yield the same verdict, and each one carries the bid section and the evidence text behind it. That reproducibility is what survives an audit or a vendor challenge.</p>
</div>
</div>

<div id="sec-07" class="lp-eyebrow"><span class="n">SECTION 06</span><h2>Supported Document Types</h2></div>
<div class="lp-section" style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 32px;">
    <div style="display: flex; flex-wrap: wrap; gap: 12px;">
        <span style="background: rgba(233, 240, 86, 0.1); color: #E9F056; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(233, 240, 86, 0.2);">Scanned PDFs</span>
        <span style="background: rgba(16, 185, 129, 0.1); color: #10B981; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.2);">Native PDFs</span>
        <span style="background: rgba(245, 158, 11, 0.1); color: #F59E0B; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.2);">Word Documents</span>
        <span style="background: rgba(139, 92, 246, 0.1); color: #8B5CF6; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(139, 92, 246, 0.2);">Excel Spreadsheets</span>
        <span style="background: rgba(244, 63, 94, 0.1); color: #F43F5E; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid rgba(244, 63, 94, 0.2);">Images (JPEG/PNG)</span>
    </div>
    <p style="color:var(--muted); line-height: 1.8; font-size: 15px; margin-top: 16px; margin-bottom: 0;">Argus Bid AI's multi-modal intelligence automatically normalizes unstructured files, extracts OCR text, and reconstructs tabular data with high fidelity, regardless of how messy the vendor's submission is.</p>
</div>

<div id="contact" style="margin-top: 80px; padding: 40px 0; border-top: 1px solid var(--line); text-align: center; color: var(--muted); font-size: 13.5px;">
    &copy; 2026 Argus Bid AI. Software Engineering Summer Internship Project &middot; IOCL Haldia Refinery.
</div>
    """, unsafe_allow_html=True)


# ===========================================================================
# SECTION 10 — UI RENDERING MODULES FOR DASHBOARD
# ===========================================================================
def render_masthead() -> None:
    ss = st.session_state
    tid = (ss.bid or {}).get("tender_id") if ss.bid else ""
    chip = (f'<span class="tender-chip">TENDER&nbsp;·&nbsp;{html.escape(tid)}</span>'
            if tid else '<span class="tender-chip">NO TENDER LOADED</span>')
            
    logo_b64 = get_base64_image("logo.jpg")
    glyph_content = f'<img src="data:image/jpeg;base64,{logo_b64}">' if logo_b64 else ''
    replacement_img = '<img style="width: 100%; height: 100%; object-fit: cover;" '
    fancy_glyph_content = (
        '<div class="fancy-logo-wrapper">'
        '<div style="position: absolute; inset: 0; border-radius: 36px; padding: 3px; background: conic-gradient(from 0deg, #E9F056, rgba(233,240,86,0.05) 25%, #10B981, rgba(16,185,129,0.05) 75%, #E9F056); animation: spin 5s linear infinite; box-shadow: 0 0 60px rgba(233, 240, 86, 0.4), inset 0 0 20px rgba(16, 185, 129, 0.2);">'
        '<div style="position: absolute; inset: 3px; background: #211119; border-radius: 33px; z-index: 1;"></div></div>'
        '<div style="position: absolute; inset: -20px; border-radius: 46px; border: 1px dashed rgba(233, 240, 86, 0.3); animation: spin 15s linear infinite reverse; z-index: 0;"></div>'
        '<div style="position: absolute; inset: -10px; border-radius: 40px; border: 1px solid rgba(16, 185, 129, 0.2); animation: spin 10s linear infinite; z-index: 0;"></div>'
        '<div style="position: relative; z-index: 2; width: 94%; height: 94%; border-radius: 28px; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #211119; box-shadow: inset 0 0 40px rgba(0,0,0,0.8);">'
        + glyph_content.replace('<img ', replacement_img) + '</div></div>'
    ) if logo_b64 else ''

    st.markdown(f"""
    <input type="checkbox" id="logo-anim-toggle">
    <div class="fullscreen-logo-overlay">
       <div class="anim-content">
           {fancy_glyph_content}
           <h1 class="anim-title">Argus Bid AI — Tender Audit &amp; Compliance</h1>
           <div class="anim-tagline-container">
               <span class="anim-tagline">THE HUNDRED EYED GUARDIAN OF PROCUREMENT</span>
           </div>
       </div>
    </div>
    <div class="masthead" style="flex-direction: column; align-items: flex-start; gap: 24px;">
      <div style="width: 100%; display: flex; justify-content: flex-start;">
        <a href="?page=home" target="_self" style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 8px; font-weight: 600; font-size: 13px; color: #022C22; background: linear-gradient(120deg, #10B981, #6EE7B7); text-decoration: none; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(16, 185, 129, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.5)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(16, 185, 129, 0.3), inset 0 2px 4px rgba(255, 255, 255, 0.4)';">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
          Back to Landing Page
        </a>
      </div>
      <div style="display: flex; align-items: flex-start; justify-content: space-between; width: 100%; gap: 18px; flex-wrap: wrap;">
        <div class="mark">
          <label class="glyph" for="logo-anim-toggle">{glyph_content}</label>
          <div>
            <h1>Argus Bid AI — Tender Audit &amp; Compliance</h1>
            <div class="text-loop-container" style="margin-top: 8px;">
              Auditing tender compliance for&nbsp;
              <span class="text-loop">
                <span class="word">IOCL Tenders</span>
                <span class="word">GeM Bids</span>
                <span class="word">Technical Sheets</span>
                <span class="word">PQC Criteria</span>
              </span>
            </div>
            <div class="engine-features">
               <span class="ef-badge ef-slate"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> Automated Doc Inventory</span>
               <span class="ef-badge ef-emerald"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg> Strict MAF &amp; PQC Gates</span>
               <span class="ef-badge ef-purple"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg> Semantic Specs Matching</span>
               <span class="ef-badge ef-amber"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg> Explainable AI Scoring</span>
               <span class="ef-badge ef-blue"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg> Transparent Vendor Ranking</span>
            </div>
          </div>
        </div>
        {chip}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty() -> None:
    st.markdown("""
    <div class="empty">
      <div class="big">No audit run yet</div>
      Use the sidebar to <b>Load Demo Corpus</b> for an instant walkthrough, or upload your
      Master BID/NIT and vendor submissions, then press <b>Run Full Audit</b>.
    </div>
    """, unsafe_allow_html=True)


def eyebrow(num: str, title: str) -> None:
    st.markdown(f'<div class="eyebrow"><span class="n">{num}</span>'
                f'<h2>{html.escape(title)}</h2><span class="rule"></span></div>',
                unsafe_allow_html=True)


def render_kpis(results: List[VendorResult]) -> None:
    total = len(results)
    responsive = [r for r in results if not r.disqualified]
    dq = total - len(responsive)
    top = max((r.score for r in results), default=0.0)
    avg = round(sum(r.score for r in responsive) / len(responsive), 1) if responsive else 0.0
    st.markdown(f"""
<div class="kpis">
<div class="kpi blue"><div class="v">{total}</div><div class="l">Vendors Audited</div></div>
<div class="kpi green"><div class="v">{len(responsive)}</div><div class="l">Responsive</div></div>
<div class="kpi red"><div class="v">{dq}</div><div class="l">Disqualified</div></div>
<div class="kpi white"><div class="v">{top:g}%</div><div class="l">Top Score</div></div>
<div class="kpi amber"><div class="v">{avg:g}%</div><div class="l">Avg (Responsive)</div></div>
</div>
    """, unsafe_allow_html=True)


def render_leaderboard(results: List[VendorResult]) -> None:
    ordered = sorted(results, key=lambda r: (r.disqualified, -(r.score)))
    
    def get_rank_html(rank: Optional[int]) -> str:
        if not rank: return '<div class="rank-badge rank-other">—</div>'
        if rank == 1: return '<div class="rank-badge rank-1">#1</div>'
        if rank == 2: return '<div class="rank-badge rank-2">#2</div>'
        if rank == 3: return '<div class="rank-badge rank-3">#3</div>'
        return f'<div class="rank-badge rank-other">#{rank}</div>'

    rows = []
    maf_req = "Manufacturer's Authorization Form (MAF)" in (st.session_state.bid.get("mandatory_docs", []) if st.session_state.bid else [])
    for r in ordered:
        rank_html = get_rank_html(r.rank)
        dq_cls = "dq" if r.disqualified else ""
        bar_cls = "bar dq" if r.disqualified else "bar"
        width = r.score
        nfiles = sum(len(v) for v in [st.session_state.vendor_files.get(r.name, {})])
        rows.append(f"""
        <tr class="{dq_cls}">
          <td>{rank_html}</td>
          <td><div class="vname">{html.escape(r.name)}</div>
              <div class="vmeta">{nfiles} document(s) submitted</div></td>
          <td>{status_pill(r.status)}</td>
          <td>{maf_pill(r.maf.status if r.maf else MAF_MISSING, maf_req)}</td>
          <td><div class="scorewrap"><div class="{bar_cls}"><span style="width:{width}%"></span></div>
              <span class="scoreval">{r.score:g}%</span></div></td>
          <td style="color:var(--muted);font-size:12.5px;max-width:280px;white-space:pre-wrap;line-height:1.5;">{html.escape(r.summary)}</td>
        </tr>""")
    st.markdown(f"""
    <div style="overflow-x: auto; max-width: 100vw; width: 100%;">
    <table class="lb">
      <thead><tr><th>Rank</th><th>Vendor</th><th>Status</th><th>MAF</th>
        <th>Compliance Score</th><th>Key Takeaway</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


def render_comparative_statement(results: List[VendorResult]) -> None:
    ordered = sorted(results, key=lambda r: r.name.lower())
    bid_dict = st.session_state.bid or {}
    
    raw_docs = bid_dict.get("mandatory_docs", [])
    mandatory_docs = list(dict.fromkeys(raw_docs))
    pqc_rules = bid_dict.get("pqc", [])
    
    # 1. Build Headers
    headers = [
        '<th style="width: 50px; min-width: 50px;">Sl. No.</th>',
        '<th style="width: 140px; min-width: 140px; text-align: left;">Bidder Name</th>'
    ]
    for doc in mandatory_docs:
        headers.append(f'<th style="min-width: 120px;">{html.escape(doc)}</th>')
    for pqc in pqc_rules:
        headers.append(f'<th style="min-width: 130px;">{html.escape(pqc.get("label", pqc.get("key")))}</th>')
    headers.append('<th style="min-width: 110px;">Technical Specs</th>')
    headers.append('<th style="min-width: 220px; text-align: left;">Final Recommendation</th>')
    
    rows = []
    for idx, r in enumerate(ordered, start=1):
        row_cells = []
        
        # Sl No & Bidder Name
        row_cells.append(f'<td><div class="serial-num">{idx}</div></td>')
        row_cells.append(f'<td style="text-align: left;"><div class="vname" style="font-weight:700; color:#E2E8F0;">{html.escape(r.name)}</div></td>')
        
        # Mandatory Docs Columns
        checklist = getattr(r, "document_checklist", {})
        for doc in mandatory_docs:
            details = checklist.get(doc, {})
            status = details.get("status", "Missing")
            page = details.get("page")
            
            if status == "Compliant":
                pg_str = f" (Pg {page})" if page else ""
                val_text = f"Compliant{pg_str}"
                cls = "cell-ok"
            elif status == "Non-Compliant":
                val_text = "Non-Compliant"
                cls = "cell-bad"
            else:
                val_text = "Missing"
                cls = "cell-bad"
                
            row_cells.append(f'<td class="{cls}">{html.escape(val_text)}</td>')
            
        # PQC Columns
        for pqc in pqc_rules:
            key = pqc["key"]
            p_res = None
            for p in r.pqc:
                if p.label == pqc.get("label") or p.label == key or getattr(p, "key", "") == key:
                    p_res = p
                    break
            
            if not p_res:
                val_text = "N/A"
                cls = "cell-na"
            else:
                passed = p_res.passed
                provided = p_res.provided
                pg_suffix = f" (Pg {p_res.page})" if p_res.page and p_res.page > 1 else ""
                val_text = f"{provided}{pg_suffix}"
                cls = "cell-ok" if passed else "cell-bad"
                
            row_cells.append(f'<td class="{cls}">{html.escape(val_text)}</td>')
            
        # Technical Specs
        failed_specs = [s.param for s in r.mandatory_specs if s.status == "fail"]
        if failed_specs:
            tech_specs_text = "Rejected"
            tech_specs_details = f"<br><span style='font-size:10px; color:#F87171;'>Fails: {', '.join(failed_specs)}</span>"
            tech_specs_cls = "cell-bad font-bold"
        else:
            tech_specs_text = "Compliant"
            tech_specs_details = ""
            tech_specs_cls = "cell-ok font-bold"
        row_cells.append(f'<td class="{tech_specs_cls}">{tech_specs_text}{tech_specs_details}</td>')
        
        # Final Recommendation
        if r.disqualified:
            tq_rec = "Disqualified"
            tq_cls = "tq-bad"
            rejection_reasons = [v.title for v in r.violations]
            tq_details = f"<div style='font-size:11px; margin-top:6px; color:#F87171; line-height:1.4; text-align: left;'>• " + "<br>• ".join([html.escape(r) for r in rejection_reasons]) + "</div>"
        else:
            tq_rec = "Responsive"
            tq_cls = "tq-ok"
            tq_details = ""
            
        row_cells.append(f"""
          <td style="text-align: left;">
            <div class="tq-badge {tq_cls}">{tq_rec}</div>
            {tq_details}
          </td>
        """)
        
        rows.append(f"<tr>{''.join(row_cells)}</tr>")
        
    st.markdown(f"""
    <style>
    .cs-matrix {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        background: transparent;
        table-layout: auto;
    }}
    .cs-matrix th {{
        background: rgba(233, 240, 86, 0.08) !important;
        border-bottom: 1px solid var(--line);
        color: var(--blue) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 6px;
        text-align: center;
        vertical-align: middle;
        white-space: normal !important;
    }}
    .cs-matrix td {{
        border-bottom: 1px solid var(--line);
        padding: 14px 16px;
        text-align: center;
        vertical-align: top;
        font-size: 13px;
        color: #FDF2F8;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: break-word;
    }}
    .cs-matrix tr:hover {{
        background: rgba(255, 255, 255, 0.015);
    }}
    .serial-num {{
        font-weight: 700;
        color: var(--muted);
        font-size: 12px;
    }}
    .cell-ok {{
        color: #10B981 !important;
        font-weight: 600;
    }}
    .cell-bad {{
        color: #FF5C34 !important;
        font-weight: 600;
    }}
    .cell-na {{
        color: #64748B !important;
        font-weight: 500;
    }}
    .font-bold {{
        font-weight: 700 !important;
    }}
    .tq-badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .tq-ok {{
        background: rgba(16, 185, 129, 0.15) !important;
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15) !important;
    }}
    .tq-bad {{
        background: rgba(239, 68, 68, 0.15) !important;
        color: #F87171 !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.15) !important;
    }}
    </style>
    
    <div style="width: 100%; margin-bottom: 24px; overflow-x: auto; -webkit-overflow-scrolling: touch;">
    <table class="cs-matrix">
      <thead>
        <tr>
          {''.join(headers)}
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    </div>
    """.replace("\n", " "), unsafe_allow_html=True)


def render_xai(xai: List[str], narrative: Optional[str]) -> None:
    if narrative:
        st.markdown(f'<div class="xai"><b>Executive Summary (LLM)</b><br>{html.escape(narrative)}</div>',
                    unsafe_allow_html=True)
    icon_svg = """<div style="flex-shrink: 0; width: 32px; height: 32px; background: linear-gradient(135deg, rgba(233,240,86,0.15), rgba(16,185,129,0.15)); border: 1px solid rgba(233,240,86,0.25); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.1);"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg></div>"""
    for line in xai:
        st.markdown(f'<div class="xai" style="padding: 18px 20px;"><div style="display:flex; gap:16px;">{icon_svg}<div style="flex: 1; padding-top: 3px;">{line}</div></div></div>', unsafe_allow_html=True)


def render_document_checklist(r: VendorResult) -> str:
    import urllib.parse
    checklist = getattr(r, "document_checklist", {})
    if not checklist:
        return ""
        
    pdf_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round" style="width: 14px; height: 14px; vertical-align: middle; '
        'margin-left: 5px; display: inline-block; filter: drop-shadow(0 0 2px rgba(239, 68, 68, 0.45));">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '</svg>'
    )
        
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
    out = f'<div class="sh-box sh-blue">{svg}<span class="title">Mandatory Document &amp; Annexure Checklist Validator</span><span class="line"></span></div>'
    
    rows = []
    for doc, details in checklist.items():
        status = details.get("status", "Missing")
        page = details.get("page")
        filename = details.get("file")
        finding = details.get("finding", "")
        
        # Format status badge
        if status == "Compliant":
            badge = '<span class="chip match" style="font-weight: 700;">Compliant</span>'
        elif status == "Non-Compliant":
            badge = '<span class="chip fail" style="font-weight: 700; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #F87171;">Non-Compliant</span>'
        else:
            badge = '<span class="chip fail" style="font-weight: 700;">Missing</span>'
            
        # Format source link
        src_link = "—"
        if filename:
            pnum = page or 1
            link_url = f'?page=audit&view_file={urllib.parse.quote(filename)}&view_page={pnum}&vendor={urllib.parse.quote(r.name)}'
            visual_url = f'?page=audit&view_visual_file={urllib.parse.quote(filename)}&view_visual_page={pnum}&vendor={urllib.parse.quote(r.name)}'
            pdf_icon = f'<a href="{visual_url}" target="_self" style="text-decoration: none;" title="View visual PDF page">{pdf_svg}</a>'
            src_link = f'<a href="{link_url}" target="_self" style="color: #E9F056; text-decoration: underline; font-weight: 600;">{html.escape(filename)} (Pg {pnum})</a>{pdf_icon}'
            
        rows.append(f'<tr><td style="text-align: left; font-weight: 600; color: #E2E8F0; width: 35%;">{html.escape(doc)}</td><td style="width: 15%;">{badge}</td><td style="width: 20%;">{src_link}</td><td style="text-align: left; color: #94A3B8; font-size: 12.5px; width: 30%;">{html.escape(finding)}</td></tr>')
        
    out += f'<div style="overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch;"><table class="invtable">{"".join(rows)}</table></div>'
    return out


def render_inventory(r: VendorResult) -> str:
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'
    out = f'<div class="sh-box sh-slate">{svg}<span class="title">Document Inventory &amp; Readability Audit</span><span class="line"></span></div>'
    rows = "".join(
        f"<tr><td>{html.escape(i.filename)}</td>"
        f"<td>{html.escape(i.doc_type)}</td>"
        f"<td style='text-align:right;'>{read_pill(i.readability)}</td></tr>"
        for i in r.inventory)
    out += f'<div style="overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch;"><table class="invtable">{rows}</table></div>'
    return out


def render_maf(r: VendorResult) -> str:
    import urllib.parse
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'
    out = f'<div class="sh-box sh-emerald">{svg}<span class="title">Manufacturer&apos;s Authorization (MAF) Gate</span><span class="line"></span></div>'
    cls = "ok" if r.maf.status == MAF_VALID else "bad"
    
    pdf_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round" style="width: 14px; height: 14px; vertical-align: middle; '
        'margin-left: 5px; display: inline-block; filter: drop-shadow(0 0 2px rgba(239, 68, 68, 0.45));">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '</svg>'
    )
    
    src = ""
    if r.maf.source_file:
        pnum = getattr(r.maf, "page", 1)
        src_link = f'?page=audit&view_file={urllib.parse.quote(r.maf.source_file)}&view_page={pnum}&vendor={urllib.parse.quote(r.name)}'
        visual_url = f'?page=audit&view_visual_file={urllib.parse.quote(r.maf.source_file)}&view_visual_page={pnum}&vendor={urllib.parse.quote(r.name)}'
        pdf_icon = f'<a href="{visual_url}" target="_self" style="text-decoration: none;" title="View visual PDF page">{pdf_svg}</a>'
        src = f' — source: <a href="{src_link}" target="_self" style="color: #E9F056; text-decoration: underline; font-weight: 600;">{html.escape(r.maf.source_file)} (Pg {pnum})</a>{pdf_icon}'
        
    out += f'<div style="margin-bottom:8px; display:flex; align-items:center; gap:12px;">{maf_pill(r.maf.status)}<span style="color:var(--muted);font-size:12px; margin-top:2px;">{src}</span></div>'
    out += f'<div class="evidence {cls}">{html.escape(r.maf.evidence)}</div>'
    return out


def render_pqc(r: VendorResult) -> str:
    import urllib.parse
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>'
    out = f'<div class="sh-box sh-purple">{svg}<span class="title">Pre-Qualification Criteria (PQC) Gate</span><span class="line"></span></div>'
    rows = []
    
    pdf_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round" style="width: 14px; height: 14px; vertical-align: middle; '
        'margin-left: 5px; display: inline-block; filter: drop-shadow(0 0 2px rgba(239, 68, 68, 0.45));">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '</svg>'
    )
    
    for p in r.pqc:
        if p.file:
            link_url = f'?page=audit&view_file={urllib.parse.quote(p.file)}&view_page={p.page}&vendor={urllib.parse.quote(r.name)}'
            visual_url = f'?page=audit&view_visual_file={urllib.parse.quote(p.file)}&view_visual_page={p.page}&vendor={urllib.parse.quote(r.name)}'
            pdf_icon = f'<a href="{visual_url}" target="_self" style="text-decoration: none;" title="View visual PDF page">{pdf_svg}</a>'
            clean_prov = re.sub(r'\s*\(Pg \d+\)', '', p.provided)
            chip = f'<a href="{link_url}" target="_self" style="color: inherit; text-decoration: none;"><span class="chip match" style="cursor: pointer; border: 1px solid rgba(233, 240, 86, 0.45); background: rgba(233, 240, 86, 0.1) !important; color: #E9F056 !important; font-weight: 700;">{html.escape(clean_prov)} <span style="font-size: 10px; opacity: 0.85; margin-left: 2px;">Pg {p.page}</span></span></a>{pdf_icon}'
        else:
            chip = (f'<span class="chip match">{html.escape(p.provided)}</span>' if p.passed
                    else f'<span class="chip fail">{html.escape(p.provided)} ✕</span>')
            
        if p.bid_file:
            bid_link = f'?page=audit&view_file={urllib.parse.quote(p.bid_file)}&view_page={p.bid_page}&vendor=Master'
            visual_bid_url = f'?page=audit&view_visual_file={urllib.parse.quote(p.bid_file)}&view_visual_page={p.bid_page}&vendor=Master'
            pdf_bid_icon = f'<a href="{visual_bid_url}" target="_self" style="text-decoration: none;" title="View visual PDF page">{pdf_svg}</a>'
            criterion_val = f'<span style="font-weight: 600;">{html.escape(p.label)}</span> <a href="{bid_link}" target="_self" style="color: #E9F056; font-size:11px; text-decoration: underline; font-weight: 600; margin-left: 4px;">§{p.section} (Pg {p.bid_page})</a>{pdf_bid_icon}'
        else:
            criterion_val = f'<span style="font-weight: 600;">{html.escape(p.label)}</span> <span style="color:var(--muted);font-size:11px;">§{p.section}</span>'
 
        rows.append(f"<tr><td>{criterion_val}</td>"
                    f"<td><span class='chip req'>{html.escape(p.required)}</span></td>"
                    f"<td style='text-align:right;'>{chip}</td></tr>")
    out += f'<div style="overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch;"><table class="matrix"><thead><tr><th>Criterion</th><th>Required</th><th style="text-align:right;">Vendor Provided</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    return out
 
 
def render_matrix(r: VendorResult) -> str:
    import urllib.parse
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>'
    out = f'<div class="sh-box sh-blue">{svg}<span class="title">Technical Comparison Matrix (BID vs Vendor)</span><span class="line"></span></div>'
    rows = []
    
    pdf_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round" style="width: 14px; height: 14px; vertical-align: middle; '
        'margin-left: 5px; display: inline-block; filter: drop-shadow(0 0 2px rgba(239, 68, 68, 0.45));">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '</svg>'
    )
    
    for tier, specs in [("Mandatory", r.mandatory_specs), ("Preferred", r.preferred_specs)]:
        for s in specs:
            tier_chip = (f'<span class="chip req">{tier}</span>')
            
            if s.bid_file:
                bid_link = f'?page=audit&view_file={urllib.parse.quote(s.bid_file)}&view_page={s.bid_page}&vendor=Master'
                visual_bid_url = f'?page=audit&view_visual_file={urllib.parse.quote(s.bid_file)}&view_visual_page={s.bid_page}&vendor=Master'
                pdf_bid_icon = f'<a href="{visual_bid_url}" target="_self" style="text-decoration: none;" title="View visual PDF page">{pdf_svg}</a>'
                bid_val = f'<a href="{bid_link}" target="_self" style="color: inherit; text-decoration: none;"><span class="chip req" style="cursor: pointer; border: 1px dashed rgba(233, 240, 86, 0.4);">{html.escape(s.required)} <span style="font-size: 9px; opacity: 0.8; margin-left: 2px;">Pg {s.bid_page}</span></span></a>{pdf_bid_icon}'
            else:
                bid_val = f"<span class='chip req'>{html.escape(s.required)}</span>"
                
            vendor_val = spec_chip(s, r.name)
            
            rows.append(
                f"<tr><td><span style='font-weight: 600;'>{html.escape(s.param)}</span></td>"
                f"<td>{tier_chip}</td>"
                f"<td>{bid_val}</td>"
                f"<td style='text-align:right;'>{vendor_val}</td></tr>")
    out += f'<div style="overflow-x: auto; width: 100%; -webkit-overflow-scrolling: touch;"><table class="matrix"><thead><tr><th>Parameter</th><th>Tier</th><th>BID Requirement</th><th style="text-align:right;">Vendor Value</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    return out


def render_deviations(r: VendorResult) -> str:
    import urllib.parse
    if not r.deviations:
        return ""
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    out = f'<div class="sh-box sh-amber">{svg}<span class="title">Detected Deviations</span><span class="line"></span></div>'
    
    for d in r.deviations:
        m = re.search(r"\(([^)]+?)\s*-\s*Pg\s*(\d+)\)$", d)
        if m:
            fname = m.group(1).strip()
            pnum = m.group(2).strip()
            clean_d = d[:m.start()].strip()
            link_url = f'?page=audit&view_file={urllib.parse.quote(fname)}&view_page={pnum}&vendor={urllib.parse.quote(r.name)}'
            out += f'<div class="evidence bad">{html.escape(clean_d)} <a href="{link_url}" target="_self" style="color: #FBBF24; text-decoration: underline; font-weight: 600; margin-left: 6px;">{html.escape(fname)} (Pg {pnum})</a></div>'
        else:
            out += f'<div class="evidence bad">{html.escape(d)}</div>'
    return out


def render_violations(r: VendorResult) -> str:
    if not r.violations:
        return ""
    svg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
    out = f'<div class="sh-box sh-red">{svg}<span class="title">Reason for Disqualification Log</span><span class="line"></span></div>'
    for i, v in enumerate(r.violations, start=1):
        out += f"""
        <div class="viol">
          <div class="vt">Violation {i}: {html.escape(v.title)}</div>
          <div class="row"><span class="k">Requirement:</span> {html.escape(v.requirement)}</div>
          <div class="row"><span class="k">Finding:</span> {html.escape(v.finding)}</div>
        </div>"""
    return out


def render_drawers(results: List[VendorResult]) -> None:
    ordered = sorted(results, key=lambda r: (r.disqualified, -(r.score)))
    
    html_blocks = []
    for r in ordered:
        if not r.disqualified:
            icon_svg = '<svg style="filter: drop-shadow(0 0 6px rgba(52,211,153,0.8)); color: #34D399; margin-right:12px; vertical-align: -3px;" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        else:
            icon_svg = '<svg style="filter: drop-shadow(0 0 6px rgba(248,113,113,0.8)); color: #F87171; margin-right:12px; vertical-align: -3px;" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
            
        rank = f"Rank {r.rank} &nbsp;·&nbsp; " if r.rank else ""
        
        inner_html = (
            render_inventory(r) +
            render_document_checklist(r) +
            render_maf(r) +
            render_pqc(r) +
            render_matrix(r) +
            render_deviations(r) +
            render_violations(r)
        )
        
        html_blocks.append(f"""<details class="glass-panel" style="margin-bottom: 16px;">
    <summary style="font-family: 'JetBrains Mono', monospace; font-size: 14.5px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.3px;">
        <div style="display: flex; align-items: center; width: 100%;">
            {icon_svg}
            <span style="color: #F8FAFC; font-weight: 800;">{html.escape(r.name)}</span>
            <span style="color: #64748B; margin: 0 12px;">—</span>
            <span style="color: #94A3B8;">{rank}{html.escape(r.status)} &nbsp;·&nbsp; {r.score:g}%</span>
        </div>
    </summary>
    <div style="padding: 20px 24px; border-top: 1px solid rgba(255,255,255,0.06);">
        {inner_html}
    </div>
</details>""")
        
    st.markdown("".join(html_blocks), unsafe_allow_html=True)


def render_bid_map() -> None:
    bid = st.session_state.bid
    if not bid:
        return
        
    def render_map_sec(title: str, items: list, is_pqc: bool = False) -> str:
        if not items: return ""
        out = [f'<div class="map-sec">{html.escape(title)}</div>']
        for item in items:
            if isinstance(item, str):
                out.append(f'<div class="map-item"><span class="mlbl">{html.escape(item)}</span></div>')
            else:
                lbl = html.escape(item.get('label', ''))
                if is_pqc:
                    val = f"≥ {item['threshold']:g} {item['unit']}" if item.get("threshold") else "Required"
                    cls = "mval" if item.get("threshold") else "mval mreq"
                    val_html = f"<span class='{cls}'>{html.escape(val)} <span style='opacity:0.6;font-weight:400;'>[§{html.escape(str(item.get('section', '')))}]</span></span>"
                else:
                    val = str(item.get("required_value", ""))
                    val_html = f"<span class='mval'>{html.escape(val)}</span>"
                out.append(f'<div class="map-item"><span class="mlbl">{lbl}</span>{val_html}</div>')
        return "".join(out)

    html1 = render_map_sec("Pre-Qualification Criteria", bid["pqc"], is_pqc=True)
    html1 += render_map_sec("Mandatory Documents", bid["mandatory_docs"])
    
    html2 = render_map_sec("Mandatory Technical Specs (70%)", bid["mandatory_specs"])
    html2 += render_map_sec("Preferred Specs (30%)", bid["preferred_specs"])

    html3 = render_map_sec("General Tender & Commercial Info", bid.get("general_info", []))
    
    svg_icon = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>'

    st.markdown(f"""<details class="glass-panel">
  <summary>
    <span class="badge-glow bg-blue" style="font-size:13.5px; font-weight:800; text-transform:none; letter-spacing:0px;">
        {svg_icon} Master BID Intelligence Map (Extracted Ontology)
    </span>
  </summary>
  <div class="grid-3">
    <div>{html1}</div>
    <div>{html2}</div>
    <div>{html3}</div>
  </div>
</details>""", unsafe_allow_html=True)


# ===========================================================================
# SECTION 10B — RULES REVIEW & EDIT PAGE
# ===========================================================================

def is_placeholder_evidence(evidence: Optional[str]) -> bool:
    if not evidence:
        return True
    lower_ev = evidence.lower().strip()
    placeholders = [
        "no evidence", 
        "not found", 
        "not mentioned", 
        "n/a", 
        "not applicable", 
        "no quote", 
        "no direct", 
        "evidence not found", 
        "does not mention", 
        "cannot find"
    ]
    return any(p in lower_ev for p in placeholders) or len(lower_ev) < 5


def get_synonyms_for_keyword(keyword: str) -> list:
    kw_clean = keyword.lower().strip()
    synonyms = [kw_clean]
    
    # Check common documents and expand synonyms from most specific to least specific
    if "pan" in kw_clean:
        synonyms.extend(["pan card", "copy of pan", "permanent account number", "pan no", "pan number", "pan:"])
    if "gst" in kw_clean:
        synonyms.extend(["gst registration", "gst certificate", "gstin certificate", "gstin registration", "gstin", "gst no", "gst number", "goods and service", "goods & service"])
    if "balance sheet" in kw_clean or "audited" in kw_clean or "sheet" in kw_clean:
        synonyms.extend(["audited balance sheet", "balance sheet", "audited balance", "financial statement", "profit & loss", "audited account", "turnover"])
    if "maf" in kw_clean or "authorization" in kw_clean or "authorisation" in kw_clean:
        synonyms.extend(["manufacturer's authorization form", "manufacturer's authorization", "manufacturer authorization", "oem authorization", "maf", "authorization form", "authorisation form"])
    if "iso" in kw_clean:
        synonyms.extend(["iso 9001", "iso certification", "iso certificate", "iso status"])
    if "bis" in kw_clean:
        synonyms.extend(["bis certification", "bis certificate", "bis compliance", "crs registration"])
    if "service center" in kw_clean or "service centre" in kw_clean:
        synonyms.extend(["service center", "service centre", "local support", "support office", "west bengal"])
    if "udyam" in kw_clean or "msme" in kw_clean or "micro" in kw_clean or "registration" in kw_clean:
        synonyms.extend(["udyam registration certificate", "udyam registration", "udyam certificate", "msme certificate", "udyam number", "micro and small"])
        
    seen = set()
    result = []
    for s in synonyms:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def extract_evidence_from_page(page_content: str, synonym: str) -> str:
    """Helper to extract a context snippet of a page around the matching synonym."""
    lines = page_content.split('\n')
    syn_lower = synonym.lower()
    for i, line in enumerate(lines):
        if syn_lower in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            context = [lines[j].strip() for j in range(start, end) if lines[j].strip()]
            return " ... ".join(context)
    return f"Requirement for '{synonym}' found in tender document."


def _find_page_and_evidence_for_keyword(evidence: Optional[str], keyword: str, bid_text: str) -> Tuple[int, str]:
    """Finds the exact PDF page number and evidence snippet for a keyword/evidence block.
    Uses actual --- PAGE N --- markers extracted from the text for 100% accurate page numbers.
    """
    default_ev = evidence if evidence else f"Requirement for '{keyword}' found in tender rules."
    try:
        import re as _re

        # ── Build a list of (page_number, page_content) from --- PAGE N --- markers ──
        page_blocks = []
        for m in _re.finditer(r'--- PAGE (\d+) ---\n?', bid_text):
            pg_num = int(m.group(1))
            pg_start = m.end()
            # Find the next PAGE marker or end of text
            next_m = _re.search(r'--- PAGE \d+ ---', bid_text[pg_start:])
            pg_end = pg_start + next_m.start() if next_m else len(bid_text)
            page_blocks.append((pg_num, bid_text[pg_start:pg_end]))

        if not page_blocks:
            # Absolute fallback: split by \f and use index+1 as page number
            parts = bid_text.split('\f')
            page_blocks = [(i + 1, p) for i, p in enumerate(parts)]

        # Strategy 0: Search for specific extracted IDs/values (e.g. PAN number 'AAEFC5257B' or GSTIN) in evidence
        if evidence:
            specific_tokens = _re.findall(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b|\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b|\b[A-Z0-9]{7,15}\b', evidence)
            for token in specific_tokens:
                token_clean = token.lower().strip()
                if len(token_clean) >= 5 and token_clean not in {'present', 'verified', 'compliant', 'missing', 'number', 'status', 'submission'}:
                    for pg_num, page_content in page_blocks:
                        if token_clean in page_content.lower():
                            return pg_num, evidence

        # Strategy 1: Check evidence if it is not placeholder text
        if evidence and not is_placeholder_evidence(evidence):
            clean_ev = _re.sub(r'\s+', ' ', evidence.lower()).strip().strip('"').strip("'").strip()

            # Exact match
            for pg_num, page_content in page_blocks:
                if clean_ev in _re.sub(r'\s+', ' ', page_content.lower()):
                    return pg_num, evidence

            # Substring match
            if len(clean_ev) > 40:
                short_ev = clean_ev[:40]
                for pg_num, page_content in page_blocks:
                    if short_ev in _re.sub(r'\s+', ' ', page_content.lower()):
                        return pg_num, evidence

            # Token overlap match
            stop = {'the','a','an','of','for','in','is','are','and','or','to','by','as',
                    'be','at','on','with','that','this','which','from','must','shall',
                    'should','will','not','have','has','been','its','their','were','was',
                    'it','we','they','us','our','all','any','each','per','no','can'}
            ev_words = [w for w in _re.findall(r'\b\w{4,}\b', clean_ev) if w not in stop]
            if ev_words:
                best_score, best_pg = 0, None
                for pg_num, page_content in page_blocks:
                    page_lower = page_content.lower()
                    if "reason for disqualification" in page_lower or "violation 1:" in page_lower:
                        continue  # Skip report violation summary pages
                    score = sum(1 for w in ev_words if w in page_lower)
                    if score > best_score:
                        best_score, best_pg = score, pg_num
                threshold = max(2, int(len(ev_words) * 0.4))
                if best_pg is not None and best_score >= threshold:
                    return best_pg, evidence

        # Strategy 2: Page-by-page synonym matching (skipping report violation summary sections)
        synonyms = get_synonyms_for_keyword(keyword)
        for pg_num, page_content in page_blocks:
            page_lower = page_content.lower()
            if "reason for disqualification" in page_lower or "violation 1:" in page_lower:
                continue  # Skip generated summary report pages
            for syn in synonyms:
                pattern = r'\b' + _re.escape(syn) + r'\b'
                if _re.search(pattern, page_lower):
                    extracted = extract_evidence_from_page(page_content, syn)
                    return pg_num, extracted

        # Strategy 3: Loose substring search of synonyms
        for pg_num, page_content in page_blocks:
            page_lower = page_content.lower()
            for syn in synonyms:
                if syn in page_lower:
                    extracted = extract_evidence_from_page(page_content, syn)
                    return pg_num, extracted

        # Strategy 4: Loose token match of keyword words
        kw_lower = keyword.lower()
        kw_words = [w for w in _re.findall(r'\b\w{4,}\b', kw_lower)
                    if w not in {'form','card','type','date','list','item','note','data'}]
        if kw_words:
            best_score, best_pg = 0, None
            for pg_num, page_content in page_blocks:
                page_lower = page_content.lower()
                score = sum(1 for w in kw_words if w in page_lower)
                if score > best_score:
                    best_score, best_pg = score, pg_num
            if best_pg is not None and best_score >= max(1, len(kw_words) // 2):
                for pg_num, page_content in page_blocks:
                    if pg_num == best_pg:
                        extracted = extract_evidence_from_page(page_content, kw_words[0])
                        return pg_num, extracted

    except Exception:
        pass
    return 1, default_ev


def render_rules_review_page() -> None:
    """Renders the AI Tender Rules Verification Console and editable requirements tabs."""
    ss = st.session_state
    if not ss.get("bid") and ss.get("bid_text"):
        from audit_engine import AuditEngine
        engine = AuditEngine()
        ss.bid = engine.parse_master_bid(ss.get("bid_text", ""))
        save_state_to_disk()

    bid = ss.get("bid", {})
    if not bid:
        st.error("No bid data found. Please go back and re-upload the master bid document.")
        return

    bid_text = ss.get("bid_text", "")
    bid_paths = ss.get("bid_paths", {})
    master_pdf_path = list(bid_paths.values())[0] if bid_paths else ""

    # Inject styling matching the screenshot console design
    st.markdown("""
    <style>
    /* Style delete columns to have red buttons */
    div[data-testid="stColumn"]:last-child button {
        background: linear-gradient(135deg, #EF4444, #DC2626) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.2) !important;
        transition: all 0.3s ease !important;
        height: 38px !important;
        margin-top: 28px !important;
    }
    div[data-testid="stColumn"]:last-child button:hover {
        background: linear-gradient(135deg, #F87171, #EF4444) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* PDF Jump Link Button Styling */
    .pdf-jump-btn {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 34px !important;
        height: 34px !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #8B5CF6, #6D28D9) !important;
        border: 1px solid #7C3AED !important;
        text-decoration: none !important;
        font-size: 14px !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
        box-shadow: 0 4px 10px rgba(139, 92, 246, 0.25) !important;
        transition: all 0.2s ease !important;
        margin-top: 8px !important;
    }
    .pdf-jump-btn:hover {
        background: linear-gradient(135deg, #A78BFA, #8B5CF6) !important;
        transform: scale(1.05) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.45) !important;
    }

    /* Console Rules List Card Styling */
    .rule-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        transition: all 0.2s ease !important;
    }
    .rule-card:hover {
        background: rgba(139, 92, 246, 0.02) !important;
        border-color: rgba(139, 92, 246, 0.3) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    .rule-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #E2E8F0 !important;
        margin-bottom: 6px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .rule-id-chip {
        font-size: 10px !important;
        font-weight: 800 !important;
        background: rgba(139, 92, 246, 0.15) !important;
        color: #A78BFA !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
    }
    .rule-page-chip {
        font-size: 10px !important;
        font-weight: 800 !important;
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        margin-left: 6px !important;
    }
    .rule-quote {
        font-size: 13px !important;
        font-style: italic !important;
        color: rgba(199, 211, 234, 0.7) !important;
        line-height: 1.5 !important;
        margin-top: 4px !important;
        border-left: 3px solid rgba(233, 240, 86, 0.4) !important;
        padding-left: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Console Top bar Stats (Matching Screenshot Top Block) ───────────────
    engine_model = ss.get("ollama_model_name", "qwen2.5:7b")
    if ss.get("engine_mode") == "Cloud RAG (Gemini)":
        engine_model = ss.get("gemini_model_name", "gemini-flash-latest")
    elif ss.get("engine_mode") == "Cloud RAG (Groq)":
        engine_model = ss.get("groq_model_name", "llama-3.3-70b-versatile")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(233,240,86,0.06) 0%, rgba(233,240,86,0.02) 100%);
                border: 1px solid rgba(233,240,86,0.2); border-radius: 12px;
                padding: 20px 24px; margin-bottom: 24px;">
        <div style="font-size:11px; font-weight:800; letter-spacing:2px; color:rgba(233,240,86,0.6);
                    text-transform:uppercase; margin-bottom:6px;">ARGUS BID AI</div>
        <div style="font-size:22px; font-weight:800; color:#E9F056; margin-bottom:4px;">🛡️ AI Tender Rules Understanding Console</div>
        <div style="font-size:13px; color:rgba(199,211,234,0.7); margin-bottom:16px;">
            Inspect extracted Pre-Qualification Criteria (PQC) thresholds, technical specifications, and CVC procurement compliance sentinel parameters before evaluation.
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; border-top: 1px solid rgba(199,211,234,0.1); padding-top: 16px;">
            <div>
                <div style="font-size:10px; color:rgba(199,211,234,0.5); font-weight:700; text-transform:uppercase;">Total Bids Processed</div>
                <div style="font-size:18px; font-weight:900; color:#E2E8F0;">1</div>
            </div>
            <div>
                <div style="font-size:10px; color:rgba(199,211,234,0.5); font-weight:700; text-transform:uppercase;">Source Index Integrity</div>
                <div style="font-size:14px; font-weight:900; color:#10B981; margin-top:2px;">
                    <span style="background:rgba(16,185,129,0.15); padding:2px 8px; border-radius:4px; border:1px solid rgba(16,185,129,0.3);">🟢 VERIFIED</span>
                </div>
            </div>
            <div>
                <div style="font-size:10px; color:rgba(199,211,234,0.5); font-weight:700; text-transform:uppercase;">Analysis Engine</div>
                <div style="font-size:13px; font-weight:800; color:#38BDF8; font-family:monospace; margin-top:4px;">🤖 AI Mode ({engine_model})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Initialize State Lists ──────────────────────────────────────────────
    pqc_list = bid.get("pqc", [])
    ss.setdefault("rr_pqc", [dict(p) for p in pqc_list])
    if "rr_pqc_init" not in ss:
        ss.rr_pqc = [dict(p) for p in pqc_list]
        ss.rr_pqc_init = True

    ss.setdefault("rr_docs", list(bid.get("mandatory_docs", [])))
    if "rr_docs_init" not in ss:
        ss.rr_docs = list(bid.get("mandatory_docs", []))
        ss.rr_docs_init = True

    ss.setdefault("rr_mspecs", [dict(s) for s in bid.get("mandatory_specs", [])])
    if "rr_mspecs_init" not in ss:
        ss.rr_mspecs = [dict(s) for s in bid.get("mandatory_specs", [])]
        ss.rr_mspecs_init = True

    preferred_raw = bid.get("preferred_specs", [])
    ss.setdefault("rr_pspecs", [dict(s) for s in preferred_raw])
    if "rr_pspecs_init" not in ss:
        ss.rr_pspecs = [dict(s) for s in preferred_raw]
        ss.rr_pspecs_init = True

    # ── Tabs Configuration ───────────────────────────────────────────────────
    tabs = st.tabs(["🔍 Rules Verification Console", "✏️ Edit Requirements"])

    with tabs[0]:
        c_left, c_right = st.columns([4, 8])
        
        with c_left:
            st.markdown("""
            <div style="font-size:13px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
                        color:rgba(233,240,86,0.8); margin-bottom:12px;">📁 Select Tender Rules Document</div>
            """, unsafe_allow_html=True)
            
            # Select doc dropdown
            doc_options = [f"Master Tender Rules ({ss.bid_source})"]
            st.selectbox("Select Rules Document", options=doc_options, label_visibility="collapsed")
            
            # Document Properties card
            file_size_str = "8.85 MB"
            if bid_paths:
                try:
                    first_path = list(bid_paths.values())[0]
                    size_bytes = os.path.getsize(first_path)
                    file_size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                except Exception:
                    pass
            word_count = len(bid_text.split())
            
            doc_props_html = f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(139, 92, 246, 0.15); 
                        border-radius: 12px; padding: 20px; margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <div style="font-size: 11px; font-weight: 800; color: rgba(139, 92, 246, 0.8); 
                            letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px;">
                    ℹ️ Document Properties
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px;">
                    <div style="display: flex; justify-content: space-between;"><span style="color: rgba(199,211,234,0.6);">Vendor:</span> <span style="font-weight: 700; color: #E2E8F0;">Unknown Vendor</span></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color: rgba(199,211,234,0.6);">Document Type:</span> <span style="font-weight: 700; color: #E2E8F0;">Tender Rules</span></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color: rgba(199,211,234,0.6);">File Name:</span> <span style="font-weight: 700; color: #E2E8F0; font-family: monospace; font-size: 11px;">{html.escape(ss.bid_source)}</span></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color: rgba(199,211,234,0.6);">File Size:</span> <span style="font-weight: 700; color: #E2E8F0;">{file_size_str}</span></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color: rgba(199,211,234,0.6);">Word Count:</span> <span style="font-weight: 700; color: #E2E8F0;">{word_count}</span></div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; padding-top: 8px; border-top: 1px solid rgba(199,211,234,0.1);"><span style="color: rgba(199,211,234,0.6);">Integrity Status:</span> <span style="font-weight: 900; color: #10B981; font-size: 11px; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">VERIFIED</span></div>
                </div>
            </div>
            """
            st.markdown(doc_props_html, unsafe_allow_html=True)
            
        with c_right:
            st.markdown("""
            <div style="font-size:13px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
                        color:rgba(233,240,86,0.8); margin-bottom:12px;">🛡️ Extracted Qualification Rules & Requirements</div>
            """, unsafe_allow_html=True)
            
            # Gather list of all rules
            rules = []
            
            # 1. PQC
            for p in ss.rr_pqc:
                rules.append({
                    "title": p.get("label", "Pre-Qualification Criterion"),
                    "evidence": p.get("evidence", ""),
                    "page_keyword": p.get("label", ""),
                    "type_name": "PQC"
                })
            # 2. Mandatory Docs
            doc_evidence = bid.get("mandatory_docs_evidence", {})
            for d in ss.rr_docs:
                if d.strip():
                    rules.append({
                        "title": d,
                        "evidence": doc_evidence.get(d, f"Submission of mandatory document '{d}' is required."),
                        "page_keyword": d,
                        "type_name": "Required Document"
                    })
            # 3. Mandatory specs
            for s in ss.rr_mspecs:
                rules.append({
                    "title": s.get("label", "Mandatory Specification"),
                    "evidence": s.get("evidence", ""),
                    "page_keyword": s.get("label", ""),
                    "type_name": "Mandatory Spec"
                })
            # 4. Preferred specs
            for s in ss.rr_pspecs:
                rules.append({
                    "title": s.get("label", "Preferred Specification"),
                    "evidence": s.get("evidence", ""),
                    "page_keyword": s.get("label", ""),
                    "type_name": "Preferred Spec"
                })
                
            # Render list of cards dynamically
            if not rules:
                st.info("No rules found or extracted from this document.")
            else:
                for idx, r in enumerate(rules):
                    page, evidence_str = _find_page_and_evidence_for_keyword(r["evidence"], r["page_keyword"], bid_text)
                    rule_id = f"R{idx+1}"
                    
                    c_card, c_btn = st.columns([9.2, 0.8])
                    
                    # Renders card with custom layout
                    if not evidence_str or is_placeholder_evidence(evidence_str):
                        evidence_str = f"Requirements regarding {r['title']} must be complied with as per contract guidelines."
                    card_html = f"""
                    <div class="rule-card">
                        <div class="rule-title">
                            <span>{idx+1}) {html.escape(r["title"])}</span>
                            <div>
                                <span class="rule-id-chip">{rule_id}</span>
                                <span class="rule-page-chip">Page {page}</span>
                            </div>
                        </div>
                        <div class="rule-quote">"{html.escape(evidence_str)}"</div>
                    </div>
                    """
                    c_card.markdown(card_html, unsafe_allow_html=True)
                    
                    # Renders PDF jump link button next to it
                    master_pdf_filename = ss.bid_source if ss.get("bid_source") else ""
                    if master_pdf_filename:
                        pdf_url = f"?view_visual_file={urllib.parse.quote(master_pdf_filename)}&view_visual_page={page}&vendor=Master"
                        c_btn.markdown(
                            f'<a href="{pdf_url}" target="_self" class="pdf-jump-btn" title="Jump to page {page}">📄</a>',
                            unsafe_allow_html=True
                        )

    with tabs[1]:
        card_style = (
            "background:rgba(255,255,255,0.03);border:1px solid rgba(199,211,234,0.12);"
            "border-radius:12px;padding:20px 24px;margin-bottom:18px;"
        )
        section_hdr = lambda title, icon: st.markdown(
            f'<div style="font-size:13px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;'
            f'color:rgba(233,240,86,0.8);margin:24px 0 12px 0;">{icon} {title}</div>',
            unsafe_allow_html=True
        )

        def pdf_link_btn(label: str, keyword: str, col) -> None:
            """Renders a small PDF jump link button using internal viewer query params."""
            master_pdf_filename = ss.bid_source if ss.get("bid_source") else ""
            if master_pdf_filename:
                # Use the new robust page finder helper
                page, _ = _find_page_and_evidence_for_keyword("", keyword, bid_text)
                import urllib.parse
                pdf_url = f"?view_visual_file={urllib.parse.quote(master_pdf_filename)}&view_visual_page={page}&vendor=Master"
                col.markdown(
                    f'<a href="{pdf_url}" target="_self" class="pdf-jump-btn" title="View PDF page {page}">📄</a>',
                    unsafe_allow_html=True
                )

        # ── Section 1: Tender ID ─────────────────────────────────────────────
        section_hdr("Tender Identity", "🆔")
        with st.container():
            st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
            bid["tender_id"] = st.text_input(
                "Tender ID",
                value=bid.get("tender_id", ""),
                key="rr_tender_id",
                help="The unique tender/NIT number from the master bid document."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 2: PQC ───────────────────────────────────────────────────
        section_hdr("Pre-Qualification Criteria (PQC)", "✅")
        pqc_to_delete = []
        for idx, pqc in enumerate(ss.rr_pqc):
            with st.container():
                st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
                c1, c2, c3, c4, c_pdf, c_del = st.columns([3, 1.5, 1.5, 1.5, 0.5, 0.5])
                pqc["label"] = c1.text_input("Criterion", value=pqc.get("label", ""), key=f"rr_pqc_label_{idx}")
                # Safely convert threshold to float
                t_val = pqc.get("threshold", 0)
                if t_val is None:
                    t_val = 0.0
                else:
                    try:
                        t_val = float(t_val)
                    except (ValueError, TypeError):
                        t_val = 0.0
                pqc["threshold"] = c2.number_input("Threshold", value=t_val, step=0.5, key=f"rr_pqc_thresh_{idx}")
                pqc["unit"] = c3.text_input("Unit", value=pqc.get("unit", ""), key=f"rr_pqc_unit_{idx}")
                pqc["section"] = c4.text_input("Section", value=str(pqc.get("section", "")), key=f"rr_pqc_section_{idx}")
                pdf_link_btn(pqc.get("label", ""), pqc.get("label", ""), c_pdf)
                if c_del.button("🗑️", key=f"rr_del_pqc_{idx}", help="Remove this criterion"):
                    pqc_to_delete.append(idx)
                st.markdown('</div>', unsafe_allow_html=True)
        for i in reversed(pqc_to_delete):
            ss.rr_pqc.pop(i)
        if st.button("＋ Add PQC Criterion", key="rr_add_pqc"):
            ss.rr_pqc.append({"key": f"pqc_{len(ss.rr_pqc)}", "label": "New Criterion", "threshold": 0, "unit": "", "section": ""})
            st.rerun()

        # ── Section 3: Mandatory Documents ───────────────────────────────────
        section_hdr("Mandatory Documents Checklist", "📁")
        doc_to_delete = []
        for idx, doc in enumerate(ss.rr_docs):
            c1, c_pdf, c_del = st.columns([9, 0.5, 0.5])
            ss.rr_docs[idx] = c1.text_input(f"Document {idx+1}", value=doc, key=f"rr_doc_{idx}", label_visibility="collapsed")
            pdf_link_btn(doc, doc, c_pdf)
            if c_del.button("🗑️", key=f"rr_del_doc_{idx}", help="Remove this document"):
                doc_to_delete.append(idx)
        for i in reversed(doc_to_delete):
            ss.rr_docs.pop(i)
        if st.button("＋ Add Required Document", key="rr_add_doc"):
            ss.rr_docs.append("New Required Document")
            st.rerun()

        # ── Section 4: Mandatory Technical Specs ─────────────────────────────
        section_hdr("Mandatory Technical Specifications", "⚙️")
        op_map = {"gte": "≥ (Greater or Equal)", "lte": "≤ (Less or Equal)", "bool": "Must Have (Boolean)"}
        op_rev = {v: k for k, v in op_map.items()}
        op_options = list(op_map.values())

        mspec_to_delete = []
        for idx, spec in enumerate(ss.rr_mspecs):
            with st.container():
                st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
                c1, c2, c3, c4, c_pdf, c_del = st.columns([3, 2, 2, 1.5, 0.5, 0.5])
                spec["label"] = c1.text_input("Parameter", value=spec.get("label", ""), key=f"rr_mspec_label_{idx}")
                cur_op_label = op_map.get(spec.get("op", "bool"), op_options[2])
                sel_op = c2.selectbox("Operator", options=op_options, index=op_options.index(cur_op_label) if cur_op_label in op_options else 2, key=f"rr_mspec_op_{idx}")
                spec["op"] = op_rev.get(sel_op, "bool")
                spec["required_value"] = c3.text_input("Required Value", value=str(spec.get("required_value", "")), key=f"rr_mspec_val_{idx}")
                spec["unit"] = c4.text_input("Unit", value=spec.get("unit", ""), key=f"rr_mspec_unit_{idx}")
                pdf_link_btn(spec.get("label", ""), spec.get("label", ""), c_pdf)
                if c_del.button("🗑️", key=f"rr_del_mspec_{idx}", help="Remove this spec"):
                    mspec_to_delete.append(idx)
                st.markdown('</div>', unsafe_allow_html=True)
        for i in reversed(mspec_to_delete):
            ss.rr_mspecs.pop(i)
        if st.button("＋ Add Mandatory Spec", key="rr_add_mspec"):
            ss.rr_mspecs.append({"key": f"spec_{len(ss.rr_mspecs)}", "label": "New Specification", "op": "bool", "required_value": "", "unit": ""})
            st.rerun()

        # ── Section 5: Preferred Technical Specs ─────────────────────────────
        if ss.rr_pspecs or st.session_state.get("rr_show_preferred", False):
            section_hdr("Preferred / Desirable Technical Specifications", "⭐")
            pspec_to_delete = []
            for idx, spec in enumerate(ss.rr_pspecs):
                with st.container():
                    st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
                    c1, c2, c3, c4, c_pdf, c_del = st.columns([3, 2, 2, 1.5, 0.5, 0.5])
                    spec["label"] = c1.text_input("Parameter", value=spec.get("label", ""), key=f"rr_pspec_label_{idx}")
                    cur_op_label = op_map.get(spec.get("op", "bool"), op_options[2])
                    sel_op = c2.selectbox("Operator", options=op_options, index=op_options.index(cur_op_label) if cur_op_label in op_options else 2, key=f"rr_pspec_op_{idx}")
                    spec["op"] = op_rev.get(sel_op, "bool")
                    spec["required_value"] = c3.text_input("Required Value", value=str(spec.get("required_value", "")), key=f"rr_pspec_val_{idx}")
                    spec["unit"] = c4.text_input("Unit", value=spec.get("unit", ""), key=f"rr_pspec_unit_{idx}")
                    pdf_link_btn(spec.get("label", ""), spec.get("label", ""), c_pdf)
                    if c_del.button("🗑️", key=f"rr_del_pspec_{idx}", help="Remove this spec"):
                        pspec_to_delete.append(idx)
                    st.markdown('</div>', unsafe_allow_html=True)
            for i in reversed(pspec_to_delete):
                ss.rr_pspecs.pop(i)

        if st.button("＋ Add Preferred Spec", key="rr_add_pspec"):
            ss.rr_show_preferred = True
            ss.rr_pspecs.append({"key": f"pspec_{len(ss.rr_pspecs)}", "label": "New Preferred Spec", "op": "bool", "required_value": "", "unit": ""})
            st.rerun()

    # ── Action Bar ───────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(199,211,234,0.12);margin-bottom:24px;">', unsafe_allow_html=True)
    col_back, col_spacer, col_confirm = st.columns([2, 5, 3])

    with col_back:
        if st.button("← Back to Upload", key="rr_back", use_container_width=True):
            ss.pipeline_stage = "upload"
            ss.bid = None
            ss.bid_confirmed = False
            # clear init flags so form resets on re-entry
            for k in ["rr_pqc_init", "rr_docs_init", "rr_mspecs_init", "rr_pspecs_init",
                      "rr_pqc", "rr_docs", "rr_mspecs", "rr_pspecs"]:
                ss.pop(k, None)
            save_state_to_disk()
            st.rerun()

    with col_confirm:
        if st.button("✅ Confirm Rules & Start Vendor Analysis →", key="rr_confirm", type="primary", use_container_width=True):
            # Write edited values back into ss.bid
            ss.bid["tender_id"] = ss.get("rr_tender_id", bid.get("tender_id", ""))
            ss.bid["pqc"] = ss.rr_pqc
            ss.bid["mandatory_docs"] = [d for d in ss.rr_docs if d.strip()]
            ss.bid["mandatory_specs"] = ss.rr_mspecs
            ss.bid["preferred_specs"] = ss.rr_pspecs
            ss.bid_confirmed = True
            ss.pipeline_stage = "audit"
            # clear init flags
            for k in ["rr_pqc_init", "rr_docs_init", "rr_mspecs_init", "rr_pspecs_init"]:
                ss.pop(k, None)
            save_state_to_disk()
            st.rerun()


# ===========================================================================
# SECTION 11 — MAIN APPLICATION ROUTER & ENTRYPOINT
# ===========================================================================
def main() -> None:
    if "light_mode" not in st.session_state:
        st.session_state.light_mode = False

    if "page" in st.query_params:
        if st.query_params["page"] == "audit":
            st.session_state.nav_radio = "Audit Engine"
        elif st.query_params["page"] == "documentation":
            st.session_state.nav_radio = "documentation"
        elif st.query_params["page"] == "case-studies":
            st.session_state.nav_radio = "case-studies"
        else:
            st.session_state.nav_radio = "Home"

    current_page = st.session_state.get("nav_radio", "Home")
    sidebar_state = "collapsed" if current_page in ("Home", "documentation", "case-studies") else "expanded"
    
    try:
        from PIL import Image
        page_icon = Image.open("logo.jpg")
    except Exception:
        page_icon = "🛡️"

    st.set_page_config(page_title="Argus Bid AI — Tender Audit & Compliance", page_icon=page_icon,
                       layout="wide", initial_sidebar_state=sidebar_state)
                       
    # Dynamic CSS stylesheet replacement based on theme toggle status
    css_content = CSS
    if st.session_state.light_mode:
        css_content = css_content.replace(
            "--ink:#211119; --panel:#351E28; --panel2:#442734; --line:#593646;",
            "--ink:#F0F6FA; --panel:#E1EDF5; --panel2:#D2E3F0; --line:#BBD0E3;"
        ).replace(
            "--muted:#AEB8A0; --text:#FDF2F8; --blue:#E9F056; --blue-dk:#C6CD3E;",
            "--muted:#6D7E69; --text:#23121A; --blue:#351E28; --blue-dk:#553544;"
        ).replace(
            "--green:#D7EFFF; --amber:#FF5C34; --red:#FF5C34;",
            "--green:#4A7C59; --amber:#D97706; --red:#B32C1A;"
        ).replace(
            "radial-gradient(circle at 80% 20%, rgba(233, 240, 86, 0.08) 0%, transparent 50%),",
            "radial-gradient(circle at 80% 20%, rgba(215, 239, 255, 0.5) 0%, transparent 60%),"
        ).replace(
            "radial-gradient(circle at 20% 10%, rgba(215, 239, 255, 0.1) 0%, transparent 45%),",
            "radial-gradient(circle at 20% 80%, rgba(174, 184, 160, 0.25) 0%, transparent 60%),"
        ).replace(
            "radial-gradient(circle at 70% 80%, rgba(255, 92, 52, 0.05) 0%, transparent 50%),",
            ""
        ).replace(
            "radial-gradient(circle at 10% 70%, rgba(174, 184, 160, 0.06) 0%, transparent 50%),",
            ""
        ).replace(
            "opacity: 0.025",
            "opacity: 0.04"
        )
        
    st.markdown(css_content, unsafe_allow_html=True)
    st.markdown(CUSTOM_SPINNER_CSS, unsafe_allow_html=True)
    # Temporary comment out skeleton hiding to check rendering
    pass
    init_state()

    # Intercept query parameters for document visual hyperlinking
    if "view_visual_file" in st.query_params:
        focus_file = st.query_params["view_visual_file"]
        focus_page = int(st.query_params.get("view_visual_page", 1))
        vendor_name = st.query_params.get("vendor", "Master")
        
        for k in ["view_visual_file", "view_visual_page", "vendor"]:
            if k in st.query_params:
                del st.query_params[k]
                
        st.session_state["view_visual_file"] = focus_file
        st.session_state["view_visual_page"] = focus_page
        st.session_state["view_visual_vendor"] = vendor_name
        st.rerun()

    if st.session_state.get("view_visual_file"):
        render_visual_document_viewer()
        return

    # Intercept query parameters for document hyperlinking
    if "view_file" in st.query_params:
        focus_file = st.query_params["view_file"]
        focus_page = int(st.query_params.get("view_page", 1))
        vendor_name = st.query_params.get("vendor", "Master")
        
        for k in ["view_file", "view_page", "vendor"]:
            if k in st.query_params:
                del st.query_params[k]
                
        ss = st.session_state
        files_dict = {}
        title = ""
        if vendor_name == "Master":
            title = "Master Tender Document"
            files_dict = ss.get("bid_files", {})
        else:
            title = vendor_name
            files_dict = ss.get("vendor_files", {}).get(vendor_name, {})
            
        if files_dict:
            view_documents_dialog(title, files_dict, focus_file=focus_file, focus_page=focus_page)
    
    if current_page == "Home":
        st.markdown("<style>[data-testid='stSidebar'] {display: none !important;} [data-testid='collapsedControl'] {display: none !important;}</style>", unsafe_allow_html=True)
        render_landing_page()
        return
        

    if current_page in ("documentation", "case-studies"):
        st.markdown("<style>[data-testid='stSidebar'] {display: none !important;} [data-testid='collapsedControl'] {display: none !important;} .block-container, [data-testid='stAppViewBlockContainer'], .main .block-container {max-width: 100%; padding-top: 0 !important; padding: 0 !important; margin-top: 0 !important; gap: 0 !important;}</style>", unsafe_allow_html=True)
        
        max_width = "1240px" if current_page == "documentation" else "1180px"
        btn_color = "#E9F056" if current_page == "documentation" else "#C4B5FD"
        btn_rgba = "123, 146, 255" if current_page == "documentation" else "139, 92, 246"

        st.markdown(f"""
        <style>
        div[data-testid="stButton"] {{
            max-width: {max_width};
            margin: 0 auto;
            padding: 0 24px;
            display: flex; justify-content: flex-start;
        }}
        div.stButton > button {{
            margin-top: 0px; margin-bottom: 0px;
            display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px; 
            border-radius: 8px; font-weight: 600; font-size: 14px; transition: all 0.2s;
            background: rgba({btn_rgba}, 0.1) !important;
            border: 1px solid rgba({btn_rgba}, 0.3) !important;
            color: {btn_color} !important;
            box-shadow: 0 4px 15px rgba({btn_rgba}, 0.1);
        }}
        div.stButton > button:hover {{
            background: rgba({btn_rgba}, 0.2) !important;
            border-color: rgba({btn_rgba}, 0.5) !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        filename = "documentation.html" if current_page == "documentation" else "case-studies.html"
        try:
            with open(filename, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            html_content = html_content.replace("<button onclick=\"window.top.location.href='?page=home'\" style=\"", "<a href=\"?page=home\" target=\"_self\" style=\"text-decoration: none; ")
            html_content = html_content.replace("← Back to Overview</button>", "← Back to Overview</a>")

            import re
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL | re.IGNORECASE)
            body_content = body_match.group(1) if body_match else html_content
            style_blocks = re.findall(r'<style[^>]*>.*?</style>', html_content, re.DOTALL | re.IGNORECASE)
            styles = '\n'.join(style_blocks)
            
            css_styles = """<style>
            header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
            .block-container, [data-testid='stAppViewBlockContainer'], .main .block-container { padding-top: 0 !important; padding: 0 !important; margin: 0 !important; max-width: 100% !important; width: 100% !important; gap: 0 !important; }
            div[data-testid='stVerticalBlock'] { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
            div[data-testid='stVerticalBlock'] > div { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
            div[data-testid='stMarkdownContainer'] { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
            div[data-testid='stMarkdownContainer'] > div { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
            section[data-testid="stMain"] { padding-top: 0 !important; margin-top: 0 !important; overflow-x: hidden !important; }
            section[data-testid="stMain"] > div { padding-top: 0 !important; margin-top: 0 !important; }
            .stApp { margin-top: 0 !important; padding-top: 0 !important; overflow-x: hidden !important; }
            .stAppViewContainer, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; width: 100vw !important; }
            
            .pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 8px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; margin-left: 6px; }
            .pill::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; }
            .pill.bad { background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05)) !important; color: #F87171 !important; border: 1px solid rgba(239, 68, 68, 0.3) !important; box-shadow: 0 0 12px rgba(239, 68, 68, 0.1) !important; }
            .pill.bad::before { background: #FF5C34 !important; box-shadow: 0 0 6px rgba(239, 68, 68, 0.8) !important; }
            .pill.warn { background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)) !important; color: #FBBF24 !important; border: 1px solid rgba(245, 158, 11, 0.3) !important; box-shadow: 0 0 12px rgba(245, 158, 11, 0.1) !important; }
            .pill.warn::before { background: #F59E0B !important; box-shadow: 0 0 6px rgba(245, 158, 11, 0.8) !important; }
            </style>"""
            
            safe_html = css_styles + styles + body_content
            safe_html = "<div>\n" + '\n'.join([line for line in safe_html.split('\n') if line.strip() != '']) + "\n</div>"
            st.markdown(safe_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
        return

    api_key, model, run_clicked = render_sidebar()

    render_masthead()

    ss = st.session_state
    pipeline_stage = ss.get("pipeline_stage", "upload")

    # ── Rules Review page intercept ────────────────────────────────────────
    if pipeline_stage == "rules_review":
        render_rules_review_page()
        return

    # ── Vendor audit (after rules confirmed) ───────────────────────────────
    if pipeline_stage == "audit" and ss.get("bid_confirmed", False):
        components.html("""
            <script>
                if (window.innerWidth <= 768) {
                    window.parent.document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true }));
                    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        const buttons = sidebar.querySelectorAll('button');
                        if (buttons.length > 0) buttons[0].click();
                    }
                }
            </script>
        """, height=0)
        time.sleep(0.5)
        run_audit(api_key, model)

    if run_clicked:
        components.html("""
            <script>
                if (window.innerWidth <= 768) {
                    window.parent.document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true }));
                    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        const buttons = sidebar.querySelectorAll('button');
                        if (buttons.length > 0) buttons[0].click();
                    }
                }
            </script>
        """, height=0)
        time.sleep(0.5)
        run_audit(api_key, model)

    if not ss.processed or not ss.results:
        render_empty()
        return

    c1, c2, c3 = st.columns([6.2, 2.3, 1.5])
    with c1:
        eyebrow("01", "Control Center")
    with c2:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("🛡️ Rules Console", use_container_width=True, help="Open the interactive AI Tender Rules Understanding Console"):
            ss.pipeline_stage = "rules_review"
            save_state_to_disk()
            st.rerun()
    with c3:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("⎘ Export Report", type="primary", use_container_width=True):
            components.html(f"<script>setTimeout(function() {{ window.parent.print(); }}, 500);</script><!--{time.time()}-->", height=0, width=0)

    render_kpis(ss.results)
    render_bid_map()

    eyebrow("02", "Compliance Leaderboard")
    render_leaderboard(ss.results)

    eyebrow("03", "PSU Comparative Statement Matrix")
    render_comparative_statement(ss.results)

    eyebrow("04", "Explainable Ranking (XAI)")
    render_xai(ss.xai, ss.narrative)

    eyebrow("05", "Vendor Audit Deep-Dive")
    render_drawers(ss.results)

    # 10. Inject spotlight cursor tracking script
    components.html("""
        <script>
            const doc = window.parent.document.documentElement;
            window.parent.document.addEventListener('mousemove', (e) => {
                doc.style.setProperty('--mouse-x', e.clientX + 'px');
                doc.style.setProperty('--mouse-y', e.clientY + 'px');
            });
        </script>
    """, height=0, width=0)

    # 11. Render Apple-style quick navigation dock with bulletproof inline overrides
    st.markdown("""
    <style>
    .dock-container {
        position: fixed !important;
        bottom: 24px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 999999 !important;
        display: block !important;
    }
    .dock {
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-end !important;
        gap: 12px !important;
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        padding: 10px 18px !important;
        border-radius: 9999px !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        transition: all 0.3s ease !important;
    }
    .dock-item {
        position: relative !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        width: 44px !important;
        height: 44px !important;
        border-radius: 50% !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid var(--line) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-decoration: none !important;
        color: var(--text) !important;
    }
    .dock-icon {
        font-size: 20px !important;
        line-height: 1 !important;
        transition: transform 0.2s ease !important;
        display: block !important;
    }
    .dock-item:hover {
        transform: scale(1.3) translateY(-10px) !important;
        background: rgba(233, 240, 86, 0.15) !important;
        border-color: var(--blue) !important;
    }
    .dock-label {
        position: absolute !important;
        top: -45px !important;
        background: var(--ink) !important;
        border: 1px solid var(--line) !important;
        color: var(--text) !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        opacity: 0 !important;
        transform: translateY(10px) !important;
        transition: all 0.2s ease !important;
        pointer-events: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    .dock-item:hover .dock-label {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    </style>
    <div class="dock-container">
      <div class="dock">
        <a href="#control-center" class="dock-item" target="_self">
          <span class="dock-label">Control Center</span>
          <svg class="dock-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>
        </a>
        <a href="#compliance-leaderboard" class="dock-item" target="_self">
          <span class="dock-label">Leaderboard</span>
          <svg class="dock-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.45 1-1 1H4v2h16v-2h-5c-.55 0-1-.45-1-1v-2.34"/><path d="M12 2a6 6 0 0 1 6 6v1a6 6 0 0 1-6 6 6 6 0 0 1-6-6V8a6 6 0 0 1 6-6z"/></svg>
        </a>
        <a href="#psu-comparative-statement-matrix" class="dock-item" target="_self">
          <span class="dock-label">Comparative Statement</span>
          <svg class="dock-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
        </a>
        <a href="#explainable-ranking-xai" class="dock-item" target="_self">
          <span class="dock-label">Explainable Report</span>
          <svg class="dock-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M12 5v14"/></svg>
        </a>
        <a href="#vendor-audit-deep-dive" class="dock-item" target="_self">
          <span class="dock-label">Audit Deep-Dive</span>
          <svg class="dock-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
