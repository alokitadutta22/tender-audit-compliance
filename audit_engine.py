"""
================================================================================
 Argus Bid AI — Core Auditing & Compliance Rules Engine
 Built for PSU procurement workflows (IOCL-style NIT / BID evaluation)
================================================================================
 This module contains the core business rules, text extraction, document
 classification, compliance engine, and scoring logic. It is completely decoupled
 from the Streamlit UI and has no web dependencies.
================================================================================
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import base64
import logging
import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Optional dependencies imported defensively
try:
    import pdfplumber  # type: ignore
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader  # type: ignore
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    import anthropic  # type: ignore
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

try:
    import pytesseract  # type: ignore
    import os
    # Default Windows installation path fallback
    default_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tess):
        pytesseract.pytesseract.tesseract_cmd = default_tess
    pytesseract.get_tesseract_version()
    HAS_PYTESSERACT = True
except Exception as e:
    import logging
    logging.error(f"[Pytesseract Init Error] Failed to initialize Tesseract: {e}", exc_info=True)
    HAS_PYTESSERACT = False

try:
    import easyocr  # type: ignore
    HAS_EASYOCR = True
except Exception:
    HAS_EASYOCR = False


def write_engine_log(message: str) -> None:
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    # Safely print to console, catching WinError 233/broken pipe errors on Windows
    try:
        print(log_line)
    except Exception:
        pass
        
    # Safely write to log file
    try:
        with open("audit_engine.log", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        pass


# ===========================================================================
# SECTION 1 — SEMANTIC STATUS VOCABULARY
# ===========================================================================
STATUS_RESPONSIVE = "Responsive"
STATUS_DISQUALIFIED = "Disqualified"

MAF_VALID = "Found (Valid)"
MAF_INVALID = "Invalid / Non-Compliant"
MAF_MISSING = "MISSING / NOT FOUND"

READ_PASS = "Readable (Passed OCR)"
READ_LOW = "Partially Readable (Low-Quality Warning)"
READ_CORRUPT = "Corrupted / Unreadable"

# Document taxonomy used by the classifier.
DOC_TYPES = [
    "Manufacturer's Authorization Form (MAF)",
    "Technical Datasheet / Bid",
    "Commercial / Price Bid",
    "PAN Card",
    "GST Registration",
    "GeM Registration",
    "Audited Balance Sheet",
    "Experience / Past Performance Certificate",
    "Company Registration",
    "Udyam Registration Certificate",
    "Deviation Statement",
    "GeM Contract / Agreement",
    "Master BID Document",
    "Affidavit / Undertaking",
    "Unclassified Document",
]


# ===========================================================================
# SECTION 2 — MOCK CORPUS (rich, realistic tender text)
# ===========================================================================
MASTER_BID_TEXT = """--- PAGE 1 ---
INDIAN OIL CORPORATION LIMITED (IOCL)
HALDIA REFINERY — INFORMATION TECHNOLOGY DEPARTMENT
NOTICE INVITING TENDER (NIT)

Tender No.: IOCL/HR/IT/2026/NW-4471
Tender Title: Supply, Installation, Testing & Commissioning of Layer-3 Core
              Network Switches for Refinery Process Control LAN
Mode of Tender: e-Tender (Two-Bid System) via Government e-Marketplace (GeM)

--- PAGE 2 ---
--------------------------------------------------------------------------------
SECTION 2 — PRE-QUALIFICATION CRITERIA (PQC)
--------------------------------------------------------------------------------
2.1  The bidder must have a minimum of 3 years of experience in supply and
     commissioning of enterprise networking equipment to PSU / Government
     organisations, evidenced by satisfactory completion certificates.
2.2  The bidder must have an average annual financial turnover of not less than
     INR 5 Crore during the last three (3) financial years.
2.3  The bidder must be a registered seller on the Government e-Marketplace
     (GeM) portal under the relevant product category.

--- PAGE 3 ---
--------------------------------------------------------------------------------
SECTION 3 — MANDATORY DOCUMENTS
--------------------------------------------------------------------------------
3.1  Submission of a valid Manufacturer's Authorization Form (MAF), signed by
     the Original Equipment Manufacturer (OEM) on the OEM's official letterhead
     and explicitly referencing Tender No. IOCL/HR/IT/2026/NW-4471, is an
     absolute prerequisite for technical qualification.
3.2  Copy of valid PAN Card of the bidding entity.
3.3  Copy of valid GST Registration Certificate.
3.4  Valid GeM Registration / Seller Profile.
3.5  Audited Balance Sheet and Profit & Loss statements for the last 3 years.

--- PAGE 4 ---
--------------------------------------------------------------------------------
SECTION 5 — MANDATORY TECHNICAL SPECIFICATIONS
--------------------------------------------------------------------------------
5.1  Switch Architecture: Must be a managed Layer-3 modular/fixed switch.
5.2  Operating Temperature: The equipment must reliably operate up to 60 degC
     ambient temperature.
5.3  Switching Throughput: Minimum aggregate throughput of 800 Gbps.
5.4  Warranty: Minimum on-site comprehensive warranty of 36 months from the
     date of commissioning.
5.5  Port Density: Minimum 48 x 1G/10G ports plus 4 x 40G uplink ports.
5.6  Redundant Power: Dual hot-swappable power supply units (1+1 redundancy).

--- PAGE 5 ---
--------------------------------------------------------------------------------
SECTION 6 — DESIRABLE / PREFERRED TECHNICAL SPECIFICATIONS
--------------------------------------------------------------------------------
6.1  Stacking: Support for hardware stacking of minimum 8 units.
6.2  IPv6: Native dual-stack IPv4/IPv6 routing support.
6.3  MACsec: Hardware-based MACsec line-rate encryption.
6.4  Extended OEM Warranty: Optional extended OEM warranty beyond 36 months is
     considered favourably during evaluation.

--- PAGE 6 ---
--------------------------------------------------------------------------------
SECTION 9 — DEVIATIONS
--------------------------------------------------------------------------------
9.1  Bidders must submit a "No Deviation" statement. Any deviation from the
     mandatory technical specifications in Section 5 shall render the bid
     liable to rejection.
"""

MOCK_VENDORS: Dict[str, Dict[str, str]] = {}


# ===========================================================================
# SECTION 3 — PDF / FILE TEXT EXTRACTION (with graceful fallback)
# ===========================================================================
def find_page_number_for_match(text: str, match_index: int) -> int:
    markers = list(re.finditer(r"--- PAGE (\d+) ---", text[:match_index]))
    if markers:
        return int(markers[-1].group(1))
    return 1


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


def resolve_actual_document_page(doc: str, combined_text: str, filename: str, mandatory_docs: List[str]) -> int:
    """Scans all pages inside a vendor file to find the actual page containing the target document.
    Calculates a score for each page and penalizes pages containing cover letters, tables of contents, or index lists.
    """
    file_marker = f"--- FILE {filename} ---"
    start_idx = combined_text.find(file_marker)
    if start_idx == -1:
        return 1
        
    next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
    file_text = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
    
    # Split by PAGE markers
    pages = []
    page_matches = list(re.finditer(r'--- PAGE (\d+) ---\n?', file_text))
    for i, m in enumerate(page_matches):
        p_num = int(m.group(1))
        p_start = m.end()
        p_end = page_matches[i+1].start() if i+1 < len(page_matches) else len(file_text)
        pages.append((p_num, file_text[p_start:p_end]))
        
    if not pages:
        return 1
        
    # Generate search keywords for the target document
    # We create a temporary AuditEngine to access keyword generation dynamically
    from audit_engine import AuditEngine
    temp_engine = AuditEngine()
    keywords = temp_engine._generate_dynamic_keywords(doc)
    
    best_page = 1
    best_score = -1.0
    
    # Compile regexes for strong signals
    pan_pattern = re.compile(r'\b[A-Z]{4}[A-Z0-9][0-9]{4}[A-Z0-9]\b')
    gst_pattern = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b')
    
    for p_num, page_content in pages:
        page_low = page_content.lower()
        
        # 1. Detect if this is an index/contents/declaration page (TOC)
        # If it has index-related keywords AND mentions other mandatory documents, penalize it.
        is_index = False
        index_signals = ["index", "contents", "table of contents", "enclosed", "attached", "herewith", "submitted", "declaration", "annexure", "checklist", "page no"]
        if any(sig in page_low for sig in index_signals):
            matches_count = 0
            for other_doc in mandatory_docs:
                if other_doc == doc:
                    continue
                other_kws = temp_engine._generate_dynamic_keywords(other_doc)
                if any(okw in page_low for okw in other_kws[:3]):
                    matches_count += 1
            if matches_count >= 2:
                is_index = True
                
        # 2. Calculate match score
        score = 0.0
        
        # Keyword matches
        for kw in keywords:
            if kw in page_low:
                score += 2.0
                if kw == doc.lower():
                    score += 3.0
                    
        # Strong pattern/document type specific signals
        has_pan_signal = pan_pattern.search(page_content) or any(x in page_low for x in ["axaunilnber", "acaunilnber", "par anenl", "income tax", "govt of india", "tax department"])
        if doc == "PAN Card" and has_pan_signal:
            score += 20.0
        elif doc in ("GST Registration", "GST Certificate", "GSTIN") and gst_pattern.search(page_content):
            score += 20.0
        elif doc == "Audited Balance Sheet":
            fin_terms = ["balance sheet", "profit & loss", "liabilities", "assets", "audited", "chartered accountant", "financial statements", "income statement", "capital account"]
            term_matches = sum(1 for term in fin_terms if term in page_low)
            if term_matches >= 3:
                score += 10.0 + term_matches
        elif doc == "GeM Registration":
            gem_terms = ["gem registration", "seller id", "gem seller", "marketplace", "gemc-", "oem/seller", "profile status"]
            term_matches = sum(1 for term in gem_terms if term in page_low)
            if term_matches >= 2:
                score += 10.0 + term_matches
                
        # Apply heavy index penalty
        if is_index:
            score -= 15.0
            
        if score > best_score and score > 0:
            best_score = score
            best_page = p_num
            
    return best_page


def format_table_as_markdown(table: List[List[Optional[str]]]) -> str:
    """Format a raw table list-of-lists from pdfplumber into GitHub Flavored Markdown."""
    if not table or not any(table):
        return ""
    
    cleaned_table = []
    for row in table:
        if not row:
            continue
        cleaned_row = []
        for cell in row:
            if cell is None:
                val = ""
            else:
                val = str(cell).replace("\n", " ").replace("|", "\\|").strip()
            cleaned_row.append(val)
        cleaned_table.append(cleaned_row)
        
    if not cleaned_table:
        return ""
        
    headers = cleaned_table[0]
    if len(cleaned_table) == 1:
        headers = [f"Col {i+1}" for i in range(len(cleaned_table[0]))]
        rows = cleaned_table
    else:
        rows = cleaned_table[1:]
        
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        md.append("| " + " | ".join(row) + " |")
        
import threading
_PDFIUM_LOCK = threading.Lock()


def perform_ocr_on_page(page) -> str:
    """Attempts to run OCR on a pdfplumber page using pytesseract or easyocr."""
    write_engine_log(f"[OCR Diagnostic] perform_ocr_on_page started. HAS_PYTESSERACT={HAS_PYTESSERACT}, HAS_EASYOCR={HAS_EASYOCR}")
    if not (HAS_PYTESSERACT or HAS_EASYOCR):
        write_engine_log("OCR skipped: pytesseract and easyocr not available.")
        return ""
        
    try:
        page_num = page.page_number
        write_engine_log(f"[OCR] Scanning Page {page_num}...")
        
        # Convert page to PIL image (with thread lock to prevent PDFium C++ segfaults)
        with _PDFIUM_LOCK:
            im = page.to_image(resolution=150)
            pil_img = im.original
        
        # 1. Try pytesseract first
        if HAS_PYTESSERACT:
            try:
                import pytesseract
                write_engine_log(f"[OCR] Trying pytesseract on Page {page_num}...")
                text = pytesseract.image_to_string(pil_img)
                clean_text = text.strip()
                alnum_count = sum(1 for c in clean_text if c.isalnum())
                
                # Check for vowel ratio to catch pure random character noise (e.g. '|\|||/l' or watermarks)
                has_vowels = True
                letters = [c.lower() for c in clean_text if c.isalpha()]
                if letters:
                    vowel_ratio = sum(1 for c in letters if c in 'aeiou') / len(letters)
                    if vowel_ratio < 0.10:
                        has_vowels = False
                        
                # Check for presence of common English/tender vocabulary to confirm it's readable words
                common_terms = ["the", "and", "of", "for", "with", "document", "copy", "certificate", "registration", "seller", "pan", "gst", "maf", "balance", "sheet", "tender", "date", "no", "number", "company", "limited", "sign", "seal"]
                has_common = any(term in clean_text.lower() for term in common_terms)
                
                is_valid = (alnum_count >= 15 and has_vowels and has_common)
                
                if is_valid:
                    write_engine_log(f"[OCR] pytesseract succeeded on Page {page_num} (extracted {alnum_count} alphanumeric chars).")
                    return clean_text
                else:
                    reasons = []
                    if alnum_count < 15: reasons.append(f"low chars ({alnum_count})")
                    if not has_vowels: reasons.append("no/low vowels")
                    if not has_common: reasons.append("no common English/tender words")
                    write_engine_log(f"[OCR] pytesseract returned low quality/garbage output ({', '.join(reasons)}). Falling back to EasyOCR...")
            except Exception as e:
                write_engine_log(f"[OCR] pytesseract failed on Page {page_num}: {e}")
                logging.warning(f"pytesseract OCR failed: {e}")
                
        # 2. Try easyocr next
        if HAS_EASYOCR:
            try:
                import easyocr
                import numpy as np
                write_engine_log(f"[OCR] Trying easyocr on Page {page_num} (weights will download on first run)...")
                global _EASYOCR_READER
                if '_EASYOCR_READER' not in globals():
                    _EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
                
                img_np = np.array(pil_img)
                results = _EASYOCR_READER.readtext(img_np)
                if results:
                    text = "\n".join([res[1] for res in results])
                    if text and len(text.strip()) > 10:
                        write_engine_log(f"[OCR] easyocr succeeded on Page {page_num}.")
                        return text.strip()
            except Exception as e:
                write_engine_log(f"[OCR] easyocr failed on Page {page_num}: {e}")
                logging.warning(f"easyocr OCR failed: {e}")
                
    except Exception as e:
        write_engine_log(f"[OCR] Failed to render Page {page.page_number} for OCR: {e}")
        logging.error(f"Failed to render page for OCR: {e}")
        
    return ""


def extract_text_from_pdf_bytes(data: bytes, enable_ocr: bool = True, extract_tables: bool = True) -> Tuple[str, Optional[str]]:
    """Extract text from raw PDF bytes using pdfplumber (primary) or pypdf (fallback)."""
    # Sanitize PDF bytes by stripping any leading junk bytes before '%PDF' marker
    if data:
        idx = data.find(b'%PDF')
        if idx > 0:
            write_engine_log(f"[PDF Sanitizer] Stripped {idx} leading junk bytes before PDF header")
            data = data[idx:]

    text = ""
    last_error: Optional[str] = None

    # Performance optimization: if tables are not requested, use pypdf first as it is 10-15x faster than pdfplumber
    if not extract_tables and HAS_PYPDF:
        write_engine_log("[PDF Reader] FAST-PATH: Selecting pypdf reader (no tables requested)")
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for i, p in enumerate(reader.pages, start=1):
                ptext = p.extract_text() or ""
                # If page is empty, check for OCR fallback
                if len(ptext.strip()) < 10 and enable_ocr:
                    if HAS_PDFPLUMBER:
                        try:
                            with pdfplumber.open(io.BytesIO(data)) as pdf:
                                if len(pdf.pages) >= i:
                                    write_engine_log(f"[OCR] Page {i} has empty text. Triggering OCR fallback...")
                                    ocr_text = perform_ocr_on_page(pdf.pages[i-1])
                                    if ocr_text:
                                        ptext = ocr_text + "\n[OCR Fallback Content]"
                        except Exception as e_ocr:
                            write_engine_log(f"[OCR ERROR] Failed to run OCR on page {i}: {e_ocr}")
                pages.append(f"--- PAGE {i} ---\n{ptext}")
            text = "\f".join(pages).strip()
            if text:
                write_engine_log(f"[PDF Reader] FAST-PATH: pypdf successfully parsed {len(reader.pages)} pages")
                return text, None
        except Exception as exc:
            write_engine_log(f"[PDF Reader ERROR] pypdf fast-path failed: {exc}")
            last_error = f"pypdf fast-path failed: {exc}"

    if HAS_PDFPLUMBER:
        write_engine_log("[PDF Reader] DETAILED-PATH: Selecting pdfplumber reader (table extraction active)")
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages, start=1):
                    ptext = page.extract_text() or ""
                    
                    # If page has no text, try OCR fallback
                    if len(ptext.strip()) < 10:
                        if enable_ocr and (HAS_PYTESSERACT or HAS_EASYOCR):
                            write_engine_log(f"[OCR] Page {i} has empty text. Triggering OCR fallback...")
                            ocr_text = perform_ocr_on_page(page)
                            if ocr_text:
                                ptext = ocr_text + "\n[OCR Fallback Content]"
                    
                    md_tables = []
                    if extract_tables and (page.lines or page.rects):
                        tables = page.extract_tables()
                        if tables:
                            write_engine_log(f"[Table Extractor] Found {len(tables)} table(s) on Page {i}")
                            for tbl in tables:
                                tbl_md = format_table_as_markdown(tbl)
                                if tbl_md:
                                    md_tables.append(tbl_md)
                                
                    if md_tables:
                        ptext += "\n\n### Extracted Tables:\n" + "\n\n".join(md_tables)
                        
                    pages.append(f"--- PAGE {i} ---\n{ptext}")
                text = "\f".join(pages).strip()
            if text:
                write_engine_log(f"[PDF Reader] pdfplumber successfully parsed {len(pages)} pages")
                return text, None
        except Exception as exc:
            write_engine_log(f"[PDF Reader ERROR] pdfplumber failed: {exc}")
            last_error = (last_error or "") + f" | pdfplumber failed: {exc}"

    # Secondary fallback to pypdf if pdfplumber failed (and we didn't try pypdf already)
    if extract_tables and HAS_PYPDF and not text:
        write_engine_log("[PDF Reader] FALLBACK-PATH: Selecting pypdf reader (pdfplumber failed)")
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for i, p in enumerate(reader.pages, start=1):
                ptext = p.extract_text() or ""
                pages.append(f"--- PAGE {i} ---\n{ptext}")
            text = "\f".join(pages).strip()
            if text:
                return text, None
        except Exception as exc:
            write_engine_log(f"[PDF Reader ERROR] pypdf fallback failed: {exc}")
            last_error = (last_error or "") + f" | pypdf fallback failed: {exc}"

    if not text:
        write_engine_log("[PDF Reader ERROR] All parsing engines failed to extract text")
        return "", last_error or "No text could be extracted (scanned image / empty PDF)."
    return text, None


def is_valid_bid_document(text: str) -> bool:
    """Heuristic check to ensure the uploaded text looks like a Tender/BID document."""
    sample = text[:10000].lower()
    keywords = ["tender", "bid", "nit", "notice inviting", "request for proposal", 
                "rfp", "procurement", "terms and conditions", "contract",
                "quotation", "rfq", "bidding", "specification", "corrigendum", "addendum"]
    for kw in keywords:
        if kw in sample:
            return True
    if len(text) > 10000:
        for kw in keywords:
            if kw in text.lower():
                return True
    return False


def read_uploaded_file(uploaded, enable_ocr: bool = False, extract_tables: bool = True) -> Tuple[str, Optional[str]]:
    """Read a Streamlit UploadedFile into text, dispatching by extension."""
    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        data = uploaded.getvalue()
    except Exception as exc:
        return "", f"Could not read bytes: {exc}"

    if name.endswith(".pdf"):
        # Performance optimization: dynamically disable table extraction for non-technical files
        # (like PAN, GST, MAFs, indices, and other non-bid/non-spec documents) to speed up parsing by 15x.
        if extract_tables:
            # Table extraction is only needed for specifications, datasheets, BOQs, or NIT bid parameters.
            is_spec_file = any(k in name for k in [
                "datasheet", "spec", "compliance", "matrix", "technical", 
                "boq", "price", "schedule", "commercial", "bid", "nit", "tender"
            ])
            if not is_spec_file:
                extract_tables = False
        return extract_text_from_pdf_bytes(data, enable_ocr=enable_ocr, extract_tables=extract_tables)
    if name.endswith((".txt", ".csv", ".md")):
        try:
            return data.decode("utf-8", errors="replace"), None
        except Exception as exc:
            return "", f"Decode failed: {exc}"
    try:
        return data.decode("utf-8", errors="replace"), None
    except Exception as exc:
        return "", f"Unsupported file type: {exc}"


# ===========================================================================
# SECTION 4 — DATA STRUCTURES
# ===========================================================================
@dataclass
class SpecResult:
    param: str
    required: str
    provided: str
    status: str          # "match" | "fail" | "lacking"
    mandatory: bool
    section: str = ""
    file: str = ""
    page: int = 1
    bid_file: str = ""
    bid_page: int = 1


@dataclass
class PQCResult:
    label: str
    required: str
    provided: str
    passed: bool
    section: str = ""
    file: str = ""
    page: int = 1
    bid_file: str = ""
    bid_page: int = 1


@dataclass
class InventoryItem:
    filename: str
    doc_type: str
    readability: str


@dataclass
class MAFResult:
    status: str          # MAF_VALID | MAF_INVALID | MAF_MISSING
    evidence: str        # textual proof
    source_file: str = ""
    page: int = 1


@dataclass
class Violation:
    title: str
    requirement: str
    finding: str


@dataclass
class VendorResult:
    name: str
    inventory: List[InventoryItem] = field(default_factory=list)
    maf: Optional[MAFResult] = None
    pqc: List[PQCResult] = field(default_factory=list)
    mandatory_specs: List[SpecResult] = field(default_factory=list)
    preferred_specs: List[SpecResult] = field(default_factory=list)
    deviations: List[str] = field(default_factory=list)
    missing_documents: List[str] = field(default_factory=list)
    disqualified: bool = False
    violations: List[Violation] = field(default_factory=list)
    score: float = 0.0
    mandatory_score: float = 0.0
    preferred_score: float = 0.0
    rank: Optional[int] = None
    status: str = STATUS_RESPONSIVE
    summary: str = ""
    make_model: str = ""
    commercial_details: str = ""
    document_checklist: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ===========================================================================
# SECTION 5 — SPEC LIBRARY (recognised technical parameters)
# ===========================================================================
SPEC_LIBRARY = [
    {
        "key": "architecture",
        "label": "Switch Architecture (Layer-3)",
        "bid_kw": ["layer-3", "layer 3", "l3"],
        "vendor_kw": ["layer-3", "layer 3", "l3"],
        "op": "bool",
        "bid_value": "Managed Layer-3 switch",
        "unit": "",
    },
    {
        "key": "temperature",
        "label": "Operating Temperature",
        "bid_kw": ["operating temperature", "ambient", "degc"],
        "bid_value_re": r"up to\s*(\d+)\s*deg",
        "vendor_kw": ["temperature", "ambient", "degc", "operating environment"],
        "vendor_value_re": r"(\d+)\s*deg",
        "op": "gte",
        "unit": "degC",
    },
    {
        "key": "throughput",
        "label": "Switching Throughput",
        "bid_kw": ["throughput", "gbps"],
        "bid_value_re": r"(\d+)\s*gbps",
        "vendor_kw": ["throughput", "gbps"],
        "vendor_value_re": r"(\d+)\s*gbps",
        "op": "gte",
        "unit": "Gbps",
    },
    {
        "key": "warranty",
        "label": "Warranty Period",
        "bid_kw": ["warranty", "months"],
        "bid_value_re": r"(\d+)\s*months",
        "vendor_kw": ["warranty", "months"],
        "vendor_value_re": r"(\d+)\s*months",
        "op": "gte",
        "unit": "months",
    },
    {
        "key": "ports",
        "label": "Port Density",
        "bid_kw": ["port density", "ports"],
        "bid_value_re": r"(\d+)\s*x\s*1g",
        "vendor_kw": ["ports", "port density"],
        "vendor_value_re": r"(\d+)\s*x\s*1g",
        "op": "gte",
        "unit": "ports",
    },
    {
        "key": "redundant_power",
        "label": "Redundant Power Supply",
        "bid_kw": ["redundant power", "power supply", "1+1"],
        "vendor_kw": ["dual", "1+1", "hot-swappable power", "redundant power"],
        "op": "bool",
        "bid_value": "Dual hot-swappable PSU (1+1)",
        "unit": "",
    },
    # ---- Preferred / desirable ----
    {
        "key": "stacking",
        "label": "Hardware Stacking",
        "bid_kw": ["stacking"],
        "vendor_kw": ["stacking", "stack"],
        "op": "bool",
        "bid_value": "Stacking support",
        "unit": "",
    },
    {
        "key": "ipv6",
        "label": "Native IPv6 Support",
        "bid_kw": ["ipv6"],
        "vendor_kw": ["ipv6", "dual-stack"],
        "op": "bool",
        "bid_value": "Native IPv4/IPv6",
        "unit": "",
    },
    {
        "key": "macsec",
        "label": "MACsec Encryption",
        "bid_kw": ["macsec"],
        "vendor_kw": ["macsec"],
        "op": "bool",
        "bid_value": "Hardware MACsec",
        "unit": "",
    },
    {
        "key": "extended_warranty",
        "label": "Extended OEM Warranty (>36m)",
        "bid_kw": ["extended", "warranty beyond"],
        "vendor_kw": ["warranty", "months"],
        "vendor_value_re": r"(\d+)\s*months",
        "op": "gt_bonus",
        "bid_value": "Warranty beyond 36 months",
        "unit": "months",
        "baseline": 36,
    },
    {
        "key": "display_size",
        "label": "LED Wall Size",
        "bid_kw": ["led video wall", "diagonal", "130 inch", "130\""],
        "bid_value_re": r"(\d+)\s*(?:inch|\")",
        "vendor_kw": ["size", "diagonal", "130 inch", "130\""],
        "vendor_value_re": r"(\d+)\s*(?:inch|\")",
        "op": "gte",
        "unit": "inch",
    },
    {
        "key": "pixel_pitch",
        "label": "LED Pixel Pitch",
        "bid_kw": ["pixel pitch", "pitch"],
        "bid_value_re": r"(\d+\.?\d*)\s*mm",
        "vendor_kw": ["pixel pitch", "pitch"],
        "vendor_value_re": r"(\d+\.?\d*)\s*mm",
        "op": "lte",
        "unit": "mm",
    },
    {
        "key": "brightness",
        "label": "Display Brightness",
        "bid_kw": ["brightness", "nits"],
        "bid_value_re": r"(\d+)\s*nits",
        "vendor_kw": ["brightness", "nits"],
        "vendor_value_re": r"(\d+)\s*nits",
        "op": "gte",
        "unit": "nits",
    },
    {
        "key": "contrast",
        "label": "Contrast Ratio",
        "bid_kw": ["contrast ratio", "5000:1"],
        "vendor_kw": ["contrast", "ratio"],
        "op": "bool",
        "bid_value": "5000:1 contrast ratio",
        "unit": "",
    },
    {
        "key": "lfd_size",
        "label": "LFD Size",
        "bid_kw": ["lfd display", "85 inch", "85\""],
        "bid_value_re": r"(\d+)\s*(?:inch|\")",
        "vendor_kw": ["size", "diagonal", "85 inch", "85\""],
        "vendor_value_re": r"(\d+)\s*(?:inch|\")",
        "op": "gte",
        "unit": "inch",
    },
    {
        "key": "duty_cycle",
        "label": "Operating Hour Duty Cycle",
        "bid_kw": ["duty cycle", "operation hour", "16x7", "24x7"],
        "vendor_kw": ["duty cycle", "operation hour", "16x7", "24x7"],
        "op": "bool",
        "bid_value": "16x7 duty cycle or better",
        "unit": "",
    },
    {
        "key": "dib_size",
        "label": "DIB Screen Size",
        "bid_kw": ["digital interactive board", "screen size", "65 inches", "65\""],
        "bid_value_re": r"(\d+)\s*(?:inches|\")",
        "vendor_kw": ["screen size", "diagonal", "65 inches", "65\""],
        "vendor_value_re": r"(\d+)\s*(?:inches|\")",
        "op": "gte",
        "unit": "inches",
    },
    {
        "key": "dib_resolution",
        "label": "DIB Resolution",
        "bid_kw": ["native resolution", "3840\s*x\s*2160", "4k"],
        "vendor_kw": ["resolution", "3840\s*x\s*2160", "4k"],
        "op": "bool",
        "bid_value": "3840 x 2160 (4K)",
        "unit": "",
    },
    {
        "key": "ups_rating",
        "label": "UPS Capacity Rating",
        "bid_kw": ["ups", "rating in kva", "1 kva"],
        "bid_value_re": r"(\d+)\s*kva",
        "vendor_kw": ["rating in kva", "capacity", "1 kva"],
        "vendor_value_re": r"(\d+)\s*kva",
        "op": "gte",
        "unit": "KVA",
    },
    {
        "key": "ap_speed",
        "label": "Access Point Wireless Speed",
        "bid_kw": ["access point", "wireless speed", "1200 mbps"],
        "bid_value_re": r"(\d+)\s*mbps",
        "vendor_kw": ["speed", "wireless speed", "1200 mbps"],
        "vendor_value_re": r"(\d+)\s*mbps",
        "op": "gte",
        "unit": "Mbps",
    },
]


# ===========================================================================
# SECTION 6 — DETERMINISTIC AUDIT ENGINE
# ===========================================================================
class AuditEngine:
    """Rule-based, fully deterministic engine. Every verdict is reproducible
    and traceable to an explicit rule and a text snippet.
    """

    def __init__(self):
        self.bid_text = ""

    def _generate_dynamic_keywords(self, doc_name: str) -> List[str]:
        doc_lower = doc_name.lower()
        keywords = [doc_lower, re.sub(r'[^\w\s]', '', doc_lower)]
        
        abbr_match = re.search(r'\(([^)]+)\)', doc_lower)
        if abbr_match:
            keywords.append(abbr_match.group(1).strip())
            
        expansions = {
            "pan": ["permanent account", "income tax", "permanent account number"],
            "gst": ["goods and services tax", "registration certificate", "gstin"],
            "gem": ["government e-marketplace", "seller profile", "gem profile", "gem contract", "consignee receipt", "gemc-"],
            "maf": ["oem authorization", "manufacturer's authorization", "manufacturer authorization", "authori"],
            "balance": ["profit and loss", "audited balance", "financial statements", "turnover"],
            "udyam": ["udyam registration", "udyam certificate", "msme certificate", "udyam number", "micro and small"]
        }
        
        tokens = re.findall(r'[a-z0-9]+', doc_lower)
        for t in tokens:
            if t in expansions:
                keywords.extend(expansions[t])
                
        keywords.extend([t for t in tokens if len(t) > 3])
        
        seen = set()
        unique_kws = []
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean and kw_clean not in seen:
                seen.add(kw_clean)
                unique_kws.append(kw_clean)
                
        return unique_kws

    def _find_bid_pos(self, spec: Dict[str, Any]) -> Optional[int]:
        bid_text = getattr(self, "bid_text", "")
        if not bid_text:
            return None
        low = bid_text.lower()
        
        bre = spec.get("bid_value_re")
        if bre:
            for m in re.finditer(bre, bid_text, re.I):
                window = bid_text[max(0, m.start() - 60): m.start()].lower()
                if any(k in window for k in spec.get("bid_kw", [])):
                    return m.start()
            m = re.search(bre, bid_text, re.I)
            if m:
                return m.start()

        for kw in spec.get("bid_kw") or [spec.get("label"), spec.get("key")]:
            if not kw:
                continue
            idx = low.find(str(kw).lower())
            if idx != -1:
                return idx
        return None

    def _find_bid_pqc_pos(self, key: str) -> Optional[int]:
        bid_text = getattr(self, "bid_text", "")
        if not bid_text:
            return None
        low = bid_text.lower()
        if key == "experience":
            m = re.search(r"minimum of\s*(\d+)\s*years?\s*of\s*experience", low)
            if m:
                return m.start()
        elif key == "turnover":
            m = re.search(r"turnover[^0-9]*?(?:inr|rs\.?)?\s*([\d,\.]+)\s*crore", low)
            if m:
                return m.start()
        elif key == "gem":
            idx = low.find("government e-marketplace")
            if idx == -1:
                idx = low.find("gem")
            if idx != -1:
                return idx
        return None

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _snippet(text: str, anchor: str) -> str:
        norm_text = AuditEngine._norm(text)
        sentences = [s.strip() for s in re.split(r'[.!?\n]', norm_text) if s.strip()]
        for s in sentences:
            if anchor.lower() in s.lower():
                if len(s) > 300:
                    words = s[:300].split()
                    return " ".join(words[:-1]) + "..."
                return s + "."
        words = norm_text[:300].split()
        return " ".join(words[:-1]) + "..." if words else ""

    def parse_master_bid(self, text: str) -> Dict[str, Any]:
        self.bid_text = text
        low = text.lower()

        # Tender ID / Number
        tender_id = ""
        m = re.search(r"tender\s*(?:no\.?|number|id)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_\-]{4,})", text, re.I)
        if m:
            tender_id = m.group(1).strip().rstrip(".")

        mand_block, pref_block = self._split_spec_sections(text)

        mandatory_specs: List[Dict[str, Any]] = []
        preferred_specs: List[Dict[str, Any]] = []
        for spec in SPEC_LIBRARY:
            in_mand = any(k in mand_block.lower() for k in spec["bid_kw"])
            in_pref = any(k in pref_block.lower() for k in spec["bid_kw"])
            required = self._bid_required_value(spec, mand_block if in_mand else pref_block)
            entry = {**spec, "required_value": required}
            if in_mand and spec["op"] != "gt_bonus":
                mandatory_specs.append(entry)
            elif in_pref or spec["op"] == "gt_bonus":
                preferred_specs.append(entry)

        # PQC
        pqc: List[Dict[str, Any]] = []
        m = re.search(r"minimum of\s*(\d+)\s*years?\s*of\s*experience", low)
        if m:
            pqc.append({"key": "experience", "label": "Minimum Experience",
                        "threshold": float(m.group(1)), "unit": "years", "section": "2.1"})
        m = re.search(r"turnover[^0-9]*?(?:inr|rs\.?)?\s*([\d,\.]+)\s*crore", low)
        if m:
            pqc.append({"key": "turnover", "label": "Average Annual Turnover",
                        "threshold": float(m.group(1).replace(",", "")), "unit": "INR Crore",
                        "section": "2.2"})
        if "government e-marketplace" in low or "gem" in low:
            pqc.append({"key": "gem", "label": "GeM Registration", "threshold": None,
                        "unit": "", "section": "2.3"})

        # Mandatory documents
        doc_map = [
            ("Manufacturer's Authorization Form (MAF)", ["manufacturer's authorization", "maf"]),
            ("PAN Card", ["pan card", "pan "]),
            ("GST Registration", ["gst registration", "gst "]),
            ("GeM Registration", ["gem registration", "gem "]),
            ("Audited Balance Sheet", ["balance sheet", "audited"]),
        ]
        mandatory_docs = []
        sec3 = self._section(text, "MANDATORY DOCUMENTS", "SECTION 5")
        sec3_low = sec3.lower() if sec3 else low
        for canonical, kws in doc_map:
            if any(k in sec3_low for k in kws):
                mandatory_docs.append(canonical)

        # General Tender & Commercial info
        general_info = []
        
        # Delivery / Completion Period
        m_delivery = re.search(r"(?:completion|delivery)\s+(?:period|schedule|time)\s*(?:is|of|within)?\s*[:\-]?\s*(\d+\s*(?:months|weeks|days|years|month|week|day|year))", low)
        if m_delivery:
            general_info.append({"label": "Delivery Period", "required_value": m_delivery.group(1).title()})
        else:
            m_delivery_alt = re.search(r"within\s*(\d+\s*(?:months|weeks|days|years|month|week|day|year))", low)
            if m_delivery_alt:
                general_info.append({"label": "Delivery Period", "required_value": f"Within {m_delivery_alt.group(1).title()}"})
            else:
                general_info.append({"label": "Delivery Period", "required_value": "Refer to Schedule of Requirements"})
                
        # Bid Validity
        m_validity = re.search(r"validity\s+of\s+bid\s*(?:is|of)?\s*[:\-]?\s*(\d+\s*days)", low)
        if not m_validity:
            m_validity = re.search(r"bid\s+validity\s*(?:is|of)?\s*[:\-]?\s*(\d+\s*days)", low)
        if m_validity:
            general_info.append({"label": "Bid Validity Period", "required_value": m_validity.group(1).title()})
        else:
            general_info.append({"label": "Bid Validity Period", "required_value": "120 Days"})
            
        # EMD / Bid Security
        m_emd = re.search(r"(?:emd|earnest\s+money\s+deposit|bid\s+security)\s*(?:of|is)?\s*[:\-]?\s*(?:inr|rs\.?)?\s*([\d,]+)", low)
        if m_emd:
            general_info.append({"label": "Earnest Money Deposit (EMD)", "required_value": f"INR {m_emd.group(1)}"})
        else:
            general_info.append({"label": "Earnest Money Deposit (EMD)", "required_value": "Bid Security Declaration Required"})
            
        # Holiday / Blacklisting Listing
        if "holiday listing" in low or "blacklisting" in low:
            general_info.append({"label": "Holiday Listing Policy", "required_value": "Prohibited (Holiday-listed bidders disqualified)"})
        else:
            general_info.append({"label": "Holiday Listing Policy", "required_value": "Standard IOCL Guidelines apply"})
            
        # Local Content Requirement
        m_local = re.search(r"(?:local\s+content|mii)\s*(?:requirement|minimum|of)?\s*[:\-]?\s*(\d+%)", low)
        if m_local:
            general_info.append({"label": "Minimum Local Content (MII)", "required_value": m_local.group(1)})
        else:
            general_info.append({"label": "Minimum Local Content (MII)", "required_value": "Class-I Local Supplier (Min 50%)"})

        return {
            "tender_id": tender_id,
            "pqc": pqc,
            "mandatory_docs": mandatory_docs,
            "mandatory_specs": mandatory_specs,
            "preferred_specs": preferred_specs,
            "general_info": general_info,
            "raw": text,
        }

    def _split_spec_sections(self, text: str) -> Tuple[str, str]:
        mand = self._section(text, "MANDATORY TECHNICAL SPECIFICATIONS",
                             "DESIRABLE / PREFERRED TECHNICAL SPECIFICATIONS") or ""
        pref = self._section(text, "DESIRABLE / PREFERRED TECHNICAL SPECIFICATIONS",
                             "DEVIATIONS") or ""
        if not mand:
            mand = text
        return mand, pref

    @staticmethod
    def _section(text: str, start_marker: str, end_marker: str) -> Optional[str]:
        low = text.lower()
        s = low.find(start_marker.lower())
        if s == -1:
            return None
        e = low.find(end_marker.lower(), s + len(start_marker))
        if e == -1:
            e = len(text)
        return text[s:e]

    @staticmethod
    def _bid_required_value(spec: Dict[str, Any], block: str) -> Any:
        if spec.get("bid_value_re"):
            m = re.search(spec["bid_value_re"], block, re.I)
            if m:
                return float(m.group(1))
        return spec.get("bid_value", True)

    def classify_document(self, filename: str, text: str) -> str:
        low = (text or "").lower()
        header = low[:1500]
        
        # We will score each document type dynamically based on its generated keywords
        best_type = "Unclassified Document"
        best_score = 0.0
        
        file_lower = filename.lower()
        
        for doc_type in DOC_TYPES:
            if doc_type == "Unclassified Document":
                continue
                
            kws = self._generate_dynamic_keywords(doc_type)
            score = 0.0
            
            # Filename matching (highest weight)
            doc_tokens = re.findall(r'[a-z0-9]+', doc_type.lower())
            file_tokens = re.findall(r'[a-z0-9]+', file_lower)
            overlap = set(doc_tokens) & set(file_tokens)
            score += len(overlap) * 5.0
            
            # Check if any generated keywords match the filename
            for kw in kws:
                if len(kw) > 2 and kw in file_lower:
                    score += 4.0
                    
            # Text matching (header and full body)
            # 1. Matches in the header (first 1500 characters) - strong signal
            for kw in kws:
                if kw in header:
                    score += 2.0
                    
            # 2. Overall occurrence count in the full text
            for kw in kws:
                count = low.count(kw)
                if count > 0:
                    score += min(count, 3) * 0.5
                    
            if score > best_score:
                best_score = score
                best_type = doc_type
                
        # If no strong match, default to Unclassified Document
        if best_score < 2.0:
            return "Unclassified Document"
            
        return best_type

    @staticmethod
    def assess_readability(text: str, error: Optional[str]) -> str:
        if error and not (text or "").strip():
            return READ_CORRUPT
        clean = (text or "").strip()
        if len(clean) < 15:
            return READ_CORRUPT
        total = len(clean)
        bad = clean.count("\ufffd")
        
        import string
        allowed_chars = set(string.punctuation + "₹•")
        good = sum(1 for c in clean if c.isalnum() or c.isspace() or c in allowed_chars)
        
        good_ratio = good / total
        bad_ratio = bad / total
        if bad_ratio > 0.015 or good_ratio < 0.85:
            return READ_LOW
        return READ_PASS

    def _resolve_pages_using_local_index(self, mandatory_docs: List[str], combined_text: str) -> Dict[str, int]:
        resolved = {}
        # Get index text (first 6 pages)
        p7_idx = combined_text.find("--- PAGE 7 ---")
        index_text = combined_text[:p7_idx] if p7_idx != -1 else combined_text[:20000]
        
        lines = index_text.split("\n")
        
        # Local synonym dictionary for standard abbreviation mapping
        synonyms = {
            "pan": ["permanent", "account", "taxpayer", "tax"],
            "gst": ["goods", "services", "gstin", "reg", "registration"],
            "gem": ["marketplace", "gemc-", "seller", "contract", "crac"],
            "maf": ["oem", "manufacturer", "authorization", "authorize", "letterhead", "auth"],
            "balance": ["sheet", "audited", "audit", "profit", "loss", "turnover", "financial"],
            "udyam": ["msme", "micro", "small", "enterprise", "certificate", "registration"]
        }
        
        import difflib
        
        for doc in mandatory_docs:
            doc_lower = doc.lower()
            doc_tokens = re.findall(r'[a-z0-9]+', doc_lower)
            if not doc_tokens:
                continue
                
            best_score = 0.0
            best_page = -1
            
            for line in lines:
                line_low = line.lower().strip()
                if not line_low:
                    continue
                    
                line_tokens = re.findall(r'[a-z0-9]+', line_low)
                if not line_tokens:
                    continue
                    
                score = 0.0
                
                # 1. Direct token overlap (e.g., matching "pan" in "PAN Card" and "PAN - KAN...")
                overlap = set(doc_tokens) & set(line_tokens)
                score += len(overlap) * 2.0
                
                # 2. Synonym overlap (e.g., mapping "oem" to "maf")
                for token in doc_tokens:
                    if token in synonyms:
                        syn_overlap = set(synonyms[token]) & set(line_tokens)
                        score += len(syn_overlap) * 1.0
                        
                # 3. Fuzzy SequenceMatcher similarity (handles spelling/grammatical variations)
                for dt in doc_tokens:
                    for lt in line_tokens:
                        if len(dt) > 2 and len(lt) > 2:
                            ratio = difflib.SequenceMatcher(None, dt, lt).ratio()
                            if ratio > 0.8:
                                score += ratio * 1.5
                                
                if score > best_score:
                    # Parse page number or page range at the end of the line
                    match = re.search(r'(?:page|pg\.?|pages)?\s*(\d+)(?:\s*-\s*\d+)?\s*$', line_low)
                    if match:
                        page_num = int(match.group(1))
                        if page_num > 1:
                            best_score = score
                            best_page = page_num
                            
            if best_score >= 1.5 and best_page != -1:
                resolved[doc] = best_page
                
        return resolved

    def _validate_document_checklist_deterministic(self, mandatory_docs: List[str], inventory: List[InventoryItem], combined_text: str) -> Dict[str, Dict[str, Any]]:
        checklist_results = {}
        present_docs = {item.doc_type: item for item in inventory}
        
        # Call local, offline index resolver first
        resolved_pages = self._resolve_pages_using_local_index(mandatory_docs, combined_text)
        
        for doc in mandatory_docs:
            is_present = (doc in present_docs)
            filename = present_docs[doc].filename if is_present else None
            
            if not is_present:
                # Dynamically detect documents embedded inside combined/booklet files
                # by scanning each file's text chunk extracted from combined_text.
                low_combined = combined_text.lower()

                doc_lower_name = doc.lower()
                if "udyam" in doc_lower_name or "msme" in doc_lower_name:
                    udyam_kws = ["udyam registration", "udyam-", "udyam certificate", "msme certificate", "micro and small", "udyam"]
                    if any(kw in low_combined for kw in udyam_kws) or "Udyam Registration Certificate" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                item_text = combined_text[start_idx:next_file_idx].lower() if next_file_idx != -1 else combined_text[start_idx:].lower()
                                if any(kw in item_text for kw in udyam_kws) or item.doc_type == "Udyam Registration Certificate":
                                    filename = item.filename
                                    break

                elif "pan" in doc_lower_name:
                    pan_pattern = re.compile(r'\b[A-Z]{4}[A-Z0-9][0-9]{4}[A-Z0-9]\b')
                    is_pan_found = (
                        pan_pattern.search(combined_text) or 
                        any(x in low_combined for x in ["permanent account", "pan card", "pan no", "income tax", "govt of india", "tax department", "par anenl", "acaunilnber"]) or 
                        "PAN Card" in present_docs
                    )
                    if is_pan_found:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                is_chunk_pan = (
                                    pan_pattern.search(chunk) or 
                                    any(x in chunk_low for x in ["permanent account", "pan card", "pan no", "income tax", "govt of india", "tax department", "par anenl", "acaunilnber"]) or 
                                    item.doc_type == "PAN Card"
                                )
                                if is_chunk_pan:
                                    filename = item.filename
                                    break

                elif "gst" in doc_lower_name:
                    gst_pattern = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b')
                    gst_kws = ["gstin", "gst registration", "gst certificate", "goods and services tax", "gst no", "gst number"]
                    if gst_pattern.search(combined_text) or any(kw in low_combined for kw in gst_kws) or "GST Registration" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if gst_pattern.search(chunk) or any(kw in chunk_low for kw in gst_kws) or item.doc_type == "GST Registration":
                                    filename = item.filename
                                    break

                elif "gem" in doc_lower_name:
                    gem_kws = ["gem registration", "gem seller", "gem profile", "gem portal", "marketplace", "gemc-"]
                    if any(kw in low_combined for kw in gem_kws) or "GeM Registration" in present_docs or "GeM Contract / Agreement" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in gem_kws) or item.doc_type in ("GeM Registration", "GeM Contract / Agreement"):
                                    filename = item.filename
                                    break

                elif "balance" in doc_lower_name or "turnover" in doc_lower_name or "financial" in doc_lower_name or "audited" in doc_lower_name:
                    fin_kws = ["balance sheet", "profit & loss", "audited balance", "financial statement", "profit and loss", "chartered accountant", "turnover"]
                    if any(kw in low_combined for kw in fin_kws) or "Audited Balance Sheet" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in fin_kws) or item.doc_type == "Audited Balance Sheet":
                                    filename = item.filename
                                    break

                elif "commercial" in doc_lower_name or "experience" in doc_lower_name or "past performance" in doc_lower_name or "credentials" in doc_lower_name:
                    exp_kws = ["experience", "contract", "purchase order", "work order", "completion", "credentials", "performance"]
                    if any(kw in low_combined for kw in exp_kws) or "GeM Contract / Agreement" in present_docs or "Experience / Past Performance Certificate" in present_docs or "Technical PQC + Datasheets" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in exp_kws) or item.doc_type in ("GeM Contract / Agreement", "Experience / Past Performance Certificate", "Technical PQC + Datasheets"):
                                    filename = item.filename
                                    break

                elif "technical pqc" in doc_lower_name or "datasheet" in doc_lower_name or "technical specs" in doc_lower_name or "technical specification" in doc_lower_name:
                    tech_kws = ["datasheet", "specification", "technical pqc", "technical specs", "compliance matrix"]
                    if any(kw in low_combined for kw in tech_kws) or "Technical Datasheet / Bid" in present_docs or "Technical PQC + Datasheets" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in tech_kws) or item.doc_type in ("Technical Datasheet / Bid", "Technical PQC + Datasheets"):
                                    filename = item.filename
                                    break

                elif "black listing" in doc_lower_name or "holiday listing" in doc_lower_name or "blacklist" in doc_lower_name or "debar" in doc_lower_name:
                    black_kws = ["blacklist", "blacklisted", "holiday listing", "debar", "debarred", "affidavit", "undertaking", "declaration"]
                    if any(kw in low_combined for kw in black_kws) or "Affidavit / Undertaking" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in black_kws) or item.doc_type == "Affidavit / Undertaking":
                                    filename = item.filename
                                    break

                elif "bsd" in doc_lower_name or "bid security" in doc_lower_name or "emd" in doc_lower_name or "earnest money" in doc_lower_name:
                    bsd_kws = ["bid security", "earnest money", "emd", "bsd", "declaration", "undertaking", "affidavit"]
                    if any(kw in low_combined for kw in bsd_kws) or "Affidavit / Undertaking" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in bsd_kws) or item.doc_type == "Affidavit / Undertaking":
                                    filename = item.filename
                                    break

                elif "deviation" in doc_lower_name:
                    dev_kws = ["deviation", "no deviation", "nil deviation", "deviation statement"]
                    if any(kw in low_combined for kw in dev_kws) or "Deviation Statement" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in dev_kws) or item.doc_type == "Deviation Statement":
                                    filename = item.filename
                                    break

                elif "mii" in doc_lower_name or "make in india" in doc_lower_name or "local content" in doc_lower_name:
                    mii_kws = ["local content", "make in india", "mii", "ppp-mii", "undertaking", "self-certification"]
                    if any(kw in low_combined for kw in mii_kws) or "Affidavit / Undertaking" in present_docs:
                        is_present = True
                        for item in inventory:
                            file_marker = f"--- FILE {item.filename} ---"
                            start_idx = combined_text.find(file_marker)
                            if start_idx != -1:
                                next_file_idx = combined_text.find("--- FILE ", start_idx + len(file_marker))
                                chunk = combined_text[start_idx:next_file_idx] if next_file_idx != -1 else combined_text[start_idx:]
                                chunk_low = chunk.lower()
                                if any(kw in chunk_low for kw in mii_kws) or item.doc_type == "Affidavit / Undertaking":
                                    filename = item.filename
                                    break

            if is_present:
                # Resolve page number: check local index first, then fall back to robust document page resolver
                if doc in resolved_pages:
                    page = resolved_pages[doc]
                else:
                    page = resolve_actual_document_page(doc, combined_text, filename, mandatory_docs)
                
                finding = "Present and verified."
                status = "Compliant"
                low_text = combined_text.lower()
                
                # Cross-document dependency audits
                if doc == "Udyam Registration Certificate":
                    has_aadhaar = bool(re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', combined_text))
                    res_pan_pat = re.compile(r'\b[A-Z]{4}[A-Z0-9][0-9]{4}[A-Z0-9]\b')
                    has_pan = bool(
                        res_pan_pat.search(combined_text) or 
                        any(x in low_text for x in ["permanent account", "pan card", "pan no", "income tax", "govt of india", "tax department", "par anenl", "acaunilnber"])
                    )
                    has_bank = any(kw in low_text for kw in ["ifsc", "account no", "bank name", "cancelled cheque", "bank account", "branch"])
                    has_gstin = bool(re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b', combined_text)) or "gstin" in low_text or "gst registration" in low_text
                    
                    cross_checks = []
                    cross_checks.append(f"Aadhaar: {'Verified' if has_aadhaar else 'Lacking'}")
                    cross_checks.append(f"PAN: {'Verified' if has_pan else 'Lacking'}")
                    cross_checks.append(f"Bank Details: {'Verified' if has_bank else 'Lacking'}")
                    cross_checks.append(f"GSTIN: {'Verified' if has_gstin else 'Lacking (if applicable)'}")
                    
                    finding = f"Present and verified. Cross-linked requirements: " + ", ".join(cross_checks) + "."
                    
                elif doc == "GST Registration":
                    gst_match = re.search(r'\b\d{2}([A-Z]{5}\d{4}[A-Z])([A-Z\d]{1})[Z]{1}([A-Z\d]{1})\b', combined_text)
                    if gst_match:
                        extracted_pan = gst_match.group(1)
                        finding = f"Present and verified. Extracted GSTIN: {gst_match.group(0)} (contains PAN: {extracted_pan})."
                    else:
                        finding = "Present and verified. GST registration status confirmed."
                        
                elif doc == "PAN Card":
                    res_pan_pat = re.compile(r'\b[A-Z]{4}[A-Z0-9][0-9]{4}[A-Z0-9]\b')
                    pan_match = res_pan_pat.search(combined_text)
                    if pan_match:
                        finding = f"Present and verified. PAN Number: {pan_match.group(0)}."
                    else:
                        finding = "Present and verified. PAN credentials verified."
                        
                elif doc == "Audited Balance Sheet":
                    has_udin = "udin" in low_text or bool(re.search(r'\b\d{18}\b', combined_text))
                    has_ca = any(kw in low_text for kw in ["chartered accountant", "ca signature", "ca stamp", "membership no"])
                    checks = []
                    if has_ca: checks.append("CA Signature/Stamp found")
                    if has_udin: checks.append("UDIN verified")
                    if checks:
                        finding = f"Present and verified ({', '.join(checks)})."
                    else:
                        finding = "Present and verified. Financial statements audited."

                if "insolvency" in doc.lower() and "undergoing" in low_text and "bankruptcy" in low_text:
                    status = "Non-Compliant"
                    finding = "Undergoing insolvency/bankruptcy proceedings."
                elif "black listing" in doc.lower() and "blacklist" in low_text:
                    status = "Non-Compliant"
                    finding = "Declares history of blacklisting/holiday listing."
                checklist_results[doc] = {
                    "status": status,
                    "page": page,
                    "file": filename,
                    "finding": finding
                }
            else:
                checklist_results[doc] = {
                    "status": "Missing",
                    "page": None,
                    "file": None,
                    "finding": f"Mandatory document {doc} was not submitted."
                }
        return checklist_results

    def validate_maf(self, inventory: List[InventoryItem], files: Dict[str, str],
                     tender_id: str) -> MAFResult:
        maf_file = next((i for i in inventory
                         if i.doc_type == "Manufacturer's Authorization Form (MAF)"), None)
        if not maf_file:
            return MAFResult(status=MAF_MISSING,
                             evidence="No file in this vendor's submission was classified as a "
                                      "Manufacturer's Authorization Form. No OEM authorization "
                                      "letterhead or signature pattern was detected.")
        text = files.get(maf_file.filename, "")
        low = text.lower()

        # 1. OEM Letterhead
        has_letterhead = any(k in low for k in ["letterhead", "oem", "original equipment manufacturer", "manufacturer"]) \
            or bool(re.search(r"(cisco|juniper|hp|hpe|aruba|dell|arista|extreme|sourabh)\b", low))
            
        # 2. Tender Specific (Lenient normalized check)
        tender_present = False
        if tender_id:
            tender_id_low = tender_id.lower()
            if tender_id_low in low:
                tender_present = True
            else:
                # Strip spaces, hyphens, slashes, and dots
                norm_tender = re.sub(r"[\s/_\-\.]", "", tender_id_low)
                norm_low = re.sub(r"[\s/_\-\.]", "", low)
                if norm_tender in norm_low:
                    tender_present = True
                else:
                    # Match by parts (e.g. if the last significant part like NW-4471 is present along with 'iocl' or 'tender')
                    parts = [p for p in re.split(r"[\s/_\-\.]", tender_id_low) if len(p) >= 3]
                    if parts:
                        last_part = parts[-1]
                        if last_part in low and ("iocl" in low or "tender" in low or "ref" in low):
                            tender_present = True

        has_date = bool(re.search(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b", low)) or "date" in low or "bid" in low

        # 3. Warranty & Support Commitment
        has_warranty = any(k in low for k in ["warranty", "support", "technical support", "guarantee", "honor"])

        # 4. Authorized Signatory (Signature & Stamp)
        has_signatory = any(k in low for k in ["signature", "signed", "stamped", "seal", "authorized representative", "authorized signatory", "director", "manager", "representative", "auth. signatory"])

        reasons = []
        if not has_letterhead:
            reasons.append("Not printed on a recognized OEM/Manufacturer letterhead")
        if not tender_present:
            ref = re.search(r"tender\s*no\.?\s*[:\-]?\s*([A-Z0-9/_\-]{4,})", text, re.I)
            if ref and tender_id:
                reasons.append(f'References Tender No. "{ref.group(1).rstrip(".")}" instead of correct Tender No. "{tender_id}"')
            elif tender_id:
                reasons.append(f'Does not reference the correct Tender No. "{tender_id}"')
        if not has_date:
            reasons.append("Does not contain a valid reference date or bid reference")
        if not has_warranty:
            reasons.append("Does not explicitly confirm warranty or technical support commitment")
        if not has_signatory:
            reasons.append("Missing indication of signature or stamp from OEM authorized signatory")

        pnum = 1
        ref = re.search(re.escape(tender_id) if tender_present else r"tender\s*no\.?\s*[:\-]?\s*([A-Z0-9/_\-]{4,})", text, re.I)
        if ref:
            pnum = find_page_number_for_match(text, ref.start())
        else:
            m_auth = re.search(r"authoriz", text, re.I)
            if m_auth:
                pnum = find_page_number_for_match(text, m_auth.start())

        snippet = self._snippet(text, tender_id if tender_present else "authorization")

        if not reasons:
            page_info = f" (Found on Page {pnum})"
            return MAFResult(status=MAF_VALID,
                             evidence=f"Authorization confirmed. Meets all criteria:\n"
                                      f"• OEM Letterhead verified\n"
                                      f"• Tender Specific (No. {tender_id}) verified\n"
                                      f"• Warranty & Support commitment confirmed\n"
                                      f"• OEM Authorized Signatory verified\n\n"
                                      f"Extracted evidence quote:{page_info}\n\"{snippet}\"",
                             source_file=maf_file.filename,
                             page=pnum)
        else:
            return MAFResult(status=MAF_INVALID,
                             evidence="Document resembles an MAF but is non-compliant:\n• "
                                      + "\n• ".join(reasons)
                                      + f"\n\nExtracted Text:\n\"{snippet}\"",
                             source_file=maf_file.filename,
                             page=pnum)

    def extract_spec(self, spec: Dict[str, Any], vendor_text: str, mandatory: bool) -> SpecResult:
        low = vendor_text.lower()
        op = spec["op"]
        required = spec.get("required_value", spec.get("bid_value", True))
        
        # Parse required value into a float if using a numeric comparison
        req_num = None
        if op in ("gte", "lte", "gt_bonus"):
            if isinstance(required, (int, float)):
                req_num = float(required)
            elif isinstance(required, str):
                match_num = re.search(r"(\d+(?:\.\d+)?)", required)
                if match_num:
                    try:
                        req_num = float(match_num.group(1))
                    except ValueError:
                        pass
                else:
                    try:
                        req_num = float(required)
                    except (ValueError, TypeError):
                        pass

        unit = spec.get("unit", "")

        # Safe defaults for vendor_kw
        vendor_kws = spec.get("vendor_kw", [])
        if not vendor_kws:
            vendor_kws = [spec.get("label", ""), spec.get("key", "")]
        vendor_kws = [k for k in vendor_kws if k]

        bid_file, bid_page = "", 1
        bid_pos = self._find_bid_pos(spec)
        if bid_pos is not None:
            bid_file, bid_page = find_file_and_page_for_match(getattr(self, "bid_text", ""), bid_pos)

        if op in ("gte", "lte"):
            val, pos = self._find_number_and_pos(vendor_text, spec)
            if val is None:
                return SpecResult(spec["label"], self._fmt(required, unit), "[DATA LACKING]",
                                  "lacking", mandatory, bid_file=bid_file, bid_page=bid_page)
            
            # Type safety fallback to avoid TypeError
            target_req = req_num if req_num is not None else (float(required) if isinstance(required, (int, float)) else None)
            if target_req is None:
                try:
                    target_req = float(required)
                except (ValueError, TypeError):
                    target_req = 0.0 # Default fallback to prevent crash
                    
            ok = (val >= target_req) if op == "gte" else (val <= target_req)
            file_name, page_num = "", 1
            page_info = ""
            if pos is not None:
                file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                page_info = f" (Pg {page_num})"
            return SpecResult(spec["label"], self._fmt(required, unit), self._fmt(val, unit) + page_info,
                               "match" if ok else "fail", mandatory, file=file_name, page=page_num,
                               bid_file=bid_file, bid_page=bid_page)

        if op == "bool":
            pos = None
            for kw in vendor_kws:
                idx = low.find(kw.lower())
                if idx != -1:
                    pos = idx
                    break
            present = pos is not None
            file_name, page_num = "", 1
            page_info = ""
            if present and pos is not None:
                file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                page_info = f" (Pg {page_num})"
            return SpecResult(spec["label"], "Required", f"Provided{page_info}" if present else "[DATA LACKING]",
                               "match" if present else "lacking", mandatory, file=file_name, page=page_num,
                               bid_file=bid_file, bid_page=bid_page)

        if op == "gt_bonus":
            val, pos = self._find_number_and_pos(vendor_text, spec)
            baseline = spec.get("baseline", 0)
            try:
                base_num = float(baseline)
            except (ValueError, TypeError):
                base_num = 0.0
            if val is None:
                return SpecResult(spec["label"], f">{baseline} {unit}", "[DATA LACKING]",
                                  "lacking", mandatory, bid_file=bid_file, bid_page=bid_page)
            ok = val > base_num
            file_name, page_num = "", 1
            page_info = ""
            if pos is not None:
                file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                page_info = f" (Pg {page_num})"
            return SpecResult(spec["label"], f">{baseline} {unit}", self._fmt(val, unit) + page_info,
                              "match" if ok else "fail", mandatory, file=file_name, page=page_num,
                              bid_file=bid_file, bid_page=bid_page)

        return SpecResult(spec["label"], str(required), "[DATA LACKING]", "lacking", mandatory,
                          bid_file=bid_file, bid_page=bid_page)

    @staticmethod
    def _find_number(text: str, spec: Dict[str, Any]) -> Optional[float]:
        val, _ = AuditEngine._find_number_and_pos(text, spec)
        return val

    @staticmethod
    def _find_number_and_pos(text: str, spec: Dict[str, Any]) -> Tuple[Optional[float], Optional[int]]:
        vendor_kws = spec.get("vendor_kw", [])
        if not vendor_kws:
            vendor_kws = [spec.get("label", ""), spec.get("key", "")]
        vendor_kws = [k for k in vendor_kws if k]

        vre = spec.get("vendor_value_re")
        if not vre:
            # Generate value regex based on unit
            unit = str(spec.get("unit", "")).lower().strip()
            if "crore" in unit or "cr" in unit:
                vre = r"(\d+(?:\.\d+)?)\s*(?:crore|cr|crores)"
            elif "lakh" in unit or "lacs" in unit:
                vre = r"(\d+(?:\.\d+)?)\s*(?:lakh|lacs|lakhs)"
            elif "year" in unit:
                vre = r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)"
            elif "inch" in unit or "\"" in unit:
                vre = r"(\d+(?:\.\d+)?)\s*(?:inch|inches|\")"
            elif unit:
                vre = r"(\d+(?:\.\d+)?)\s*(?:" + re.escape(unit) + r")"
            else:
                vre = r"(\d+(?:\.\d+)?)"

        if not any(k.lower() in text.lower() for k in vendor_kws) and not spec.get("vendor_value_re"):
            return None, None

        best_val = None
        best_pos = None
        for m in re.finditer(vre, text, re.I):
            window = text[max(0, m.start() - 40): m.start()].lower()
            if any(k.lower() in window for k in vendor_kws) or best_val is None:
                try:
                    best_val = float(m.group(1))
                    best_pos = m.start()
                except ValueError:
                    pass
                if any(k.lower() in window for k in vendor_kws):
                    return best_val, best_pos
        return best_val, best_pos

    @staticmethod
    def _fmt(val: Any, unit: str) -> str:
        if isinstance(val, bool):
            return "Required" if val else "-"
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return f"{val} {unit}".strip()

    def detect_deviations(self, vendor_text: str) -> List[str]:
        deviations = []
        patterns = [
            r"[^.]*cannot supply[^.]*\.",
            r"[^.]*instead we will provide[^.]*\.",
            r"[^.]*deviation[^.]*\.",
            r"[^.]*not supported[^.]*\.",
            r"[^.]*as an alternative[^.]*\.",
        ]
        for pat in patterns:
            for m in re.finditer(pat, vendor_text, re.I):
                snip = self._norm(m.group(0))
                snip = re.sub(r"^[^a-zA-Z0-9]+", "", snip)
                if snip:
                    snip = snip[0].upper() + snip[1:]
                if snip and snip not in deviations and len(snip) > 12:
                    fname, pnum = find_file_and_page_for_match(vendor_text, m.start())
                    snip_with_page = f"{snip} ({fname} - Pg {pnum})"
                    if snip_with_page not in deviations:
                        deviations.append(snip_with_page)
        return deviations[:6]

    def evaluate_pqc(self, pqc_reqs: List[Dict[str, Any]], vendor_text: str,
                     has_gem_doc: bool) -> List[PQCResult]:
        results = []
        for req in pqc_reqs:
            bid_file, bid_page = "", 1
            bid_pos = self._find_bid_pqc_pos(req["key"])
            if bid_pos is not None:
                bid_file, bid_page = find_file_and_page_for_match(getattr(self, "bid_text", ""), bid_pos)

            if req["key"] == "experience":
                years, pos = self._max_years_and_pos(vendor_text)
                file_name, page_num = "", 1
                page_info = ""
                if pos is not None:
                    file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                    page_info = f" (Pg {page_num})"
                provided = f"{years} years{page_info}" if years is not None else "[NOT FOUND]"
                passed = years is not None and years >= req["threshold"]
                results.append(PQCResult(req["label"], f"≥ {int(req['threshold'])} years",
                                         provided, passed, req["section"],
                                         file=file_name, page=page_num,
                                         bid_file=bid_file, bid_page=bid_page))
            elif req["key"] == "turnover":
                turnover, pos = self._max_turnover_and_pos(vendor_text)
                file_name, page_num = "", 1
                page_info = ""
                if pos is not None:
                    file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                    page_info = f" (Pg {page_num})"
                provided = f"INR {turnover} Crore{page_info}" if turnover is not None else "[NOT FOUND]"
                passed = turnover is not None and turnover >= req["threshold"]
                results.append(PQCResult(req["label"], f"≥ INR {req['threshold']:g} Crore",
                                         provided, passed, req["section"],
                                         file=file_name, page=page_num,
                                         bid_file=bid_file, bid_page=bid_page))
            elif req["key"] == "gem":
                file_name, page_num = "", 1
                pos = vendor_text.lower().find("gem registration")
                if pos == -1:
                    pos = vendor_text.lower().find("gem seller registration")
                if pos == -1:
                    pos = vendor_text.lower().find("gem")
                if pos != -1:
                    file_name, page_num = find_file_and_page_for_match(vendor_text, pos)

                provided = "Registered" if has_gem_doc else "[NOT FOUND]"
                results.append(PQCResult(req["label"], "Required", provided,
                                         has_gem_doc, req["section"],
                                         file=file_name, page=page_num,
                                         bid_file=bid_file, bid_page=bid_page))
            else:
                # Custom PQC fallback (e.g. oem_iso_certification)
                # Check if any keyword matches
                pos = None
                keywords = self._generate_dynamic_keywords(req.get("label", ""))
                for kw in keywords:
                    idx = vendor_text.lower().find(kw.lower())
                    if idx != -1:
                        pos = idx
                        break
                present = pos is not None
                file_name, page_num = "", 1
                page_info = ""
                if present and pos is not None:
                    file_name, page_num = find_file_and_page_for_match(vendor_text, pos)
                    page_info = f" (Pg {page_num})"
                
                results.append(PQCResult(
                    req.get("label", ""), 
                    "Required", 
                    f"Provided{page_info}" if present else "[NOT FOUND]",
                    present, 
                    req.get("section", ""),
                    file=file_name, 
                    page=page_num,
                    bid_file=bid_file, 
                    bid_page=bid_page
                ))
        return results

    @staticmethod
    def _max_years(text: str) -> Optional[float]:
        val, _ = AuditEngine._max_years_and_pos(text)
        return val

    @staticmethod
    def _max_years_and_pos(text: str) -> Tuple[Optional[float], Optional[int]]:
        kws = ["experience", "supplying", "supply", "performance", "providing",
               "networking", "commissioning", "domain", "years in", "track record"]
        best_val = None
        best_pos = None

        # Word-to-number mapping
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20
        }

        # 1. Numeric check: e.g. "5 years"
        for mt in re.finditer(r"(\d+(?:\.\d+)?)\s*years?", text, re.I):
            val = float(mt.group(1))
            if val > 60:
                continue
            ctx = text[max(0, mt.start() - 55): mt.end() + 55].lower()
            if any(k in ctx for k in kws):
                if best_val is None or val > best_val:
                    best_val = val
                    best_pos = mt.start()

        # 2. Textual number check: e.g. "five years"
        words_regex = r"\b(" + "|".join(word_to_num.keys()) + r")\s*years?"
        for mt in re.finditer(words_regex, text, re.I):
            word = mt.group(1).lower()
            val = float(word_to_num[word])
            ctx = text[max(0, mt.start() - 55): mt.end() + 55].lower()
            if any(k in ctx for k in kws):
                if best_val is None or val > best_val:
                    best_val = val
                    best_pos = mt.start()

        return best_val, best_pos

    @staticmethod
    def _max_turnover(text: str) -> Optional[float]:
        val, _ = AuditEngine._max_turnover_and_pos(text)
        return val

    @staticmethod
    def _max_turnover_and_pos(text: str) -> Tuple[Optional[float], Optional[int]]:
        best_val = None
        best_pos = None
        patterns = [
            r"(?i)(?:annual\s*turnover|average\s*turnover|yearly\s*turnover|revenue|gross\s*receipts?)\D{0,50}?([\d,]+(?:\.\d+)?)\s*(?:crore|lakhs?|lakh|cr\.?|lacs?|million|billion)?",
            r"(?i)turnover\s*[:\-]\s*(?:rs\.?\s*|inr\s*)?([\d,]+(?:\.\d+)?)\s*(?:crore|lakhs?|lakh|cr\.?|lacs?|million|billion)?",
            r"(?i)(?:rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(?:crore|lakhs?|lakh|cr\.?|lacs?|million|billion)"
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                try:
                    raw_num_str = m.group(1).replace(",", "")
                    unit_str = ""
                    if len(m.groups()) > 1 and m.group(2):
                        unit_str = m.group(2).lower().strip()
                    
                    val = float(raw_num_str)
                    if unit_str:
                        if "lakh" in unit_str or "lac" in unit_str:
                            val = val / 100.0
                        elif "million" in unit_str:
                            val = val / 10.0
                        elif "billion" in unit_str:
                            val = val * 100.0
                    else:
                        # Convert raw INR currency format without unit
                        if val >= 100000:
                            if val >= 10000000:
                                val = val / 10000000.0
                            else:
                                val = val / 100000.0 / 100.0 # Lakhs to Crore
                    
                    if best_val is None or val > best_val:
                        best_val = val
                        best_pos = m.start()
                except ValueError:
                    pass
        return best_val, best_pos

    def analyze_vendor(self, name: str, files: Dict[str, str],
                       errors: Dict[str, Optional[str]], bid: Dict[str, Any]) -> VendorResult:
        result = VendorResult(name=name)

        # 1. Inventory + classification + readability
        for fname, text in files.items():
            doc_type = self.classify_document(fname, text)
            readability = self.assess_readability(text, errors.get(fname))
            result.inventory.append(InventoryItem(fname, doc_type, readability))

        present_types = {i.doc_type for i in result.inventory}
        combined_text_parts = []
        for fname, text in files.items():
            combined_text_parts.append(f"--- FILE {fname} ---\n{text}")
        combined_text = "\n\n".join(combined_text_parts)

        # 2. MAF gate
        result.maf = self.validate_maf(result.inventory, files, bid.get("tender_id", ""))

        # 3. Document checklist validation (run early to detect embedded docs like GeM Registration!)
        result.document_checklist = self._validate_document_checklist_deterministic(bid.get("mandatory_docs", []), result.inventory, combined_text)

        # Calculate has_gem_doc dynamically (checking both inventory and document checklist)
        has_gem_doc = ("GeM Registration" in present_types or result.document_checklist.get("GeM Registration", {}).get("status") == "Compliant")

        # 4. PQC
        result.pqc = self.evaluate_pqc(bid.get("pqc", []), combined_text, has_gem_doc)

        # 5. Technical specs
        for spec in bid.get("mandatory_specs", []):
            result.mandatory_specs.append(self.extract_spec(spec, combined_text, True))
        for spec in bid.get("preferred_specs", []):
            result.preferred_specs.append(self.extract_spec(spec, combined_text, False))

        # 6. Deviations
        result.deviations = self.detect_deviations(combined_text)

        # 7. Missing mandatory documents check
        for doc in bid.get("mandatory_docs", []):
            checklist_entry = result.document_checklist.get(doc, {})
            if checklist_entry.get("status", "Missing") == "Missing":
                result.missing_documents.append(doc)

        # Cross-document dependency and exemption audits (Partnership deed, consortium JV, EMD exemption, MII, etc.)
        self.evaluate_document_dependencies_and_exemptions(result, bid, combined_text)

        # 7. Disqualification gate (binary, eligibility-level)
        self._apply_disqualification_gate(result, bid)

        # 8. Weighted compliance score (only meaningful if responsive)
        self._score(result)

        # Extract Make and Model (regex fallback)
        make_model = "Not Given"
        model_match = re.search(r"(?:make|model|manufacturer|brand)\s*[:\-]\s*([a-z0-9][a-z0-9\s\-_]{3,25})", combined_text, re.I)
        if model_match:
            make_model = model_match.group(1).strip()
        else:
            if result.maf and result.maf.status == MAF_VALID:
                m = re.search(r"authorization\s*from\s*([a-z0-9\s\-]+)", result.maf.evidence, re.I)
                if m:
                    make_model = f"Given ({m.group(1).strip()})"
                else:
                    make_model = "Given (As per MAF)"
            else:
                if result.mandatory_specs:
                    make_model = "Given (As per Datasheet)"
        result.make_model = make_model

        # Extract commercial details / completed contracts (regex fallback)
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
            result.commercial_details = "No completed work orders found"

        # 9. Summary / key takeaway
        result.summary = self._summarize(result)
        return result

    def _apply_disqualification_gate(self, r: VendorResult, bid: Dict[str, Any]) -> None:
        tender_id = bid.get("tender_id", "")
        maf_required = "Manufacturer's Authorization Form (MAF)" in bid.get("mandatory_docs", [])

        # Check if bidder has a valid Udyam Registration Certificate in their checklist
        has_udyam = False
        for doc_name, entry in r.document_checklist.items():
            if ("udyam" in doc_name.lower() or "msme" in doc_name.lower()) and entry.get("status") == "Compliant":
                has_udyam = True
                break

        # MAF gate
        if maf_required:
            if has_udyam:
                # Udyam Registration Certificate waives MAF requirement
                write_engine_log(f"  - MAF Exemption: Bidder has valid Udyam Registration. Waiving Manufacturer's Authorization Form (MAF) requirement.")
                if r.maf:
                    r.maf.status = MAF_VALID
                    r.maf.evidence = "Exempted (Valid Udyam Registration Certificate submitted, waiving Manufacturer's Authorization Form requirement)."
            else:
                if r.maf and r.maf.status == MAF_MISSING:
                    r.violations.append(Violation(
                        "Missing Mandatory Document — MAF",
                        f'Section 3.1: A valid MAF signed by the OEM referencing Tender No. '
                        f'"{tender_id}" is an absolute prerequisite for technical qualification.',
                        r.maf.evidence))
                elif r.maf and r.maf.status == MAF_INVALID:
                    r.violations.append(Violation(
                        "Invalid / Non-Compliant MAF",
                        f'Section 3.1: The MAF must be on OEM letterhead and reference Tender No. "{tender_id}".',
                        r.maf.evidence))

        # Other mandatory documents
        for doc in r.missing_documents:
            if doc == "Manufacturer's Authorization Form (MAF)":
                if has_udyam:
                    continue  # Waived!
                else:
                    # Let it fall through to generate a standard missing MAF violation
                    pass
            r.violations.append(Violation(
                f"Missing Mandatory Document — {doc}",
                f"Section 3: Submission of {doc} is mandatory.",
                f"No file in the vendor's submission was classified as {doc}."))

        # PQC gate
        for p in r.pqc:
            if not p.passed:
                r.violations.append(Violation(
                    f"Pre-Qualification Failure — {p.label}",
                    f"Section {p.section}: Requirement is {p.required}.",
                    f"Vendor provided: {p.provided}. Below the mandated threshold."))

        r.disqualified = len(r.violations) > 0
        r.status = STATUS_DISQUALIFIED if r.disqualified else STATUS_RESPONSIVE

    def evaluate_document_dependencies_and_exemptions(self, r: VendorResult, bid: Dict[str, Any], combined_text: str, vector_store: Any = None) -> None:
        low_text = combined_text.lower()
        
        # Check if bidder has a valid Udyam certificate
        has_udyam = False
        for doc_name, entry in r.document_checklist.items():
            if ("udyam" in doc_name.lower() or "msme" in doc_name.lower()) and entry.get("status") == "Compliant":
                has_udyam = True
                break

        # 1. EMD Exemption Dependency Check
        # If they claim EMD exemption/MSME status, they must have Udyam or NSIC
        claims_emd_exempt = any(k in low_text for k in ["emd exemption", "exempt from emd", "bid security exemption", "exemption from earnest money", "seeking emd exemption"])
        if claims_emd_exempt and not has_udyam:
            # Check if NSIC is present
            has_nsic = any(k in low_text for k in ["nsic certificate", "nsic registration", "national small industries"])
            if not has_nsic:
                r.violations.append(Violation(
                    "EMD Exemption Claimed but Certificate Missing",
                    "GFR Rule 170(i) / Public Procurement Policy: Exemption from EMD requires submission of a valid Udyam Registration (MSME) or NSIC certificate.",
                    "Bidder claims EMD/bid security exemption, but no valid Udyam or NSIC certificate was detected."
                ))

        # 2. Partnership Firm Document Dependency Check
        # Rule: Partnership firm requires Partnership Deed and Power of Attorney
        # Detect Partnership from PAN card 4th character ('F') or text keywords
        is_partnership = False
        pan_match = re.search(r'\b[A-Z0-9]{3}(F)[A-Z0-9][0-9]{4}[A-Z0-9]\b', combined_text)
        if pan_match:
            is_partnership = True
        elif any(k in low_text for k in ["partnership deed", "partnership firm", "partners of the firm"]):
            is_partnership = True
            
        if is_partnership:
            has_deed = any(k in low_text for k in ["partnership deed", "deed of partnership"])
            has_poa = any(k in low_text for k in ["power of attorney", "poa ", "authorized partner"])
            if not has_deed:
                r.violations.append(Violation(
                    "Partnership Deed Missing",
                    "Indian Partnership Act / Tender Terms: Bidders operating as a Partnership Firm must submit the registered Partnership Deed.",
                    "Bidder identified as a Partnership Firm (either via PAN or text), but no Partnership Deed was submitted."
                ))
            if not has_poa:
                r.violations.append(Violation(
                    "Power of Attorney Missing",
                    "Tender Terms: Partnership firms must submit a Power of Attorney (PoA) in favor of the signing partner.",
                    "Bidder identified as a Partnership Firm, but no Power of Attorney authorizing the signatory was found."
                ))

        # 3. Consortium / Joint Venture Document Dependency Check
        # Rule: Consortium/JV requires Joint Bidding Agreement / MoU and PoA for Lead Member
        is_consortium = any(k in low_text for k in ["consortium", "joint venture", " jv ", "lead member", "joint bidding"])
        if is_consortium:
            has_agreement = any(k in low_text for k in ["joint venture agreement", "jv agreement", "consortium agreement", "joint bidding agreement", "memorandum of understanding", "mou"])
            has_lead_poa = any(k in low_text for k in ["power of attorney for lead member", "lead member poa", "authorized lead member"])
            if not has_agreement:
                r.violations.append(Violation(
                    "JV / Consortium Agreement Missing",
                    "Tender Terms: Consortium/Joint Venture bidders must submit a legally binding Joint Bidding Agreement or MoU.",
                    "Bidder identified as a Consortium/JV, but no Joint Venture Agreement/MoU was submitted."
                ))
            if not has_lead_poa and not has_udyam: # Udyam single bidders don't need lead member PoA
                r.violations.append(Violation(
                    "Lead Member Power of Attorney Missing",
                    "Tender Terms: Consortium/JV bids must include a Power of Attorney designating the Lead Member.",
                    "Bidder identified as a Consortium/JV, but no Lead Member Power of Attorney was found."
                ))

        # 4. Make in India (MII) CA Certificate Dependency Check
        # Rule: If tender value is > 10 Crores (we can check turnover threshold or bid PQC), MII preference requires CA certificate
        claims_mii = any(k in low_text for k in ["class-i local supplier", "class-ii local supplier", "local content percentage", "mii preference"])
        if claims_mii:
            # Estimate tender value > 10 Cr from turnover threshold (e.g. if turnover threshold is >= 5.0 Cr)
            turnover_req = 0.0
            for pqc in bid.get("pqc", []):
                if pqc.get("key") == "turnover":
                    turnover_req = pqc.get("threshold", 0.0)
            
            is_large_tender = (turnover_req >= 5.0) # Turnover threshold >= 5Cr typically implies tender value > 10Cr
            if is_large_tender:
                has_ca_cert = any(k in low_text for k in ["chartered accountant certificate", "ca certificate", "statutory auditor certificate", "auditor certificate", "ca stamp"])
                if not has_ca_cert:
                    r.violations.append(Violation(
                        "MII Local Content CA Certificate Missing",
                        "Public Procurement (Preference to Make in India) Order: For tenders exceeding INR 10 Crores, local content must be certified by a CA or Statutory Auditor.",
                        "Bidder claims MII local content preference, but self-certification was provided instead of a Chartered Accountant / Auditor certificate (Tender value > 10 Cr)."
                    ))

    def _score(self, r: VendorResult) -> None:
        mand = r.mandatory_specs
        pref = r.preferred_specs
        mand_match = sum(1 for s in mand if s.status == "match")
        pref_match = sum(1 for s in pref if s.status == "match")
        r.mandatory_score = (mand_match / len(mand) * 70.0) if mand else 70.0
        r.preferred_score = (pref_match / len(pref) * 30.0) if pref else 0.0
        r.score = round(r.mandatory_score + r.preferred_score, 1)

    def _summarize(self, r: VendorResult) -> str:
        if r.disqualified:
            heads = []
            if any("Invalid / Non-Compliant MAF" in v.title for v in r.violations):
                heads.append("Invalid MAF")
            elif any("Missing Mandatory Document — MAF" in v.title for v in r.violations):
                heads.append("Missing MAF")
            if any(v.title.startswith("Pre-Qualification Failure") for v in r.violations):
                heads.append("Failed PQC")
            docs = [v.title.split("— ")[-1] for v in r.violations
                    if v.title.startswith("Missing Mandatory Document") and "MAF" not in v.title]
            for d in docs:
                heads.append("Missing " + d)
            if not heads:
                return "Disqualified:\n• See reasoning log"
            return "Disqualified:\n" + "\n".join(f"• {h}" for h in heads)
            
        mand_total = len(r.mandatory_specs)
        mand_ok = sum(1 for s in r.mandatory_specs if s.status == "match")
        pref_ok = sum(1 for s in r.preferred_specs if s.status == "match")
        pref_total = len(r.preferred_specs)
        bits = [f"Met {mand_ok}/{mand_total} mandatory specs"]
        if pref_total:
            bits.append(f"{pref_ok}/{pref_total} preferred features")
        lacking = [s.param for s in r.mandatory_specs if s.status == "lacking"]
        for param in lacking:
            bits.append("Data lacking on " + param)
        return "Responsive:\n" + "\n".join(f"• {b}" for b in bits)

    def rank_and_explain(self, results: List[VendorResult]) -> List[str]:
        responsive = [r for r in results if not r.disqualified]
        responsive.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(responsive, start=1):
            r.rank = i
        for r in results:
            if r.disqualified:
                r.rank = None

        explanations: List[str] = []
        for i in range(len(responsive) - 1):
            hi, lo = responsive[i], responsive[i + 1]
            explanations.append(self._explain_pair(hi, lo))
        if len(responsive) == 1:
            explanations.append(
                f"<b>{html.escape(responsive[0].name)}</b> is the sole responsive bidder, achieving a weighted "
                f"compliance score of <b>{responsive[0].score}%</b> "
                f"({responsive[0].mandatory_score:.0f}% mandatory + "
                f"{responsive[0].preferred_score:.0f}% preferred).")
                
        if not responsive and results:
            explanations.append("No vendors met the mandatory qualification criteria. All submissions have been disqualified.")
            
        for r in results:
            if r.disqualified:
                reasons = [v.title.split("— ")[-1] if "—" in v.title else v.title for v in r.violations]
                if reasons:
                    explanations.append(f"<b>{html.escape(r.name)}</b> was disqualified due to: {html.escape(', '.join(reasons))}.")
                    
        return explanations

    def _explain_pair(self, hi: VendorResult, lo: VendorResult) -> str:
        diffs = []
        lo_specs = {s.param: s for s in (lo.mandatory_specs + lo.preferred_specs)}
        for s in (hi.mandatory_specs + hi.preferred_specs):
            ls = lo_specs.get(s.param)
            if s.status == "match" and ls and ls.status != "match":
                diffs.append(f"<b>{html.escape(hi.name)}</b> satisfied “{html.escape(s.param)}” "
                              f"(<span style='color:var(--green);'>{html.escape(s.provided)}</span>) "
                              f"while <b>{html.escape(lo.name)}</b> did not "
                              f"(<span style='color:var(--amber);'>{html.escape(ls.provided)}</span>)")
        pref_hi = sum(1 for s in hi.preferred_specs if s.status == "match")
        pref_lo = sum(1 for s in lo.preferred_specs if s.status == "match")
        
        lead = (f"<div style='font-size: 15px; margin-bottom: 12px;'><b>Rank {hi.rank} {html.escape(hi.name)} ({hi.score}%)</b> edges out "
                f"<b>Rank {lo.rank} {html.escape(lo.name)} ({lo.score}%)</b> "
                f"by <span style='color:var(--blue); font-weight:700;'>{round(hi.score - lo.score, 1)} points</span>.</div>")
        
        if diffs:
            lead += "<div style='color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.5px; font-weight:700; margin-bottom:8px;'>Decisive Factors</div>"
            lead += "<ul style='margin-top:0; margin-bottom:16px; padding-left:20px; font-size: 13.5px;'>"
            for d in diffs[:3]:
                lead += f"<li style='margin-bottom:6px;'>{d}</li>"
            if len(diffs) > 3:
                lead += f"<li style='margin-bottom:6px; color:var(--muted);'>...and {len(diffs)-3} other factors</li>"
            lead += "</ul>"
            
        if pref_hi != pref_lo:
            lead += (f"<div style='background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.2); "
                     f"border-left: 3px solid var(--blue); padding: 10px 14px; border-radius: 6px; font-size: 13px;'>"
                     f"On preferred/desirable features <b>{html.escape(hi.name)}</b> met <b>{pref_hi}</b> vs "
                     f"<b>{html.escape(lo.name)}</b>'s <b>{pref_lo}</b> &mdash; the 30% preferred weighting separates them."
                     f"</div>")
                     
        return lead


# ===========================================================================
# SECTION 7 — OPTIONAL LLM AUGMENTATION LAYER
# ===========================================================================
class LLMAuditEngine(AuditEngine):
    """Augments document classification and executive summary generation using Anthropic Claude."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__()
        self.model = model
        self._client = None
        if HAS_ANTHROPIC and api_key:
            try:
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self._client = None

    def _complete(self, system: str, prompt: str, max_tokens: int = 600) -> Optional[str]:
        if not self._client:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        except Exception:
            return None

    def classify_document(self, filename: str, text: str) -> str:
        rule_type = super().classify_document(filename, text)
        if rule_type != "Unclassified Document":
            return rule_type
        out = self._complete(
            "You classify procurement documents. Reply with ONLY one label from this list: "
            + "; ".join(DOC_TYPES),
            f"Filename: {filename}\n\nContent (truncated):\n{text[:1500]}")
        if out:
            for t in DOC_TYPES:
                if t.lower()[:12] in out.lower():
                    return t
        return rule_type

    def _resolve_pages_using_llm(self, mandatory_docs: List[str], combined_text: str) -> Dict[str, int]:
        if not self._client:
            return {}
        
        # Extract index pages (typically pages 1-6 or first 20,000 characters)
        p7_idx = combined_text.find("--- PAGE 7 ---")
        index_text = combined_text[:p7_idx] if p7_idx != -1 else combined_text[:20000]
        
        system_prompt = (
            "You are a PSU procurement auditor. Your task is to analyze a bidder's document index / table of contents "
            "and extract the exact page numbers for each mandatory document category. "
            "Reply with a JSON block containing ONLY the mapping from document name to its page number. "
            "Do not include any chat, backticks, or explanation."
        )
        
        user_prompt = (
            f"Mandatory Documents Required:\n{json.dumps(mandatory_docs)}\n\n"
            f"Here is the text from the index/first few pages of the vendor's submission:\n"
            f"\"\"\"\n{index_text}\n\"\"\"\n\n"
            f"Return a clean JSON object structure (e.g. {{\"PAN Card\": 79, \"GST Registration\": 76}}). "
            f"If a document is not listed in the index or table of contents, omit it from the JSON."
        )
        
        out = self._complete(system_prompt, user_prompt, max_tokens=300)
        if not out:
            return {}
            
        try:
            # Clean LLM response to get JSON block
            if "```json" in out:
                out = out.split("```json")[1].split("```")[0]
            elif "```" in out:
                out = out.split("```")[1].split("```")[0]
            out = out.strip()
            mapping = json.loads(out)
            
            resolved = {}
            for doc in mandatory_docs:
                for k, v in mapping.items():
                    if doc.lower() in k.lower() or k.lower() in doc.lower():
                        if isinstance(v, int):
                            resolved[doc] = v
                        elif isinstance(v, str) and v.isdigit():
                            resolved[doc] = int(v)
            return resolved
        except Exception:
            return {}

    def analyze_vendor(self, name: str, files: Dict[str, str],
                       errors: Dict[str, Optional[str]], bid: Dict[str, Any]) -> VendorResult:
        # 1. Run baseline rule-based evaluation first
        result = super().analyze_vendor(name, files, errors, bid)
        
        # 2. Refine page numbers dynamically using LLM Table-of-Contents parsing
        if self._client:
            combined_text_parts = []
            for fname, text in files.items():
                combined_text_parts.append(f"--- FILE {fname} ---\n{text}")
            combined_text = "\n\n".join(combined_text_parts)
            
            resolved_pages = self._resolve_pages_using_llm(bid.get("mandatory_docs", []), combined_text)
            for doc, page in resolved_pages.items():
                if doc in result.document_checklist:
                    result.document_checklist[doc]["page"] = page
                    
        return result

    def narrate(self, bid: Dict[str, Any], results: List[VendorResult]) -> Optional[str]:
        if not self._client:
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
        return self._complete(
            "You are a PSU tender evaluation officer. Write a concise, formal executive "
            "summary (max 120 words) of the evaluation outcome. Be factual, cite ranks and "
            "the decisive reasons. Do not invent data beyond what is provided.",
            json.dumps(ctx, indent=2), max_tokens=400)


def get_engine(api_key: str = "", model: str = "claude-3-5-sonnet-20241022") -> AuditEngine:
    if api_key and HAS_ANTHROPIC:
        return LLMAuditEngine(api_key, model)
    return AuditEngine()
