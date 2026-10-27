"""
rag_engine.py
Core Retrieval-Augmented Generation (RAG) Engine for Home Appliance Troubleshooting & User Guide Assistant.

Technologies:
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2) via ChromaDB embedding function
- Vector Store: ChromaDB (Persistent local storage)
- LLM API: Google Gemini API (gemini-2.5-flash / gemini-1.5-flash via google.genai)
- Document Processing: PyPDF (PDF), Pandas (CSV), Built-in Text Handling (TXT)
"""

import os
import io
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# Optional dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try importing Google GenAI SDK
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# Known appliance categories for classification and filtering
APPLIANCE_CATEGORIES = [
    "Washing Machine",
    "Refrigerator",
    "Air Conditioner",
    "Microwave Oven",
    "Dishwasher",
    "General Appliance"
]

# High-risk safety keywords that require immediate warning flags
HAZARDOUS_PATTERNS = [
    (r"\b(high voltage|capacitor|magnetron)\b", 
     "DANGER: Microwave magnetrons and high-voltage capacitors store lethal electrical charges (2,000V+) even when unplugged. Never remove the cabinet or touch internal electrical components."),
    (r"\b(refrigerant|freon|r600a|r134a|copper pipe|brazing|soldering)\b", 
     "SAFETY WARNING: Sealed refrigerant lines are under high pressure and contain flammable/asphyxiating gases. Never cut or puncture refrigerant lines. Contact an EPA/HVAC certified technician."),
    (r"\b(open wire|sparking cord|shock|burnt smell from outlet)\b", 
     "ELECTRICAL HAZARD: Disconnect main breaker immediately. Do not touch frayed cords or smoking outlets. Replace damaged power components before re-energizing.")
]


def detect_appliance_category(text_or_filename: str) -> str:
    """Detect the appliance category based on filename or text content."""
    lower = text_or_filename.lower()
    if any(k in lower for k in ["dishwasher", "dish-washer", "aquaclean"]):
        return "Dishwasher"
    elif any(k in lower for k in ["wash", "wm-", "washer", "laundry", "dryer"]):
        return "Washing Machine"
    elif any(k in lower for k in ["refrig", "fridge", "freezer", "frost"]):
        return "Refrigerator"
    elif any(k in lower for k in ["air condition", " ac ", "coolwave", "split ac", "inverter ac"]):
        return "Air Conditioner"
    elif any(k in lower for k in ["microwave", "oven", "magnetron", "turntable"]):
        return "Microwave Oven"
    elif "dish" in lower:
        return "Dishwasher"
    return "General Appliance"


def compute_doc_hash(content: str) -> str:
    """Compute MD5 hash to detect duplicate documents and avoid re-indexing."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def chunk_text(text: str, chunk_size: int = 650, chunk_overlap: int = 120) -> List[str]:
    """
    Split long document text into overlapping chunks respecting sentence/paragraph boundaries.
    """
    # Clean redundant whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    
    if len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    # Split primarily on paragraphs or section headers
    paragraphs = re.split(r"(\n\n+|(?=###?\s))", text)
    
    current_chunk = ""
    for segment in paragraphs:
        if not segment.strip():
            continue
            
        if len(current_chunk) + len(segment) <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + segment.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: keep the tail of current chunk
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + segment.strip()
            else:
                # Segment alone exceeds chunk_size, split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", segment)
                sub_chunk = ""
                for s in sentences:
                    if len(sub_chunk) + len(s) <= chunk_size:
                        sub_chunk += (" " if sub_chunk else "") + s
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = s
                if sub_chunk:
                    current_chunk = sub_chunk
                    
    if current_chunk and current_chunk.strip() not in chunks:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if len(c.strip()) > 20]


class ApplianceRAGEngine:
    """
    Manages Document Ingestion, ChromaDB Vector Indexing, Semantic Retrieval,
    and Grounded Answer Generation via Google Gemini API.
    """

    def __init__(self, db_persist_dir: str = "./chroma_db"):
        self.db_persist_dir = os.path.abspath(db_persist_dir)
        os.makedirs(self.db_persist_dir, exist_ok=True)
        
        # 1. Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.db_persist_dir)
        
        # 2. Embedding Model: Sentence Transformers (all-MiniLM-L6-v2) 384 dimensions
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # 3. Create or load collection
        self.collection_name = "appliance_manuals_knowledge_base"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "description": "Home Appliance Troubleshooting & Maintenance Manuals"}
        )

    def extract_text_from_file(self, file_source: Any, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text and structured chunks from PDF, TXT, or CSV files.
        Returns: (full_text, list_of_chunk_dicts)
        """
        ext = os.path.splitext(filename)[1].lower()
        chunks_data: List[Dict[str, Any]] = []
        full_text = ""
        category = detect_appliance_category(filename)

        if ext == ".txt":
            if isinstance(file_source, (str, os.PathLike)):
                with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
            else:
                full_text = file_source.read().decode("utf-8", errors="ignore")
                
            # Content-based category refinement
            if category == "General Appliance":
                category = detect_appliance_category(full_text[:300])

            raw_chunks = chunk_text(full_text)
            for i, chunk in enumerate(raw_chunks):
                chunks_data.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "appliance": category,
                        "chunk_index": i,
                        "file_type": "TXT"
                    }
                })

        elif ext == ".pdf":
            if isinstance(file_source, (str, os.PathLike)):
                reader = PdfReader(file_source)
            else:
                reader = PdfReader(file_source)
                
            pages_text = []
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages_text.append(extracted)
                    raw_chunks = chunk_text(extracted)
                    for i, chunk in enumerate(raw_chunks):
                        chunks_data.append({
                            "text": chunk,
                            "metadata": {
                                "source": filename,
                                "appliance": category,
                                "page": page_num + 1,
                                "chunk_index": i,
                                "file_type": "PDF"
                            }
                        })
            full_text = "\n\n".join(pages_text)

        elif ext == ".csv":
            if isinstance(file_source, (str, os.PathLike)):
                df = pd.read_csv(file_source)
            else:
                df = pd.read_csv(file_source)

            # Build natural text passages from tabular columns
            rows_texts = []
            for i, row in df.iterrows():
                appliance_val = row.get("Appliance") or row.get("Category") or category
                issue_val = row.get("Issue") or row.get("Problem") or "Appliance Issue"
                err_code = row.get("Error_Code") or row.get("Code") or "N/A"
                cause_val = row.get("Possible_Cause") or row.get("Cause") or "Not specified"
                steps_val = row.get("Troubleshooting_Steps") or row.get("Solution") or row.get("Steps") or ""
                safety_val = row.get("Safety_Precaution") or row.get("Safety") or row.get("Precaution") or ""

                passage = (
                    f"Appliance: {appliance_val}\n"
                    f"Issue / Symptom: {issue_val} (Error Code: {err_code})\n"
                    f"Possible Causes: {cause_val}\n"
                    f"Troubleshooting Steps: {steps_val}\n"
                    f"Safety Precautions: {safety_val}"
                ).strip()

                rows_texts.append(passage)
                chunks_data.append({
                    "text": passage,
                    "metadata": {
                        "source": filename,
                        "appliance": str(appliance_val),
                        "row_index": int(i),
                        "error_code": str(err_code),
                        "file_type": "CSV"
                    }
                })
            full_text = "\n\n---\n\n".join(rows_texts)

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .csv")

        return full_text, chunks_data

    def ingest_document(self, file_source: Any, filename: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Process, chunk, and embed a document into ChromaDB.
        Avoids duplicates unless force_update=True.
        """
        full_text, chunks_data = self.extract_text_from_file(file_source, filename)
        if not chunks_data:
            return {"status": "empty", "message": f"No readable text found in {filename}", "chunks_added": 0}

        doc_hash = compute_doc_hash(full_text)

        # Check existing items with this source
        existing = self.collection.get(where={"source": filename})
        if existing and existing["ids"]:
            if not force_update:
                return {
                    "status": "already_exists",
                    "message": f"'{filename}' is already ingested in knowledge base ({len(existing['ids'])} chunks).",
                    "chunks_added": 0
                }
            else:
                # Remove existing chunks for clean update
                self.collection.delete(where={"source": filename})

        # Prepare for ChromaDB addition
        ids = [f"{doc_hash}_{idx}" for idx in range(len(chunks_data))]
        documents = [c["text"] for c in chunks_data]
        metadatas = [c["metadata"] for c in chunks_data]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        appliance_type = chunks_data[0]["metadata"].get("appliance", "General Appliance")

        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}' ({len(chunks_data)} chunks) under category '{appliance_type}'.",
            "chunks_added": len(chunks_data),
            "appliance": appliance_type
        }

    def ingest_sample_manuals(self, sample_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scan sample manuals directory and ingest all sample guides.
        """
        if sample_dir is None:
            sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_manuals")
            
        results = []
        if not os.path.exists(sample_dir):
            return results

        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".txt", ".csv", ".pdf")):
                fpath = os.path.join(sample_dir, fname)
                res = self.ingest_document(fpath, fname, force_update=False)
                results.append(res)
        return results

    def retrieve_relevant_chunks(
        self,
        query: str,
        appliance_filter: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search in ChromaDB using embeddings.
        Supports appliance category filtering.
        """
        if self.collection.count() == 0:
            return []

        where_clause = None
        if appliance_filter and appliance_filter != "All Appliances":
            where_clause = {"appliance": appliance_filter}

        query_res = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where_clause
        )

        retrieved_items = []
        if query_res and query_res["documents"] and query_res["documents"][0]:
            docs = query_res["documents"][0]
            metas = query_res["metadatas"][0] if query_res["metadatas"] else [{}] * len(docs)
            distances = query_res["distances"][0] if "distances" in query_res and query_res["distances"] else [0.0] * len(docs)
            ids = query_res["ids"][0] if query_res["ids"] else [""] * len(docs)

            for d, m, dist, cid in zip(docs, metas, distances, ids):
                # Cosine similarity estimation from L2/cosine distance
                similarity_score = max(0.0, 1.0 - (dist if dist is not None else 0.5))
                retrieved_items.append({
                    "id": cid,
                    "text": d,
                    "metadata": m,
                    "distance": dist,
                    "similarity_score": round(similarity_score, 4)
                })

        return retrieved_items

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        Generate a structured, source-grounded response using Google Gemini API.
        Enforces safety rules and handles unknown questions gracefully.
        """
        # Safety hazard scan
        safety_warnings = []
        for pattern, warning in HAZARDOUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                safety_warnings.append(warning)

        # Check if retrieved chunks are sufficiently relevant
        # If no chunks or highest similarity is below threshold (e.g. < 0.38)
        similarity_threshold = 0.38
        has_relevant_info = bool(retrieved_chunks) and any(c.get("similarity_score", 0.0) >= similarity_threshold for c in retrieved_chunks)

        if not has_relevant_info:
            fallback_answer = (
                "### 🔍 Verification Status: Information Unavailable\n\n"
                "I cannot find verified troubleshooting steps or manual guidance for this question "
                "in the current knowledge base.\n\n"
                "**Safety Recommendation:**\n"
                "- Please do not guess or attempt unauthorized electrical or mechanical repairs.\n"
                "- Refer to your appliance's original manufacturer manual or contact an authorized service center.\n"
                "- If you have the manual, upload it using the document uploader to update the assistant's knowledge base."
            )
            return {
                "answer": fallback_answer,
                "sources": [],
                "safety_warnings": safety_warnings,
                "model_used": "Knowledge Base Verification Rule",
                "is_grounded": False
            }

        # Build context from top chunks
        context_blocks = []
        for i, c in enumerate(retrieved_chunks):
            src = c["metadata"].get("source", "Manual")
            app = c["metadata"].get("appliance", "Appliance")
            context_blocks.append(f"[Source {i+1}: {src} | Appliance: {app}]\n{c['text']}")
        joined_context = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert, highly meticulous Home Appliance Troubleshooting & User Guide Assistant.\n"
            "Your job is to provide accurate, easy-to-understand, safety-conscious answers grounded STRICTLY "
            "in the provided manual excerpts.\n\n"
            "RULES TO FOLLOW:\n"
            "1. Base your answer ONLY on the retrieved excerpts. Do NOT hallucinate or extrapolate beyond the provided text.\n"
            "2. If the excerpts do not contain the answer, explicitly state: 'The provided appliance manuals do not contain verified information for this issue.'\n"
            "3. Format your response clearly using standard Markdown:\n"
            "   - **Appliance & Issue Detected**: State what appliance and fault you identified.\n"
            "   - **Probable Cause(s)**: Bullet points explaining why this happens.\n"
            "   - **Step-by-Step Solution**: Numbered actionable troubleshooting steps.\n"
            "   - **Critical Safety Precautions**: Emphasize uncoupling power, electrical shock warnings, avoiding DIY repair of hazardous parts, and when to call a certified technician.\n"
            "4. Be direct, respectful, and authoritative on safety."
        )

        user_content = (
            f"USER QUESTION:\n{query}\n\n"
            f"RETRIEVED MANUAL EXCERPTS:\n{joined_context}\n\n"
            f"Please generate the troubleshooting solution following the prescribed format."
        )

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if resolved_key and GENAI_AVAILABLE:
            try:
                client = genai.Client(api_key=resolved_key)
                # Try preferred model or fallback
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_content,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,  # Low temperature for strict factual grounding
                        max_output_tokens=1000
                    )
                )
                answer_text = response.text
                model_used = model_name
            except Exception as e:
                # Graceful handling for API error or quota
                answer_text = (
                    f"⚠️ **Gemini API Error:** `{str(e)}`\n\n"
                    f"**Retrieved Context Summary (Ground Truth):**\n\n"
                    + "\n\n".join([f"- **From {c['metadata'].get('source')}**: {c['text'][:250]}..." for c in retrieved_chunks[:3]])
                )
                model_used = "Offline Retrieval Fallback (API Error)"
        else:
            # Clean offline grounded synthesizer (demonstrates RAG retrieval without requiring an active LLM key)
            key_notice = "*(Note: Live Gemini API key not provided or offline mode active; showing deterministic grounded extraction from retrieved manuals)*\n\n"
            bullets = []
            for i, c in enumerate(retrieved_chunks[:3]):
                src = c["metadata"].get("source", "Document")
                app = c["metadata"].get("appliance", "Appliance")
                score = c.get("similarity_score", 0.0)
                bullets.append(
                    f"#### 📄 Source {i+1}: {src} ({app}) - Relevance: {score*100:.1f}%\n"
                    f"{c['text']}\n"
                )
            
            offline_solution = (
                f"{key_notice}"
                f"### 🛠️ Extracted Appliance Troubleshooting Guide\n\n"
                f"Based on the most relevant sections retrieved from **{retrieved_chunks[0]['metadata'].get('source')}**:\n\n"
                + "\n".join(bullets)
                + "\n\n### ⚠️ General Safety Precaution\n"
                "- Always unplug the appliance from electrical mains before cleaning or inspecting internal components.\n"
                "- If the issue involves gas, high-voltage capacitors, or refrigerant, consult an authorized technician immediately."
            )
            answer_text = offline_solution
            model_used = "ChromaDB Semantic Extraction (Offline RAG Mode)"

        return {
            "answer": answer_text,
            "sources": retrieved_chunks,
            "safety_warnings": safety_warnings,
            "model_used": model_used,
            "is_grounded": True
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return counts and metadata statistics of the current vector database."""
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "sources": [],
                "appliances": []
            }

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        sources = set()
        appliances = set()
        for m in all_meta:
            if m:
                if "source" in m:
                    sources.add(m["source"])
                if "appliance" in m:
                    appliances.add(m["appliance"])

        return {
            "total_chunks": total_chunks,
            "sources": sorted(list(sources)),
            "appliances": sorted(list(appliances))
        }

    def clear_database(self) -> None:
        """Clear all entries in the ChromaDB collection."""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "description": "Home Appliance Troubleshooting & Maintenance Manuals"}
        )
