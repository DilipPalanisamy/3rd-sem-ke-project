"""
app.py
Streamlit Web Application for AI-Powered Home Appliance Troubleshooting & User Guide Assistant
Using Retrieval-Augmented Generation (RAG).

Tech Stack:
- Python 3.12
- Streamlit
- ChromaDB Vector Store
- Sentence Transformers (all-MiniLM-L6-v2)
- Google Gemini API (gemini-2.5-flash)
- PyPDF, Pandas
"""

import os
import streamlit as st
from rag_engine import ApplianceRAGEngine, APPLIANCE_CATEGORIES, GENAI_AVAILABLE

# Page configuration
st.set_page_config(
    page_title="Appliance Troubleshooting AI Assistant (RAG)",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
CUSTOM_CSS = """
<style>
/* Modern typography and general styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}
.main-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #FFFFFF !important;
}
.main-header p {
    font-size: 14px;
    margin: 0;
    opacity: 0.92;
    color: #E2E8F0 !important;
}

/* Safety Alert Banner */
.safety-alert-box {
    background: #FEF2F2;
    border-left: 5px solid #EF4444;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 22px;
    color: #991B1B;
    font-size: 13.5px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.safety-alert-box strong {
    color: #7F1D1D;
}

/* Pill Badges */
.badge-pill {
    display: inline-block;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 6px;
}
.badge-blue { background: #DBEAFE; color: #1E40AF; }
.badge-teal { background: #CCFBF1; color: #0F766E; }
.badge-amber { background: #FEF3C7; color: #92400E; }
.badge-purple { background: #EDE9FE; color: #5B21B6; }

/* Response Box */
.assistant-response-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px;
    margin-top: 14px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* Metric cards in sidebar */
.metric-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-number {
    font-size: 22px;
    font-weight: 700;
    color: #1E3A8A;
}
.metric-label {
    font-size: 11.5px;
    color: #64748B;
    font-weight: 500;
}

/* Excerpt box in citations */
.source-citation-card {
    background: #F8FAFC;
    border-left: 4px solid #0F766E;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 12.5px;
    margin-bottom: 10px;
    color: #334155;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize RAG Engine in Streamlit Cache/State
@st.cache_resource(show_spinner="Initializing ChromaDB & Sentence Transformer Vector Engine...")
def get_rag_engine():
    engine = ApplianceRAGEngine(db_persist_dir="./chroma_db")
    # Ingest default sample manuals if collection is empty
    if engine.collection.count() == 0:
        engine.ingest_sample_manuals()
    return engine

engine = get_rag_engine()


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""


# SIDEBAR: Control Panel & Knowledge Base Manager
with st.sidebar:
    st.markdown("### ⚙️ System Control & KB")
    
    # 1. API Configuration
    st.markdown("#### 🔑 Gemini API Settings")
    default_env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.get("api_key", default_env_key),
        type="password",
        help="Free API key from Google AI Studio (aistudio.google.com). If empty, app runs in deterministic semantic extraction mode."
    )
    st.session_state["api_key"] = api_key_input
    
    model_choice = st.selectbox(
        "LLM Generation Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Fast, lightweight model for grounded retrieval."
    )

    if api_key_input:
        st.success("API Key configured", icon="✅")
    else:
        st.info("No API key provided. App will run in Offline Semantic Search mode with source verification.", icon="ℹ️")

    st.markdown("---")

    # 2. Vector DB Stats
    stats = engine.get_collection_stats()
    st.markdown("#### 📊 Vector Store Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{stats["total_chunks"]}</div>'
            f'<div class="metric-label">Total Chunks</div></div>',
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-number">{len(stats["sources"])}</div>'
            f'<div class="metric-label">Manuals Ingested</div></div>',
            unsafe_allow_html=True
        )

    with st.expander("📁 Ingested Manuals & Categories", expanded=False):
        for s in stats["sources"]:
            st.markdown(f"- 📄 `{s}`")
        st.markdown("**Categories Covered:**")
        for a in stats["appliances"]:
            st.markdown(f"- 🏷️ {a}")

    st.markdown("---")

    # 3. Document Uploader & Update
    st.markdown("#### 📤 Upload New Appliance Manual")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or CSV Manual",
        type=["pdf", "txt", "csv"],
        help="Add new manuals to update the RAG knowledge base dynamically without code modification."
    )
    overwrite_existing = st.checkbox("Overwrite if file already exists", value=True)

    if uploaded_file is not None:
        if st.button("🚀 Ingest & Update Knowledge Base", use_container_width=True):
            with st.spinner(f"Extracting, chunking, and embedding '{uploaded_file.name}'..."):
                try:
                    result = engine.ingest_document(
                        uploaded_file,
                        uploaded_file.name,
                        force_update=overwrite_existing
                    )
                    if result["status"] == "success":
                        st.success(result["message"], icon="✅")
                        st.rerun()
                    elif result["status"] == "already_exists":
                        st.warning(result["message"], icon="⚠️")
                    else:
                        st.error(result["message"], icon="❌")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}", icon="🚨")

    st.markdown("---")

    # 4. Maintenance / Reset
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Reload Samples", use_container_width=True, help="Re-indexes the 5 default sample manuals"):
            engine.clear_database()
            engine.ingest_sample_manuals()
            st.success("Sample manuals reloaded!")
            st.rerun()
    with col_r2:
        if st.button("🗑️ Clear KB", use_container_width=True, help="Wipes all chunks in ChromaDB"):
            engine.clear_database()
            st.warning("Database cleared!")
            st.rerun()


# MAIN CONTENT AREA
# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>🛠️ AI-Powered Home Appliance Troubleshooting & User Guide Assistant</h1>
        <p>Grounded Retrieval-Augmented Generation (RAG) system using ChromaDB vector database, Sentence Transformers, and Google Gemini API</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Safety Alert Banner
st.markdown(
    """
    <div class="safety-alert-box">
        <strong>⚠️ CRITICAL SAFETY NOTICE:</strong> Always disconnect the appliance from the mains electrical power supply before inspecting filters, hoses, or interior parts. Never attempt DIY repair on microwave high-voltage capacitors (lethal 2000V+ charge), air conditioner / refrigerator pressurized refrigerant copper lines, or gas lines. When in doubt, always contact a certified, authorized technician.
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs
tab_chat, tab_kb, tab_viva = st.tabs([
    "💬 Troubleshooting Assistant",
    "📚 Knowledge Repository Explorer",
    "🔬 System Architecture & Viva Q&A"
])


# ==============================================================================
# TAB 1: TROUBLESHOOTING ASSISTANT
# ==============================================================================
with tab_chat:
    # Filter Row
    col_filter, col_topk = st.columns([3, 1])
    with col_filter:
        appliance_filter = st.selectbox(
            "Filter Knowledge Search by Appliance Category:",
            options=["All Appliances"] + APPLIANCE_CATEGORIES,
            index=0,
            help="Narrow vector retrieval to a specific appliance or search across all uploaded manuals."
        )
    with col_topk:
        top_k_val = st.slider("Retrieved Context Chunks (Top-K):", min_value=1, max_value=6, value=4)

    # Sample Questions Buttons
    st.markdown("##### 💡 Quick Sample Troubleshooting Questions (Click to test):")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    def set_sample_query(q_text):
        st.session_state["user_question"] = q_text

    with sample_col1:
        if st.button("🧺 Washer Error E2 (Not Draining)", use_container_width=True):
            set_sample_query("Why is my washing machine showing error E2 and not draining water?")
        if st.button("❄️ Refrigerator Frost & Water Pooling", use_container_width=True):
            set_sample_query("Why is water pooling under the crisper drawers and frost building up in the fridge?")

    with sample_col2:
        if st.button("⚡ Microwave Sparking Inside Cavity", use_container_width=True):
            set_sample_query("What causes sparking or arcing inside a microwave oven and how do I fix it?")
        if st.button("🌡️ AC Blowing Warm Air / Ice on Pipe", use_container_width=True):
            set_sample_query("My air conditioner is blowing warm air instead of cooling and there is ice on the pipe.")

    with sample_col3:
        if st.button("🍽️ Dishwasher Error E15 AquaStop", use_container_width=True):
            set_sample_query("What does error code E15 on a dishwasher mean and how do I resolve it?")
        if st.button("✈️ Unknown Query (Hallucination Test)", use_container_width=True):
            set_sample_query("How do I fix the turbo compressor on a commercial Boeing 777 jet engine?")

    st.markdown("")

    # Chat Input Form
    with st.form(key="rag_query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your appliance question or error code:",
            value=st.session_state.get("user_question", ""),
            placeholder="e.g. My washing machine drum won't spin and gives UE error code..."
        )
        submit_button = st.form_submit_button("🔍 Diagnose & Find Solution", use_container_width=True)

    if submit_button and user_query.strip():
        with st.spinner("Searching ChromaDB vector repository and generating grounded solution..."):
            # 1. Retrieval
            retrieved_chunks = engine.retrieve_relevant_chunks(
                query=user_query,
                appliance_filter=appliance_filter,
                top_k=top_k_val
            )

            # 2. Grounded Generation
            rag_output = engine.generate_grounded_answer(
                query=user_query,
                retrieved_chunks=retrieved_chunks,
                api_key=st.session_state.get("api_key"),
                model_name=model_choice
            )

            # Add to conversation history
            st.session_state["chat_history"].insert(0, {
                "query": user_query,
                "answer": rag_output["answer"],
                "sources": rag_output["sources"],
                "safety_warnings": rag_output["safety_warnings"],
                "model_used": rag_output["model_used"],
                "is_grounded": rag_output["is_grounded"],
                "appliance_filter": appliance_filter
            })

    # Clear History Button
    if st.session_state["chat_history"]:
        if st.button("🧹 Clear Conversation History"):
            st.session_state["chat_history"] = []
            st.session_state["user_question"] = ""
            st.rerun()

    # Display History
    if not st.session_state["chat_history"]:
        st.info("👋 Ask an appliance troubleshooting question or click one of the sample buttons above to see the RAG retrieval pipeline in action!", icon="💡")
    else:
        for idx, chat in enumerate(st.session_state["chat_history"]):
            st.markdown(f"### ❓ Query: *\"{chat['query']}\"*")
            
            # Badges
            filter_badge = chat.get("appliance_filter", "All Appliances")
            st.markdown(
                f'<span class="badge-pill badge-blue">Category: {filter_badge}</span>'
                f'<span class="badge-pill badge-teal">Engine: {chat["model_used"]}</span>'
                f'<span class="badge-pill badge-purple">Retrieved Chunks: {len(chat["sources"])}</span>',
                unsafe_allow_html=True
            )

            # Specific Hazardous Query Warning
            if chat["safety_warnings"]:
                for warn in chat["safety_warnings"]:
                    st.error(warn, icon="🚨")

            # Answer Card
            st.markdown('<div class="assistant-response-card">', unsafe_allow_html=True)
            st.markdown(chat["answer"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Expandable Source Citations (Explainability & Verification)
            with st.expander(f"🔎 View Source Citations & Retrieved Document Excerpts ({len(chat['sources'])} matches)", expanded=True):
                if not chat["sources"]:
                    st.write("No matching document excerpts were found in the knowledge base.")
                else:
                    for s_idx, src in enumerate(chat["sources"]):
                        meta = src.get("metadata", {})
                        source_file = meta.get("source", "Unknown Manual")
                        app_cat = meta.get("appliance", "Appliance")
                        sim_pct = src.get("similarity_score", 0.0) * 100
                        page_info = f" | Page {meta.get('page')}" if "page" in meta else ""
                        file_type = meta.get("file_type", "DOC")

                        st.markdown(
                            f'<div class="source-citation-card">'
                            f'<strong>Source {s_idx+1}:</strong> <code>{source_file}</code> ({file_type}) '
                            f'| <strong>Appliance:</strong> {app_cat}{page_info} '
                            f'| <strong>Cosine Similarity:</strong> {sim_pct:.1f}%<br>'
                            f'<div style="margin-top:6px; font-family:monospace; font-size:11.5px; white-space:pre-wrap;">{src["text"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            st.markdown("---")


# ==============================================================================
# TAB 2: KNOWLEDGE REPOSITORY EXPLORER
# ==============================================================================
with tab_kb:
    st.markdown("### 📚 ChromaDB Knowledge Base Contents")
    st.write(
        "Inspect all indexed appliance manuals, technical guides, and document chunks currently "
        "stored in the persistent ChromaDB vector store."
    )

    kb_stats = engine.get_collection_stats()
    
    if kb_stats["total_chunks"] == 0:
        st.warning("The knowledge base is currently empty. Click 'Reload Samples' in the sidebar or upload a manual.")
    else:
        st.markdown(f"**Total Indexed Chunks:** `{kb_stats['total_chunks']}` across `{len(kb_stats['sources'])}` documents.")
        
        all_data = engine.collection.get(include=["documents", "metadatas"])
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        ids = all_data["ids"]

        # Search inside KB
        kb_search = st.text_input("Filter chunks by keyword:", placeholder="e.g. E1, filter, gasket, capacitor, impeller...")

        displayed = 0
        for doc_id, doc_text, meta in zip(ids, docs, metas):
            if kb_search and kb_search.lower() not in doc_text.lower() and kb_search.lower() not in meta.get("source", "").lower():
                continue

            displayed += 1
            src = meta.get("source", "Manual")
            app = meta.get("appliance", "General")
            ftype = meta.get("file_type", "TEXT")

            with st.container():
                st.markdown(
                    f"**Chunk ID:** `{doc_id}` | **Source:** `{src}` | **Appliance:** `{app}` | **Type:** `{ftype}`"
                )
                st.code(doc_text, language="markdown")
                st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
                
            if displayed >= 25:
                st.caption("Showing first 25 matching chunks. Use keyword filter to refine.")
                break


# ==============================================================================
# TAB 3: SYSTEM ARCHITECTURE & VIVA Q&A
# ==============================================================================
with tab_viva:
    st.markdown("### 🔬 System Architecture & Viva Voce Guide")
    st.markdown(
        "Comprehensive technical documentation of the Retrieval-Augmented Generation (RAG) pipeline, "
        "algorithms, and core questions for academic evaluation."
    )

    col_arch1, col_arch2 = st.columns([1, 1])

    with col_arch1:
        st.markdown("#### 🔄 End-to-End RAG Architecture")
        st.markdown(
            """
            ```
            [User Uploads Manuals (PDF/TXT/CSV)]
                            │
                            ▼
               [Document Text Extraction]
               - PyPDF (PDF pages)
               - Built-in UTF-8 parser (TXT)
               - Pandas DataFrame (CSV rows)
                            │
                            ▼
             [Sliding-Window Text Chunking]
             - 650 chars with 120-char overlap
             - Sentence boundary preservation
                            │
                            ▼
             [Sentence Transformers Embedding]
             - Model: all-MiniLM-L6-v2
             - 384-dimensional dense vectors
                            │
                            ▼
             [ChromaDB Vector Store (Persistent)]
             - HNSW Cosine Indexing
             - Metadata: Appliance, Source, Type
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      [Offline Storage]            [User Query Entered]
      ./chroma_db                  - Appliance Category Filter
                                   - Dense Query Embedding
                                           │
                                           ▼
                              [Top-K Semantic Retrieval]
                              - Cosine Similarity Matching
                              - Top-4 Chunks Extracted
                                           │
                                           ▼
                             [Strict Prompt Augmentation]
                             - System Rules + Safety Shield
                             - Retrieved Excerpts Context
                                           │
                                           ▼
                             [Google Gemini 2.5 Flash]
                             - Context-grounded synthesis
                             - Verified, explainable response
            ```
            """
        )

    with col_arch2:
        st.markdown("#### 🎯 Core Technical Highlights")
        st.markdown(
            """
            - **No Hallucination Design**: The LLM prompt explicitly restricts generation to retrieved manual excerpts. If cosine similarity is too low or context is missing, the system outputs an explicit unverified notice.
            - **Local Vector Persistence**: Embeddings are persisted in `./chroma_db`, avoiding costly re-computation on application restarts.
            - **Deduplication Engine**: Uses MD5 content hashing and filename matching to prevent redundant chunk insertion.
            - **Multimodal Document Ingestion**: Handles text files, tabular CSV troubleshooting sheets, and formatted PDF manuals.
            - **Safety-First Engineering**: Detects hazardous queries involving high-voltage capacitors, flammable refrigerants, or exposed live mains wiring, injecting immediate safety warnings.
            """
        )

    st.markdown("---")
    st.markdown("#### 🎓 Viva Voce & Technical Evaluation Questions")
    
    with st.expander("Q1: What is Retrieval-Augmented Generation (RAG) and why is it needed for appliance troubleshooting?"):
        st.write(
            "**Answer:** Traditional Large Language Models (LLMs) rely on parametric memory frozen at training time, "
            "which often results in hallucinations, generic non-specific answers, or out-of-date model information. "
            "RAG pairs an external non-parametric knowledge base (vector database with actual manufacturer manuals) "
            "with an LLM. When a user asks an appliance question, relevant manual excerpts are retrieved first and passed "
            "as grounding context to the LLM, ensuring verified, model-specific, and source-attributed answers."
        )

    with st.expander("Q2: How does ChromaDB perform semantic search and how are similarities computed?"):
        st.write(
            "**Answer:** ChromaDB uses Hierarchical Navigable Small World (HNSW) graphs to perform approximate nearest "
            "neighbor (ANN) search over 384-dimensional dense vectors generated by `all-MiniLM-L6-v2`. "
            "We configure the distance space to Cosine: Distance = 1 - CosineSimilarity. "
            "A query is projected into the same 384-dimensional latent semantic space, and the chunks closest in angular "
            "direction (highest cosine similarity) are returned."
        )

    with st.expander("Q3: Why is text chunking with overlap necessary instead of embedding full manuals?"):
        st.write(
            "**Answer:** Full manuals are thousands of tokens long. Embedding entire documents collapses fine-grained "
            "details (like specific error codes such as 'E2' or 'UE') into an average vector, severely degrading retrieval precision. "
            "By chunking text into 650-character segments with 120-character overlap, each chunk represents a cohesive "
            "troubleshooting step, and the overlap ensures sentences on boundaries are not cut in half."
        )

    with st.expander("Q4: How does the system handle safety hazards and questions outside the manuals?"):
        st.write(
            "**Answer:** 1) **Heuristic Safety Interceptor:** Regular expressions detect high-risk components (e.g. microwave "
            "high-voltage capacitors, freon refrigerant lines) and display critical warning banners. 2) **Unknown Question Fallback:** "
            "If retrieved chunks have insufficient similarity scores (< 0.25) or the context does not contain the answer, "
            "the prompt instructs the assistant to explicitly decline and advise consulting an authorized service technician."
        )
