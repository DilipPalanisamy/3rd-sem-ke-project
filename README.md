# 🛠️ AI-Powered Home Appliance Troubleshooting & User Guide Assistant (RAG)

An intelligent, context-grounded **Retrieval-Augmented Generation (RAG)** assistant that helps users diagnose and troubleshoot household appliance problems (washing machines, refrigerators, air conditioners, microwaves, dishwashers) using technical manuals, error code sheets, and maintenance guides.

---

## 📌 Project Objectives

1. **Grounded Question Answering:** Prevent LLM hallucinations by retrieving exact technical excerpts from manuals before generating answers.
2. **Multimodal Document Processing:** Ingest user manuals and troubleshooting tables across **PDF**, **TXT**, and **CSV** formats.
3. **Local Vector Database Persistence:** Store 384-dimensional dense semantic embeddings using **ChromaDB** with persistent local disk storage.
4. **Dynamic Knowledge Updates:** Allow users to upload new manuals on-the-fly without restarting or re-indexing the entire database.
5. **Source Attribution & Explainability:** Display every retrieved document chunk, similarity score, and source filename alongside the answer.
6. **Safety & Hazard Warnings:** Automatically warn users about dangerous high-voltage components (microwaves), pressurized refrigerants (ACs/fridges), and live mains power, advising when to call a qualified technician.
7. **Graceful Handling of Unknowns:** If an issue is not covered in the knowledge repository, explicitly inform the user instead of guessing.

---

## 🏗️ System Architecture & Workflow

```
               [ User Input / Query ]
                         │
                         ▼
             [ Dense Query Embedding ]
        (Sentence Transformers: all-MiniLM-L6-v2)
                         │
                         ▼
        [ ChromaDB HNSW Vector Repository ]
        - Persistent local storage: ./chroma_db
        - Cosine distance semantic similarity
        - Appliance category filtering
                         │
                         ▼
             [ Top-K Relevant Chunks ]
        - Source manuals + page/row metadata
        - Cosine similarity scores
                         │
                         ▼
            [ Prompt Augmentation Engine ]
        - Safety hazard rules verification
        - Strict factual constraint ("Answer ONLY using context")
                         │
                         ▼
             [ Google Gemini API LLM ]
         (gemini-2.5-flash / gemini-1.5-flash)
                         │
                         ▼
          [ Structured & Explainable Answer ]
        1. Appliance & Issue Detected
        2. Probable Cause(s)
        3. Step-by-Step Solution
        4. Safety Warnings & Technician Callouts
        5. Expandable Source Citations with Excerpts
```

---

## 💻 Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| **Programming Language** | Python 3.12 | Standard for AI/ML and RAG systems |
| **Vector Database** | ChromaDB | Open-source, lightweight, native local persistence, HNSW indexing |
| **Embedding Model** | Sentence Transformers (`all-MiniLM-L6-v2`) | High-speed 384-dimensional dense semantic representations |
| **LLM Provider** | Google Gemini API (`gemini-2.5-flash`) | Fast, accurate, cost-effective context processing via `google-genai` |
| **Frontend UI** | Streamlit | Clean, interactive web interface with chat history, uploaders, and metrics |
| **Document Processing** | PyPDF, Pandas, ReportLab | Reliable parsing of PDF pages, CSV datasets, and TXT manuals |

---

## 📂 Project Structure

```
3rd-sem-ke-project/
│
├── app.py                      # Streamlit web application & user interface
├── rag_engine.py               # Core RAG engine: chunking, embedding, ChromaDB & Gemini API
├── prepare_sample_data.py      # Script generating sample appliance manuals (PDF, TXT, CSV)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for Gemini API key configuration
├── README.md                   # Complete documentation and viva guide
│
├── data/
│   └── sample_manuals/         # Pre-populated appliance documentation
│       ├── washing_machine_guide.txt      # Washing machine errors & maintenance
│       ├── refrigerator_manual.txt        # Fridge cooling, defrost & coils
│       ├── air_conditioner_guide.txt      # AC warm air, filters & leaks
│       ├── microwave_troubleshooting.csv  # Microwave sparking, keypad & magnetron
│       └── dishwasher_user_guide.pdf      # Dishwasher E15 flood sensor & spray arms
│
└── chroma_db/                  # Local persistent ChromaDB vector repository (auto-created)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone / Open the Workspace
```bash
cd "c:\Users\DILIP\OneDrive\Desktop\3rd-sem-ke-project"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key (Optional but Recommended)
Create a `.env` file from `.env.example`:
```bash
copy .env.example .env
```
Open `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Tip:** You can obtain a free API key from [Google AI Studio](https://aistudio.google.com/). Alternatively, you can enter your API key directly in the Streamlit sidebar input during runtime.
> If no API key is entered, the app runs in **Offline Semantic Mode**, performing vector retrieval and displaying grounded answers and exact source citations without needing an active internet LLM connection.

### 5. Generate Sample Manuals (if not already generated)
```bash
python prepare_sample_data.py
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 🧪 Testing Sample Troubleshooting Questions

Try these questions to observe specific capabilities of the RAG pipeline:

| Test Case | Sample Query | Expected RAG Behavior |
|---|---|---|
| **Washer Drain Issue** | *"Why is my washing machine showing error E2 and not draining?"* | Retrieves `washing_machine_guide.txt`; explains drain hose height, debris filter cleaning, hot water caution. |
| **AC Blowing Warm Air** | *"My air conditioner is running but blowing warm air into the room."* | Retrieves `air_conditioner_guide.txt`; covers dirty filters, outdoor condenser clearance, ice on copper pipe. |
| **Microwave Sparking** | *"What causes sparking inside my microwave cavity?"* | Retrieves `microwave_troubleshooting.csv`; identifies metal objects and scorched mica waveguide cover. |
| **Fridge Water Pooling** | *"Why is water pooling under the crisper drawers in my refrigerator?"* | Retrieves `refrigerator_manual.txt`; details defrost drain hole blockage and warm water flushing. |
| **Dishwasher PDF Guide** | *"What does error E15 on my dishwasher mean?"* | Retrieves `dishwasher_user_guide.pdf`; explains AquaStop flood float sensor and 45-degree tilt procedure. |
| **Safety Interceptor** | *"How do I discharge the high-voltage capacitor inside a microwave?"* | Triggers high-priority red danger banner advising against lethal DIY electrical repair. |
| **Unknown Query Test** | *"How do I fix a Boeing 777 jet engine compressor?"* | Detects low similarity / missing context and explicitly declines to invent an answer. |

---

## 🎓 Viva Voce & Academic Defense Guide

### Q1: What is the primary difference between RAG and Fine-Tuning?
- **Fine-Tuning:** Updates the internal model weights of an LLM using domain-specific datasets. It is computationally expensive, prone to catastrophic forgetting, and cannot easily cite exact sources.
- **RAG (Retrieval-Augmented Generation):** Keeps the LLM frozen and supplements it with an external, dynamic vector database. It allows instant document updates without re-training and guarantees factual grounding with source attribution.

### Q2: Why is ChromaDB chosen as the vector database?
- ChromaDB is an open-source, embedded vector database designed specifically for AI applications.
- It operates in-process with Python, requiring no external server or cloud account.
- It provides native local disk persistence (`PersistentClient`), HNSW (Hierarchical Navigable Small World) indexing, and flexible distance metrics (cosine, L2, inner product).

### Q3: What is the role of the Embedding Model (`all-MiniLM-L6-v2`)?
- It maps variable-length text sentences into a fixed 384-dimensional dense vector space.
- Semantically similar texts (e.g., *"drum won't turn"* and *"washing machine not spinning"*) are mapped close together in vector space, allowing semantic retrieval even when exact keywords differ.

### Q4: How is duplicate document ingestion prevented?
- When a document is processed, an MD5 hash is calculated over its normalized text content.
- Existing records in ChromaDB are checked via metadata filters (`where={"source": filename}`).
- Users can choose to skip duplicates or toggle overwrite to replace outdated manual editions cleanly.

---

## 📜 License
Developed as an academic mini-project for Knowledge Engineering / AI-RAG demonstration. Free for educational and personal use.
