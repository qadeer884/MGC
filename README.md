# MGC Developments – AI Engineer Task Submission

This repository contains my solution for the MGC Developments practical task.

It consists of four parts, each tackling a real business problem:

- **Part 1** – Document assistant (RAG)
- **Part 2** – Database schema & SQL queries
- **Part 3** – Lead scoring (ML baseline)
- **Part 4** – Web interface (Streamlit)

---

## 📁 Project Structure

```text
.
├── Part_1/
│   ├── Part_1.ipynb            # Notebook to build vectorstore
│   └── chroma_db/              # Persisted Chroma vector store
├── Part_2/
│   ├── schema.sql              # Minimal SQL schema
│   └── queries.sql             # Two required queries
├── Part_3/
│   ├── ML.ipynb                # Notebook for data cleaning & model training
│   └── lead_model.pkl          # Saved Logistic Regression model
├── app.py                      # Streamlit web app (Part 4)
├── rag_engine.py               # Loads Chroma, embeddings, LLM, provides ask()
├── requirements.txt
├── .env                        # Local environment variables 
├── .gitignore
├── README.md                   # This file

```

## ⚙️ Setup & Dependencies

### 1. Clone the repository

```bash
git clone https://github.com/qadeer884/MGC
cd <repo-folder>
```

### 2. Create a virtual environment (recommended)

**Linux/macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Set environment variable

Create a `.env` file in the project root:

```env
groq=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Do not commit the actual API key.

---

## 🚀 Running the Application (Part 4)

The web interface is built with **Streamlit** and provides both features.

```bash
streamlit run app.py
```

The application provides two main tabs:

- **📄 Document Assistant** – ask questions about MGC documents.
- **🎯 Lead Scoring** – enter lead details and get a conversion probability.

---

## 📄 Part 1 – Document Assistant (RAG)

The document assistant uses Retrieval-Augmented Generation (RAG) to answer questions from the provided MGC documents.

### Pipeline

```text
Documents
    ↓
Data loading
    ↓
MarkdownHeaderTextSplitter
    ↓
HuggingFace Embeddings
    ↓
Chroma Vector Store
    ↓
Similarity Retrieval (Top 4)
    ↓
Groq LLM
    ↓
Grounded Answer
```

### Components

- **Documents:** `01_brochure.md`, `02_price_list.md`, and `03_faq.md`
- **Splitter:** `MarkdownHeaderTextSplitter`
- **Embeddings:** `BAAI/bge-base-en-v1.5` via HuggingFace
- **Vector Store:** Chroma, persisted to `./Part_1/chroma_db`
- **LLM:** Groq using `openai/gpt-oss-120b`

The `ask()` function retrieves the top-4 relevant chunks and uses a strict prompt to ensure grounded responses.

### Grounding behavior

The assistant is instructed to:

- Answer only from the retrieved document context.
- Avoid unsupported claims or hallucinations.
- Respond with:

> "I don't have that, ask the marketing manager."

when the requested information is not available in the documents.

- Detect conflicting information and identify the relevant sources.

## 🗄️ Part 2 – Database Schema & Queries

### Schema (`schema.sql`)

The database contains a single `leads` table with the columns required from the CSV.

Key design decisions:

- `crm_record_hash` is defined as **UNIQUE** to prevent duplicate records.

### Queries (`queries.sql`)

Two required queries are included:

#### 1. Conversion rate by source

The query:

- Groups leads by `source`.
- Calculates `converted / total * 100`.
- Includes only sources with at least 200 leads.
- Orders results from highest conversion rate to lowest.

#### 2. Duplicate leads

The query:

- Groups records by `crm_record_hash`.
- Identifies hashes appearing more than once.
- Lists the associated `lead_id` values.

### Duplicate prevention

The `UNIQUE` constraint on `crm_record_hash` ensures that duplicate hashes cannot be inserted into the table. This matters because the raw CSV contains ~160 leads that were re-entered into the CRM under a second `lead_id` (e.g. `MGC-104183` and `MGC-104183-B`) while sharing the same `crm_record_hash` and identical field values — the constraint is what catches those at the database level, which is also why Part 3 uses this column to dedupe before training rather than dropping it outright.

---

## 📊 Part 3 – Lead Scoring (Honest Baseline)

This part provides a simple, leakage-free Logistic Regression baseline for predicting lead conversion.

### Data Preparation

- Duplicate leads were removed using `crm_record_hash` before splitting the data.
- `lead_id`, `crm_record_hash`, `created_at`, and `bedrooms` were excluded.
- `city` values were standardized and missing `area` values were set to `"Unknown"`.
- Numeric missing values were median-imputed using the training data only.
- `token_amount_received_pkr` was removed because it is a post-conversion booking deposit and causes target leakage.
- Remaining numeric features were scaled; categorical features were one-hot encoded.



## 🌐 Part 4 – Web Interface

The web interface is implemented using **Streamlit**.
