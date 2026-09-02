# app.py
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Part_3" / "lead_model.pkl"
load_dotenv(BASE_DIR / ".env")


@st.cache_resource(show_spinner=False)
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = Chroma(
        persist_directory=str(BASE_DIR / "Part_1" / "chroma_db"),
        collection_name="mgc_docs",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.environ["groq"],
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return retriever, llm


def ask(question):
    retriever, llm = load_rag()
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"{doc.page_content}\n[Source: {doc.metadata.get('file', 'unknown')} | "
        f"{doc.metadata.get('Section', 'General')}]"
        for doc in docs
    )
    prompt = f'''Answer ONLY from the context. If not found, say "I don't have that, ask the marketing manager."
If conflicting information appears, flag it and list both sources. Never invent.

Context:
{context}

Question: {question}

Answer:'''
    return llm.invoke(prompt).content, docs

# ---------- Load model (if it exists) ----------
try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    model_loaded = True
except:
    model_loaded = False

st.set_page_config(page_title="MGC Sales Assistant", layout="centered")
st.markdown(
    """
    <style>
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #080808;
        color: #ffffff;
        border-color: #555555;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: #ffffff;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #aaaaaa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🏢 MGC Sales Assistant")

# ---------- Tab 1: Document Assistant ----------
tab1, tab2 = st.tabs(["📄 Document Assistant", "🎯 Lead Scoring"])

with tab1:
    st.subheader("Ask a question about MGC Aurora Heights")
    question = st.text_input("Your question:", placeholder="e.g., What is the base price of a 2-bed in Block B?")
    
    if st.button("Ask", type="primary"):
        if question.strip():
            with st.spinner(""):
                answer, _ = ask(question)
            st.success("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

# ---------- Tab 2: Lead Scoring ----------
with tab2:
    st.subheader("Score a lead")
    
    if not model_loaded:
        st.warning("⚠️ Model not found. Train and save lead_model.pkl first.")
    else:
        with st.form("lead_form"):
            col1, col2 = st.columns(2)
            with col1:
                source = st.selectbox("Source", ["Facebook Ads", "Google Search", "Instagram", "Referral", "Walk-in", "Property Portal", "WhatsApp Campaign", "Expo Stall", "Billboard"], index=None, placeholder="Select source")
                city = st.text_input("City", placeholder="e.g., Islamabad")
                area = st.text_input("Area", placeholder="e.g., B-17")
                property_type = st.selectbox("Property Type", ["Apartment", "Plot", "Villa", "Commercial Shop", "Farmhouse", "Penthouse"], index=None, placeholder="Select property type")
                budget = st.number_input("Budget (PKR lac)", min_value=0.0, value=None, placeholder="Enter budget")
                first_response = st.number_input("First response (minutes)", min_value=0, value=None, placeholder="Enter minutes")
                calls = st.number_input("Calls made", min_value=0, value=None, placeholder="Enter call count")

            with col2:
                call_seconds = st.number_input("Total call seconds", min_value=0, value=None, placeholder="Enter seconds")
                whatsapp = st.number_input("WhatsApp replies", min_value=0, value=None, placeholder="Enter reply count")
                visits = st.number_input("Site visits", min_value=0, value=None, placeholder="Enter visit count")
                agent_exp = st.number_input("Agent experience (years)", min_value=0.0, value=None, step=0.5, placeholder="Enter years")
                overseas = st.selectbox("Overseas", [0, 1], index=None, placeholder="Select option")
                referred = st.selectbox("Referred by existing client", [0, 1], index=None, placeholder="Select option")
                financing = st.selectbox("Has financing approved", [0, 1], index=None, placeholder="Select option")
            
            submitted = st.form_submit_button("Score Lead")
        
        if submitted:
            lead_fields = [source, city, area, property_type, budget,
                           first_response, calls, call_seconds, whatsapp, visits,
                           agent_exp, overseas, referred, financing]
            if any(field is None or field == "" for field in lead_fields):
                st.warning("Please complete all lead fields before scoring.")
                st.stop()

            # Create input DataFrame with same columns as training
            input_data = pd.DataFrame([{
                "source": source,
                "city": city,
                "area": area,
                "property_type": property_type,
                "budget_pkr_lac": budget,
                "first_response_minutes": first_response,
                "calls_made": calls,
                "total_call_seconds": call_seconds,
                "whatsapp_replies": whatsapp,
                "site_visits": visits,
                "agent_experience_years": agent_exp,
                "is_overseas": overseas,
                "referred_by_existing_client": referred,
                "has_financing_approved": financing,
            }])
            
            # Predict
            prob = model.predict_proba(input_data)[0][1]
            st.metric("Conversion Probability", f"{prob:.1%}")
            
            if prob > 0.5:
                st.success("🔴 **High priority** – likely to convert")
            elif prob > 0.3:
                st.info("🟡 **Medium priority** – follow up")
            else:
                st.warning("🟢 **Low priority** – score for later")