import streamlit as st
from rag_model import rag_chain

st.title("📊 Time-Series RAG Forecasting Assistant")

query = st.text_input("Ask your question:")

if query:
    response = rag_chain(query)
    st.write("### 🤖 Answer:")
    st.write(response)