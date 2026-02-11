import streamlit as st
import requests

st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG PDF Chat with Groq + FAISS")

st.markdown("Ask questions from the National AI Policy PDF.")

# User input
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        # Call FastAPI backend
        try:
            response = requests.post("http://localhost:8000/ask", json={"question": question})
            if response.status_code == 200:
                answer = response.json()["answer"]
                st.success(answer)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")
