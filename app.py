from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

# ----------------------------
# Load PDF and create vectorstore (do this once on startup)
# ----------------------------
PDF_PATH = r"C:\Users\Yasir\Desktop\Rag-2026\National Ai policy.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Hugging Face embeddings
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # top 5 chunks

# Groq LLM
llm = ChatGroq(
    api_key="",
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Tum ek helpful AI ho.
Neeche diye gaye context se hi jawab dena.
Agar jawab context me na ho to "Mujhe is document me jawab nahi mila" likho.

Context:
{context}

Question:
{question}

Answer (English):
""")

# RAG chain
rag_chain = (
    {
        "context": (
            lambda x: x["question"]
        )
        | retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="RAG PDF Chat API")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    response = rag_chain.invoke({"question": query.question})
    return {"answer": response.content}
