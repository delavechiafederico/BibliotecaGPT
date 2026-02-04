from flask import Flask, render_template, request, jsonify
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from duckduckgo_search import DDGS
import os

app = Flask(__name__)

DATA_PATH = "data"
DB_PATH = "vectordb"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    return docs

def build_db():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

if not os.path.exists(DB_PATH):
    build_db()

db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature":0.3, "max_new_tokens":512},
    huggingfacehub_api_token=os.environ.get("HF_TOKEN")
)

def web_search(q):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(q, max_results=3):
            results.append(r["body"])
    return "\n".join(results)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]

    docs = db.similarity_search(question, k=3)
    local_context = "\n".join([d.page_content for d in docs])

    online = web_search(question)

    prompt = f"""
Usá este contexto para responder:

BIBLIOTECA:
{local_context}

ONLINE:
{online}

Pregunta:
{question}
"""

    answer = llm(prompt)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
