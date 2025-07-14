from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_huggingface_embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-bot")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

model_path = r".\experiment\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

langchain_llm = LlamaCpp(
    model_path=model_path,  # Path to quantized GGUF model (e.g., llama-2-7b-chat.Q4_K_M.gguf)
    n_ctx=512,              # Reduced context window for faster inference
    n_threads=8,            # Utilize multiple CPU threads
    temperature=0.3,        # Keep generation deterministic
    top_p=0.9,
    repeat_penalty=1.1,
    use_mmap=True,
    verbose=True
)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # Top 2 similar documents
)


chain_type_kwargs = {"prompt": prompt}

# ✅ Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=langchain_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)