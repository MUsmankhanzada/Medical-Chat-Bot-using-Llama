from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob = "*.pdf",
                             loader_cls = PyPDFLoader)
    documents = loader.load()
    
    return documents


def text_splitter(extracted_text):
    text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    return text_split.split_documents(extracted_text)

def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings