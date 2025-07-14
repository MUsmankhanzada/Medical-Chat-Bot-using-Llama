from src.helper import load_pdf, text_splitter, download_huggingface_embedings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from uuid import uuid4
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print(PINECONE_API_KEY)

extracted_text = load_pdf("D:\Medical-Chat-Bot-using-Llama\data")
text_chunks = text_splitter(extracted_text)

embeddings = download_huggingface_embedings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-bot")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

uuids = [str(uuid4()) for _ in range(len(text_chunks))]
vector_store.add_documents(documents=text_chunks, ids=uuids)




