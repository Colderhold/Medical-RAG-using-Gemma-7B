import os
import streamlit as st
import json
import time
import logging
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Configure logging
logging.basicConfig(filename="rag/logs.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Streamlit App
st.set_page_config(page_title="MAFLONG - PDF Processor", layout="wide")
st.title("📄 Upload PDF for Processing")

# Initialize Qdrant Client (Avoid using config that causes validation errors)
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url)

# Define collection name
COLLECTION_NAME = "medical_research_papers_db"

# Check if collection exists; if not, create it
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Adjust `size` based on the embedding model
    )

# Load Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload your PDF file", type=["pdf"], accept_multiple_files=False)

# Process PDF if uploaded
if uploaded_file:
    try:
        # Save uploaded file
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

        # Load PDF using DirectoryLoader
        loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
        documents = loader.load()

        # Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        texts = text_splitter.split_documents(documents)

        # Show preview of extracted text
        st.subheader("🔍 Sample Extracted Text")
        st.text(texts[0].page_content if texts else "No content extracted.")

        # Generate embeddings and insert into Qdrant manually
        vectors = []
        for i, doc in enumerate(texts):
            embedding_vector = embeddings.embed_query(doc.page_content)  # Generate embedding
            vectors.append((i, embedding_vector, {"text": doc.page_content}))

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=vectors
        )

        st.success("✅ PDF processed and stored successfully in Vector Database!")

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logging.error(f"Error processing PDF: {str(e)}")
