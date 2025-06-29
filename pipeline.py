from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Use the up-to-date import!
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize the embedding model (bge-base-en-v1.5)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Load only a small sample for fast testing
from utils.chunker import split_documents
from utils.loader import load_resumes_from_txt, load_job_descriptions_from_csv

resumes_path = r'C:\Users\dell\Downloads\res_job\data\preprocessed_resumes.txt'
job_descs_path = r'C:\Users\dell\Downloads\res_job\data\job_descriptions_cleaned.csv'

# Load a small sample for speed
resumes = load_resumes_from_txt(resumes_path)[:80] #
job_descs = load_job_descriptions_from_csv(job_descs_path)[:80]#

# Chunk the documents
chunked_resumes = split_documents(resumes)
chunked_job_descs = split_documents(job_descs)

# Add type metadata for dynamic filtering and enrich with more info if available
for doc in chunked_resumes:
    doc.metadata["type"] = "resume"
    doc.metadata["source"] = doc.metadata.get("source", "resume")
    # Optionally add more resume metadata if available
    doc.metadata["name"] = doc.metadata.get("name", "")  # e.g., candidate name
    doc.metadata["email"] = doc.metadata.get("email", "")
    doc.metadata["skills"] = doc.metadata.get("skills", "")
    doc.metadata["experience"] = doc.metadata.get("experience", "")
    doc.metadata["education"] = doc.metadata.get("education", "")
    doc.metadata["location"] = doc.metadata.get("location", "")
    doc.metadata["summary"] = doc.metadata.get("summary", "")
    

for doc in chunked_job_descs:
    doc.metadata["type"] = "job_desc"
    doc.metadata["source"] = doc.metadata.get("source", "job_desc")
    # Add more job description metadata if available
    doc.metadata["job_title"] = doc.metadata.get("job_title", "")
    doc.metadata["role"] = doc.metadata.get("role", "")
    doc.metadata["company"] = doc.metadata.get("company", "")
    doc.metadata["description"] = doc.metadata.get("description", "")
    doc.metadata["section"] = doc.metadata.get("section", "")
    doc.metadata["location"] = doc.metadata.get("location", "")
    doc.metadata["salary"] = doc.metadata.get("salary", "")
    doc.metadata["job_id"] = doc.metadata.get("job_id", "")
    doc.metadata["experience_required"] = doc.metadata.get("experience_required", "")

# Combine all chunks
all_chunks = chunked_resumes + chunked_job_descs

# Create a Chroma vector store and persist it
persist_directory = "chroma_store"
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="resume_job_store"
)

def get_retriever(doc_type=None, top_k=5):
    """Return a retriever for a specific document type or all documents."""
    search_kwargs = {"k": top_k}
    if doc_type:
        search_kwargs["filter"] = {"type": doc_type}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rerank_by_embedding(query, docs, embedding_model, top_k=5):
    query_emb = embedding_model.embed_query(query)
    doc_scores = []
    for doc in docs:
        doc_emb = doc.embedding if hasattr(doc, "embedding") else embedding_model.embed_query(doc.page_content)
        score = cosine_similarity(query_emb, doc_emb)
        doc_scores.append((score, doc))
    doc_scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in doc_scores[:top_k]]

# pipeline.py should only build and persist the vectorstore.
if __name__ == "__main__":
    print("Vectorstore built and ready.")
