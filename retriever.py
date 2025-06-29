import os
import numpy as np
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

# === Load persisted vectorstore ===
persist_directory = "chroma_store"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

def get_retriever(doc_type=None, top_k=5, extra_filter=None):
    """Return a retriever for a specific document type or all documents."""
    search_kwargs = {"k": top_k}
    if doc_type:
        search_kwargs["filter"] = {"type": doc_type}
    if extra_filter:
        search_kwargs.setdefault("filter", {}).update(extra_filter)
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

# === Load the LLM locally using llama-cpp and a GGUF model ===
llm = LlamaCpp(
    model_path=r"C:\Users\dell\Downloads\res_job\models\mistral-7b-instruct-v0.2.Q2_K.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048
)

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

def format_metadata(meta, doc_type):
    if doc_type == "job_desc":
        return (
            f"[{meta.get('type', 'unknown')}] "
            f"[Title: {meta.get('job_title', 'N/A')}] "
            f"[Section: {meta.get('section', 'N/A')}] "
            f"[Location: {meta.get('location', 'N/A')}] "
            f"[Experience Required: {meta.get('experience_required', 'N/A')}] "
            f"[Salary: {meta.get('salary', 'N/A')}] "
            f"[Job ID: {meta.get('job_id', 'N/A')}] "
            f"[Role: {meta.get('role', 'N/A')}] "
            f"[Company: {meta.get('company', 'N/A')}] "
            f"[Description: {meta.get('description', 'N/A')}] "
            f"[Summary: {meta.get('summary', 'N/A')}]"
        )
    elif doc_type == "resume":
        return (
            f"[{meta.get('type', 'unknown')}] "
            f"[Name: {meta.get('name', 'N/A')}] "
            f"[Email: {meta.get('email', 'N/A')}] "
            f"[Skills: {meta.get('skills', 'N/A')}] "
            f"[Experience: {meta.get('experience', 'N/A')}] "
            f"[Education: {meta.get('education', 'N/A')}] "
            f"[Location: {meta.get('location', 'N/A')}] "
        )
    else:
        return f"[{meta.get('type', 'unknown')}] [Source: {meta.get('source', 'unknown')}]"

def query_match_agent(question: str, doc_type=None, top_k=10, section_filter=None):
    retriever = get_retriever(doc_type, top_k)
    results = retriever.invoke(question)
    reranked_results = rerank_by_embedding(question, results, embedding_model, top_k=top_k)

    # Optional: filter by section if specified (for job_desc)
    if section_filter and doc_type == "job_desc":
        reranked_results = [
            doc for doc in reranked_results
            if doc.metadata.get("section", "").lower() in section_filter
        ]

    # Build context for LLM
    context = "\n\n".join(
        [
            f"Title: {doc.metadata.get('job_title', 'N/A')}\n"
            f"Section: {doc.metadata.get('section', 'N/A')}\n"
            f"Experience Required: {doc.metadata.get('experience_required', 'N/A')}\n"
            f"Location: {doc.metadata.get('location', 'N/A')}\n"
            f"Salary: {doc.metadata.get('salary', 'N/A')}\n"
            f"Content: {doc.page_content}"
            for doc in reranked_results
        ]
    )
    full_prompt = (
        "Answer the question ONLY using the context below. "
        "If the answer is not in the context, say 'Not found in provided job descriptions.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Helpful Answer:"
    )
    answer = llm(full_prompt)

    # Prepare results for UI or CLI
    formatted_chunks = [
        format_metadata(doc.metadata, doc_type) + "\n" + doc.page_content[:300] + "\n---"
        for doc in reranked_results
    ]
    formatted_sources = [
        format_metadata(doc.metadata, doc_type)
        for doc in reranked_results
    ]

    # Print for CLI usage
    print("\n🔎 Top Retrieved Chunks:")
    for chunk in formatted_chunks:
        print(chunk)
    print("\n🧠 Answer:\n", answer)
    print("\n📄 Sources:")
    for src in formatted_sources:
        print(" -", src)

    # Return for UI usage
    return {
        "answer": answer,
        "context": context,
        "chunks": formatted_chunks,
        "sources": formatted_sources
    }

# === Example usage ===
if __name__ == "__main__":
    print("=== Querying Job Descriptions ===")
    query =" mention jobs in Macao SAR? ."
    query_match_agent(query, doc_type="job_desc", section_filter=["experience", "requirements", "qualification", "responsibilities", "skills", "location", "salary","job_id","role", "company", "description", "summary"])

    print("\n=== Querying Resumes ===")
    query = "Which resumes mention experience with web ui ?"
    query_match_agent(query, doc_type="resume")

