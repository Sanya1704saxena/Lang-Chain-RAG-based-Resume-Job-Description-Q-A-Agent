# Lang-Chain-RAG-based-Resume-Job-Description-Q-A-Agent
This project was built to enable recruiters &amp; job seekers to semantically search, filter, and analyze large volumes of resumes and jobs . It goes far beyond keyword search it understands intent, extracts relevant content, and uses local LLM reasoning to answer questions like:“What experience is required for digital marketing roles in Bangalore?” 
# Project Description
This project enables recruiters and job seekers to semantically search, filter, and analyze resumes and job descriptions.  
It leverages chunked document storage, vector embeddings, and a local LLM (Llama.cpp) to answer questions like:
- "What experience is required for digital marketing roles in Bangalore?"
- "Which resumes mention experience with Flask?"
# Features

- ✅ Intelligent chunking of resumes and job descriptions (by section)
- ✅ Embedding with BGE (BAAI/bge-base-en-v1.5)
- ✅ Metadata extraction (experience, location, salary, etc.)
- ✅ Fast vector search with ChromaDB
- ✅ Reranking with cosine similarity for precise results
- ✅ Retrieval-augmented generation via local LLM (Llama.cpp)
- ✅ Streamlit UI for interactive Q&A and filtering



