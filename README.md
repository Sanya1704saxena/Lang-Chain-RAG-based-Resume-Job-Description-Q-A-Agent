# Lang-Chain-RAG-based-Resume-Job-Description-Q-A-Agent
This project was built to enable recruiters &amp; job seekers to semantically search, filter, and analyze large volumes of resumes and jobs . It goes far beyond keyword search it understands intent, extracts relevant content, and uses local LLM reasoning to answer questions like:“What experience is required for digital marketing roles in Bangalore?” 

#Key Features & Why They Matter
✅Document Chunking & Metadata Tagging : 
We use `RecursiveCharacterTextSplitter` + LangChain `Document` objects to split large resumes and JDs into semantically relevant chunks. Metadata like job ID, title, location, and skills are embedded for precise filtering  a crucial step in structuring unstructured data.
✅ Embedding with BGE or MiniLM 
We leveraged sentence-transformers like `BAAI/bge-base-en-v1.5` to embed these chunks into vector space. These embeddings capture semantic similarity , letting us match resumes and roles even when the keywords don’t align 1:1.
✅ Chroma for Vector Store 
Why Chroma?  
It supports fast similarity search, metadata filtering, local persistence, and easy dev experience making it ideal over alternatives like FAISS or Qdrant for this use case. Chroma's filterable collections allowed us to isolate resume vs JD chunks efficiently during retrieval.
✅ Cosine Similarity & Reranking
Initial results are reranked using cosine similarity of the query and candidate vectors,This ensures that documents most semantically aligned with the query rise to the top crucial for high-accuracy matching in long-form text.
✅ Local LLM Reasoning via llama.cpp + GGUF  
No cloud calls! All responses are generated using a locally hosted `Mistral-7B` GGUF model via `llama-cpp-python`. This keeps the system fast, private, and portable.
✅ Metadata Filtering + Search Interface 
You can filter for resumes mentioning Django, or find job descriptions requiring “3+ years experience” in Delhi all thanks to rich metadata indexing.

#Setup & Installation
Requirements:
- Python 3.10+
- CMake + g++ (for compiling llama.cpp)
- ~8GB+ RAM for local inference
Download Models:
- Embedding: `BAAI/bge-base-en-v1.5`
- LLM: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`

# Tech Stack
- `LangChain` (chunking, RAG)
- `Chroma` (vector DB)
- `sentence-transformers` (embeddings)
- `llama-cpp-python` (local LLM inference)
This is still evolving next steps include job–resume scoring, Streamlit UI.





