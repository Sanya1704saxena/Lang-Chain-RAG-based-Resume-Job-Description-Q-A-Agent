import streamlit as st
from retriever import query_match_agent 

st.title("Resume & Job Description Q&A")

# Sidebar filter for document type
doc_type = st.sidebar.selectbox(
    "Select document type to search:",
    options=["job_desc", "resume"],
    format_func=lambda x: "Job Description" if x == "job_desc" else "Resume"
)

# Optional: Section filter for job descriptions
section_filter = None
if doc_type == "job_desc":
    section_options = [
        "experience", "requirements", "qualification", "responsibilities", "skills", "location", "salary","job_id","role", "company", "job_title"
    ]
    selected_sections = st.sidebar.multiselect(
        "Filter by section :", section_options
    )
    section_filter = [s.lower() for s in selected_sections] if selected_sections else None

st.write("Ask a question about the selected document type:")

question = st.text_input("Your question:")

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching and generating answer..."):
            result = query_match_agent(
                question, doc_type=doc_type, section_filter=section_filter
            )
            st.markdown("### 🧠 Answer")
            st.write(result["answer"])
            st.markdown("### 🔎 Top Retrieved Chunks")
            for chunk in result["chunks"]:
                st.info(chunk)
            st.markdown("### 📄 Sources")
            for src in result["sources"]:
                st.code(src)

          
    else:
        st.warning("Please enter a question.")