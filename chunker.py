import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.loader import load_resumes_from_txt, load_job_descriptions_from_csv
import pandas as pd
from langchain.schema import Document

def split_documents(documents, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def split_job_desc_by_sections(jd_text):
    """
    Split job description into sections based on common headers.
    Returns a list of (section_name, section_text) tuples.
    """
    section_headers = [
        r"Responsibilities?:",
        r"Requirements?:",
        r"Qualifications?:",
        r"Skills?:",
        r"About (the )?Company:",
        r"Benefits?:",
        r"Job Description:",
        r"Role:",
        r"Profile:",
        r"Experience:",
        r"Education:"
    ]
    pattern = "|".join(section_headers)
    splits = [m for m in re.finditer(pattern, jd_text,re.IGNORECASE)]
    sections = []
    if not splits:
        return [("General", jd_text.strip())]
    for i, match in enumerate(splits):
        section_name = match.group(0).replace(":", "").strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(jd_text)
        section_text = jd_text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))
    return sections

# --- Metadata extraction helpers ---
def extract_experience(text):
    match = re.search(r"(\d+\+?)\s+(years|yrs).{0,10}experience", text, re.I)
    return match.group(0) if match else ""

def extract_location(text):
    match = re.search(r"(location\s*[:\-]?\s*)([^\n,;]+)", text, re.I)
    return match.group(2).strip() if match else ""

def extract_salary(text):
    match = re.search(r"(salary\s*[:\-]?\s*)([^\n,;]+)", text, re.I)
    return match.group(2).strip() if match else ""

# Example usage:
if __name__ == "__main__":
    resumes_path = r'C:\Users\dell\Downloads\res_job\data\preprocessed_resumes.txt'
    job_descs_path = r'C:\Users\dell\Downloads\res_job\data\job_descriptions_cleaned.csv'

    resumes = load_resumes_from_txt(resumes_path)

    # Load job descriptions as DataFrame
    df = pd.read_csv(job_descs_path)
    # Convert each JD to LangChain Documents split by section, with metadata
    jd_documents = []
    for idx, row in df.iterrows():
        jd_text = str(row.get("Job Description", "")).strip()
        job_title = str(row.get("Job Title", "")).strip() if "Job Title" in row else ""
        if not jd_text:
            continue
        sections = split_job_desc_by_sections(jd_text)
        for section_name, section_text in sections:
            metadata = {
                "type": "job_desc",
                "job_id": row.get("Job ID", f"JD_{idx}"),
                "source": f"JD_{idx}",
                "section": section_name,
                "job_title": job_title,
                "path": job_descs_path,
                "experience_required": extract_experience(section_text),
                "location": extract_location(section_text),
                "salary": extract_salary(section_text)
            }
            jd_documents.append(
                Document(
                    page_content=section_text,
                    metadata=metadata
                )
            )

    chunked_resumes = split_documents(resumes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    # Preserve section metadata: only split long sections, keep short ones whole
    jd_chunks = []
    for doc in jd_documents:
        # Only split long sections
        if len(doc.page_content) > 512:
            jd_chunks.extend(splitter.split_documents([doc]))
        else:
            jd_chunks.append(doc)

    print(f"Chunked {len(resumes)} resumes into {len(chunked_resumes)} chunks.")
    print(f"Loaded {len(jd_documents)} job description sections.")
    print(f"Chunked job descriptions into {len(jd_chunks)} chunks.")
