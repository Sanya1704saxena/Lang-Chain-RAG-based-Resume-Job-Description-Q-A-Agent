from langchain_community.document_loaders import Docx2txtLoader
from pathlib import Path
import pandas as pd
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_resumes_from_txt(txt_path):
    print(f"Loading preprocessed resumes from: {txt_path}")
    docs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split each resume by double newlines (assuming this is how they're separated)
    resumes = content.strip().split('\n\n')
    for idx, resume in enumerate(resumes):
        if resume.strip():
            doc = type('Doc', (), {})()
            doc.page_content = resume
            doc.metadata = {"type": "resume", "source": f"preprocessed_{idx}", "path": txt_path}
            docs.append(doc)
    return docs

def load_job_descriptions_from_csv(csv_path, text_column='Job Description'):
    print(f"Loading job descriptions from: {csv_path}")
    df = pd.read_csv(r'C:\Users\dell\Downloads\res_job\data\job_descriptions_cleaned.csv')
    docs = []
    for idx, row in df.iterrows():
        text = str(row.get(text_column, ''))
        if text.strip():
            doc = type('Doc', (), {})()
            doc.page_content = text
            doc.metadata = {"type": "job_desc", "source": f"row_{idx}", "path": csv_path}
            docs.append(doc)
    return docs

# Example usage:
resumes_path = r'C:\Users\dell\Downloads\res_job\data\preprocessed_resumes.txt'
job_descs_path = r'C:\Users\dell\Downloads\res_job\data\job_descriptions_cleaned.csv'
resumes = load_resumes_from_txt(resumes_path)
job_descs = load_job_descriptions_from_csv(job_descs_path)

