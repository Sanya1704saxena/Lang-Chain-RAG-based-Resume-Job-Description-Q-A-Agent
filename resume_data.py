from langchain_community.document_loaders import Docx2txtLoader
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
# Uncomment the following line if you want to use sentence-transformers
# from sentence_transformers import SentenceTransformer

folder_path = r"C:\Users\dell\Downloads\res_job\data\Resumes"

def clean_resume_text(text):
    # Remove bullet symbols (•, -, *, etc.)
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\-\*]', '', text)
    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text)
    # Lowercase all text
    text = text.lower()
    # Remove headers/footers like "Page 1 of 2" or "Confidential Resume"
    text = re.sub(r'page \d+ of \d+', '', text)
    text = re.sub(r'confidential resume', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def load_resumes_from_folder(folder_path):
    all_docs = []
    for file in Path(folder_path).glob("*.docx"):
        loader = Docx2txtLoader(str(file))
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_resume_text(doc.page_content)
            doc.metadata["type"] = "resume"
            doc.metadata["source"] = file.name
        all_docs.extend(docs)
    return all_docs


section_keywords = [
    "Education", "Work Experience", "Professional Experience", 
    "Projects", "Skills", "Certifications", "Achievements", "Summary"
]

def segment_resume_sections(resume_text, section_keywords=None):
    if section_keywords is None:
        section_keywords = [
            "Education", "Work Experience", "Professional Experience",
            "Projects", "Skills", "Certifications", "Achievements", "Summary"
        ]
    # Build regex pattern for section headers
    pattern = r"(" + "|".join([re.escape(k) for k in section_keywords]) + r")"
    # Split text into sections
    splits = re.split(pattern, resume_text, flags=re.IGNORECASE)
    # Group into (section, content) pairs
    sections = {}
    current_section = None
    for part in splits:
        part = part.strip()
        if not part:
            continue
        if part.lower() in [k.lower() for k in section_keywords]:
            current_section = part.title()
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += part + " "
        else:
            # Text before any section header
            sections.setdefault("Other", "")
            sections["Other"] += part + " "
    # Strip whitespace from all section contents
    for k in sections:
        sections[k] = sections[k].strip()
    return sections

docs = load_resumes_from_folder(folder_path)
# Example usage:
for doc in docs:
    sections = segment_resume_sections(doc.page_content, section_keywords)
    print(sections)

# --- TF-IDF Vectorization ---
def tfidf_vectorize(docs):
    texts = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# Example usage:
tfidf_matrix, tfidf_vectorizer = tfidf_vectorize(docs)
print('TF-IDF shape:', tfidf_matrix.shape)

# --- Embedding Vectorization (Sentence Transformers) ---
def embed_vectorize(docs, model_name='all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, model

# Example usage:
# embeddings, embed_model = embed_vectorize(docs)
# print('Embeddings shape:', embeddings.shape)

def hash_identifier(identifier):
    return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

def tokenize_and_hash_personal_info(text):
    # Hash and replace emails
    def email_replacer(match):
        email = match.group(0)
        return f"[EMAIL_{hash_identifier(email)[:8]}]"
    text = re.sub(r'[\w\.-]+@[\w\.-]+', email_replacer, text)
    # Hash and replace phone numbers
    def phone_replacer(match):
        phone = match.group(0)
        return f"[PHONE_{hash_identifier(phone)[:8]}]"
    text = re.sub(r'\b\d{10}\b', phone_replacer, text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', phone_replacer, text)
    # Hash and replace names (optional, requires NER)
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                hashed = f"[NAME_{hash_identifier(ent.text)[:8]}]"
                text = text.replace(ent.text, hashed)
    except ImportError:
        pass
    except OSError:
        pass
    return text

# Example usage:
# for doc in docs:
#     doc.page_content = tokenize_and_hash_personal_info(doc.page_content)

def save_preprocessed_text(docs, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(f"Source: {doc.metadata['source']}\n")
            f.write(doc.page_content + "\n\n")

# Example usage:
# for doc in docs:
#     doc.page_content = tokenize_and_hash_personal_info(doc.page_content)
save_preprocessed_text(docs, 'preprocessed_resumes.txt')
print('Preprocessed text saved to preprocessed_resumes.txt')

