import pandas as pd
import re
import hashlib

def clean_job_text(text):
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\-\*]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'page \d+ of \d+', '', text)
    text = re.sub(r'confidential', '', text)
    text = text.strip()
    return text

def hash_identifier(identifier):
    return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

def tokenize_and_hash_personal_info(text):
    def email_replacer(match):
        email = match.group(0)
        return f"[EMAIL_{hash_identifier(email)[:8]}]"
    text = re.sub(r'[\w\.-]+@[\w\.-]+', email_replacer, text)
    def phone_replacer(match):
        phone = match.group(0)
        return f"[PHONE_{hash_identifier(phone)[:8]}]"
    text = re.sub(r'\b\d{10}\b', phone_replacer, text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', phone_replacer, text)
    return text

# Columns to preprocess
columns_to_clean = [
    'Job Description', 'Benefits', 'skills', 'Responsibilities', 'Company Profile',
    'Contact Person', 'Contact'
]

chunksize = 1000
input_path = 'data/job_descriptions.csv'
output_path = 'data/job_descriptions_cleaned.csv'

header_written = False
for chunk in pd.read_csv(input_path, chunksize=chunksize):
    for col in columns_to_clean:
        if col in chunk.columns:
            chunk[col] = chunk[col].apply(clean_job_text).apply(tokenize_and_hash_personal_info)
    chunk.to_csv(output_path, mode='a', index=False, header=not header_written)
    header_written = True
print('Job descriptions preprocessed and saved to', output_path)
