import fitz #PyMuPDF
import sys
import os
from pathlib import Path
import re

import nltk
nltk.download('punkt') # need to wrap this shit up so it doesn't run every time


# get path from user input, check if path is valid, create a list of all pdf files in path
def check_dir():
    files = []
    path = input("Enter path to pdf directory: ")
    if os.path.isdir(path) == True:
        dir = Path(path)
        print(f"Path \"{path}\" exists")
        files = list(dir.glob("*.pdf"))
        return files
    else:
        print(f"ERROR: path {path} does not exist. Make sure the directory you entered is correct")

# for each file in files, extract all text and save to "extracted_text" 
def parse_dir(files):
    extracted_text = ""
    for file in files:
        with fitz.open(file) as doc:
            full_text = ""
            print(f"Processing file: {file}")
            for page in doc:
                full_text += page.get_text()
            extracted_text += full_text
    return extracted_text

# iterate over text and clean it for use with NLP
def clean(text):
    # Normalize encoding
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Remove special characters and excessive whitespace
    text = re.sub(r'\x00-\x1F\x7F-\x9f', '', text) # Removes control characters
    text = re.sub(r'\s+', ' ', text)               # Replace multiple spaces witn a single space
    text = text.strip()

    # Remove common PDF artifacts (page numbers, headers, formatting)
    text = re.sub(r'\bPage \d+\b', '', text) # Remove "Page X"
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE) # Remove lone numbers like page numbers
    return text

def segment_text(text):
    sentences = nltk.sent_tokenize(text)
    # We can group sentences into paragraphs or fixed-length sequences
    sequences = []
    current_sequence = []
    max_sequence_length = 512 # Adjust based on model requirements
    for sentence in sentences:
        current_sequence.append(sentence)
        if len(" ".join(current_sequence)) >= max_sequence_length:
            sequences.append(" ".join(current_sequence))
            current_sequence = []
    if current_sequence:
        sequences.append(" ".join(current_sequence))
    return sequences

def filter(sequences):
    # Remove short or duplicate sequences
    filtered = []
    seen = set()
    for seq in sequences:
        if len(seq) > 10 and seq not in seen: # Minimum length and not duplicate
            filtered.append(seq)
            seen.add(seq)
    return filtered

# these function calls are for testing only: comment out when running program as they will be called by main    
path = check_dir()
text = parse_dir(path)
cleaned_text = clean(text)
sequences = segment_text(cleaned_text)
cleaned_sequences = filter(sequences)
print(cleaned_sequences)

