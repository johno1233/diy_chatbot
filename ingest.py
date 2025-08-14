import fitz #PyMuPDF
import sys
import os
from pathlib import Path
import re
from transformers import AutoTokenizer
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

console = Console()

import nltk
nltk.download('punkt', quiet=True)


# get path from user input, check if path is valid, create a list of all pdf files in path
def check_dir(pdf_dir=None):
    if pdf_dir is None:
        pdf_dir = input("Enter path to pdf directory: ")
    
    if os.path.isdir(pdf_dir):
        dir_path = Path(pdf_dir)
        files = list(dir_path.glob("*.pdf"))
        if not files:
            console.print(f"[bold red]No PDF files found in {pdf_dir}[/bold red]")
        return files
    else:
        console.print(f"[bold red]ERROR: path {pdf_dir} does not exist. [/bold red]")
        return []



# for each file in files, extract all text and save to "extracted_text" 
def parse_dir(files):
    extracted_text = ""
    with Progress(
        SpinnerColumn(),
        "[progress.desctiption]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing PDFs...", total=len(files))
        for file in files:
            with fitz.open(file) as doc:
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                extracted_text += full_text
            progress.update(task, description=f"Processed: {file.name}")
            progress.advance(task)
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

def normalize(sequences):
    normalized = []
    for seq in sequences:
        #seq = seq.lower()
        seq = re.sub(r'\b{A-Za-z0-9._%+-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', seq) # Anonymize emails
        normalized.append(seq)
    return normalized

# these function calls are for testing only: comment out when running program as they will be called by main    

def nomnom(pdf_dir=None):
    files = check_dir(pdf_dir)
    if not files:
        return

    text = parse_dir(files)
    cleaned_text = clean(text)
    sequences = segment_text(cleaned_text)
    cleaned_sequences = filter(sequences)
    normalized_sequences = normalize(cleaned_sequences)

    with open("training_data.txt", "w", encoding="utf-8") as f:
        for seq in normalized_sequences:
            f.write(seq + "\n")
    
    console.print(f"[bold green]Saved training daat from {len(files)} PDFs to training_data.txt[/bold green]")
