import fitz #PyMuPDF
import sys
import os
from pathlib import Path
import re

import nltk
#nltk.download('all') # need to wrap this shit up so it doesn't run every time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    # remove special characters and digits (not sure if i want to do this given the texts)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #print(text)
    # tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in words]

    return ' '.join(words)

path = check_dir()
text = parse_dir(path)
cleaned_text = clean(text)
print(cleaned_text)

