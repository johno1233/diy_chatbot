import fitz #PyMuPDF
import sys
import os
from pathlib import Path

#n = len(sys.argv)
#path = ""
#
#if n > 1:
#    path = sys.argv[1]
#
#print(path)

# get path from user input
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

    # check if .pdf files exist in the directory
    #if any(File.endswith(".pdf") for File in os.listdir(path)):
    #    print(f"{path} is a valid directory")
    #    return path
    #else:
    #    print(f"No .pdf files exist in {path}")

def parse_dir(files):
    extracted_text = []
    for file in files:
        with fitz.open(file) as doc:
            full_text = ""
            print(f"Processing file: {file}")
            for page in doc:
                full_text += page.get_text()
            extracted_text.append(full_text)
    return extracted_text

path = check_dir()
text = parse_dir(path)
print(text)
