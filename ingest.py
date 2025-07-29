from pypdf import PdfReader
import sys
import os

#n = len(sys.argv)
#path = ""
#
#if n > 1:
#    path = sys.argv[1]
#
#print(path)

# get path from user input
path = input("Enter path to pdf directory: ")
if os.path.isdir(path) == True:
    print(f"Path \"{path}\" exists")
else:
    print(f"ERROR: path {path} does not exist. Make sure the directory you entered is correct")

# check if .pdf files exist in the directory
if any(File.endswith(".pdf") for File in os.listdir(path)):
    print(f"{path} is a valid directory")
else:
    print(f"No .pdf files exist in {path}")

