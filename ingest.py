from pypdf import PdfReader
import sys

n = len(sys.argv)
path = ""

if n > 1:
    path = sys.argv[1]

print(path)

