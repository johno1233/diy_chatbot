# DIY Chatbot

## Current Functionality:
- Asks for directory, ideally containing PDF files
    - Can handle if the directory is invalid/exists and whether or not it has PDFs in it
- If the directory is valid, it grabs all docs with .pdf and puts it into a list
- Parses the files and pulls all valid text from them
- Tokenizes and cleans the text for use with NLP 
- Takes a look a system specs and recommends a model to run based on what resources you have
- 2 variations: one is CLI the other uses streamlit for a nice-ish GUI
- Each Variation (CLI and GUI) use "hybrid" mode:
    - users can either have the model interact with the PDF files provided at the start or
    - just use the model's base knowledge
        - This functionality is iffy on the GUI version (will update in the future)
- Removed GUI functionality/streamlit (it was a pain) -- now completely cli
    - Made it look pretty with Rich CLI library
- Added webscraping functionality to fall back on when a query is made that isn't covered in provided RAG environment
## Planned Functionality/Updates:
- Continue to refine CLI interface
- Currently "Sources" are returned regardlessd of whether or not we fell back to web scraping or used provided context
- want to add support for more file extensions
- want to make sure that GPU is being detected accurately for model suggestion
- Want to dynamically pull list of models from Ollama instead of a fixed list of models
