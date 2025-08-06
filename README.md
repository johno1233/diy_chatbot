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

## Planned Functionality/Updates:
- Have the GUI look prettier
- GUI version should be able to take input for path/to/files in the GUI not in the command line
- Add web scraping functionality to get up-to-date information based on what is found online 

### NOTE:
This is a vibe coding project and serves as a learning tool for me to understand AI Dev workflows, how to leverage AI chat bots (i.e., Grok, ChatGPT, etc.) in my development work flow, and also learn python.
