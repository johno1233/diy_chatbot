import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import quote_plus

#------------ CONFIG --------------------
SEARCH_ENGINE = "https://duckduckgo.com/html/?q=" # HTML results, easy to parse
MAX_RESULTS = 3
EMBEDDINT_MODEL = "BAAI/bge-small-en-v1.5"
#----------------------------------------

def search_web(query):
    """Perfrom a simple DuckDuckGosearch and return top result URLs"""
    search_url = SEARCH_ENGINE + quote_plus(query)
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    for a in soup.select("a.result_a")[:MAX_RESULTS]:
        href = a.get("href")
        if href and href.startswith("http"):
            urls.append(href)
    return urls

def scrape_url(url):
    """Download and extract visible text from a web page."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = " ".soin(soup.stripped_strings)
        return text
    except Exceptin as e:
        print(f"Error scraping {url}: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Split text inot smaller chunks for embedding."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def web_retrieve(query, k=3):
    """Retrieve context chunks from live web search."""
    urls = search_web(query)
    if not urls:
        return []
    
    pages_text = []
    for url in urls:
        page = scrape_url(url)
        if page:
            pages_text.append(page)

    full_text = "\n".join(pages_text)
    chunks = chunk_text(full_text)

    embeddings = HuggingFaceBgeEmbeddings(
        mode_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    temp_store = FAISS.from_texts(chunks, embeddings)
    return temp_store.similartity_search(query, k=k)
