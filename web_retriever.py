import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import quote_plus
import trafilatura

# ------------ CONFIG --------------------
SEARCH_ENGINE = "https://duckduckgo.com/html/?q="  # HTML results, easy to parse
MAX_RESULTS = 5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# ----------------------------------------


def search_web(query):
    """Perfrom a simple DuckDuckGosearch and return top result URLs"""
    search_url = SEARCH_ENGINE + quote_plus(query)
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    for a in soup.select("a.result__a")[:MAX_RESULTS]:
        href = a.get("href")
        if href:
            full_href = "https://html.duckduckgo.com" + href
            urls.append(full_href)
    return urls


def scrape_url(url):
    """Download and extract visible text from a web page."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded, include_comments=False, include_tables=False
            )
            return text if text else ""
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


def chunk_text(text, chunk_size=800):
    """Split text inot smaller chunks for embedding."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


def web_retrieve(query, k=5):
    """Retrieve context chunks from live web search."""
    urls = search_web(query)
    print(f"[Debug] Found URLs: {urls}")
    if not urls:
        return [], []

    pages_text = [scrape_url(url) for url in urls if scrape_url(url)]
    sources = [url for url in urls if scrape_url(url)]

    if not pages_text:
        print("[DEBUG] No text scraped from URLs.")
        return [], []

    full_text = "\n".join(pages_text)
    chunks = chunk_text(full_text)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    )

    temp_store = FAISS.from_texts(chunks, embeddings)
    results = temp_store.similarity_search(query, k=k)

    return results, sources
