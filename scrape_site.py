import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

BASE_URL = "https://www.qubitdynamics.ai"
DOMAIN = "qubitdynamics.ai"

visited = set()
all_data = []

def get_page_text(url):
    """Extract visible text from a given URL"""
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
            tag.extract()
        text = " ".join(soup.stripped_strings)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def get_links_from_page(url):
    """Find all internal links on a page"""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0].strip()
            if not href:
                continue
            full_url = urljoin(url, href).rstrip("/")
            # Skip non-HTML files
            if any(full_url.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".zip"]):
                continue
            if DOMAIN in full_url and full_url.startswith("http"):
                links.add(full_url)
        return list(links)
    except Exception as e:
        print(f"Error getting links from {url}: {e}")
        return []

def crawl(url, depth=0, max_depth=6):
    """Recursive crawler"""
    if url in visited or depth > max_depth:
        return
    visited.add(url)

    text = get_page_text(url)
    if len(text) > 100:
        all_data.append({"url": url, "content": text})
        print(f"[+] Scraped ({depth}): {url} - chars: {len(text)}")

    links = get_links_from_page(url)
    for link in links:
        crawl(link, depth + 1, max_depth)
        time.sleep(0.3)  # gentle delay

def main():
    print("Starting full Qubit Dynamics site crawl...")
    crawl(BASE_URL, 0, max_depth=6)
    with open("qd_full_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nDone! Scraped {len(all_data)} pages.")
    print("Saved as qd_full_data.json")

if __name__ == "__main__":
    main()
