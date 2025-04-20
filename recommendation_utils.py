import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
import difflib

def get_recent_editors(title, limit=20):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvlimit": limit,
        "titles": title,
        "rvprop": "user|comment",
        "formatversion": 2,
        "format": "json"
    }
    res = requests.get(endpoint, params=params).json()
    revisions = res['query']['pages'][0].get('revisions', [])
    editors = set()
    for rev in revisions:
        if any(kw in rev.get('comment', '').lower() for kw in ['revert', 'undo', 'rv']):
            editors.add(rev.get('user'))
    return list(editors)

def get_edit_count(title):
    try:
        url = f"https://en.wikipedia.org/w/index.php?title={quote(title)}&action=history"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        return len(soup.find_all('li', class_='mw-history-histlinks'))
    except:
        return 0

def get_page_backlinks(title, limit=10):
    bl_url = f"https://en.wikipedia.org/w/api.php?action=query&list=backlinks&bltitle={quote(title)}&bllimit={limit}&format=json"
    response = requests.get(bl_url).json()
    if 'query' in response:
        return [link['title'] for link in response['query']['backlinks']]
    return []

def recommend_safe_pages(backlinked_titles):
    safe_pages = []
    for title in backlinked_titles:
        count = get_edit_count(title)
        if count < 10:  # Threshold for low-edit "safe" articles
            safe_pages.append((title, count))
    return sorted(safe_pages, key=lambda x: x[1])

def get_past_revision_contents(title, limit=5):
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvlimit": limit,
        "titles": title,
        "rvslots": "main",
        "rvprop": "ids|user|timestamp|content",
        "formatversion": 2,
        "format": "json"
    }
    response = requests.get(endpoint, params=params).json()
    return response['query']['pages'][0].get('revisions', [])

def recommend_safe_versions(title, current_text, top_k=3):
    past = get_past_revision_contents(title)
    candidates = []
    for rev in past:
        old = rev['slots']['main'].get('content', '')
        sim = difflib.SequenceMatcher(None, old, current_text).ratio()
        if sim > 0.8:
            candidates.append((rev['revid'], rev['timestamp'], sim))
    return sorted(candidates, key=lambda x: x[2], reverse=True)[:top_k]
