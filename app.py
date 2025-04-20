import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st
from urllib.parse import quote
from bs4 import BeautifulSoup
import networkx as nx
import difflib
 
from recommendation_utils import (
    get_recent_editors,
    get_page_backlinks,
    recommend_safe_pages,
    recommend_safe_versions
)
 
# ---------- TRAINING PHASE ----------
 
# Step 1: Load dataset
df = pd.read_csv("D:\M.Tech\Machine Learning with Big Data\Project\Dataset\pan-wikipedia-vandalism-corpus-2011\pan-wikipedia-vandalism-corpus-2011\edits-en.csv")
 
# Step 2: Preprocessing
df['editcomment'] = df['editcomment'].fillna('')
X = df['editcomment']
y = df['class']  # 0 = regular, 1 = vandalism
 
# Step 3: Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)
 
# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
 
# Step 5: Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
 
# Step 6: Save model and vectorizer
joblib.dump(model, "vandalism_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
 
# ---------- STREAMLIT APP ----------
 
st.title("Wikipedia Vandalism Detector & Recommender")
 
wiki_url = st.text_input("Enter Wikipedia Page Link (e.g., https://en.wikipedia.org/wiki/Air_Jordan)")
 
if wiki_url:
    def extract_page_title(url):
        return url.split("/")[-1].replace("_", " ")
 
    def get_page_content(title):
        api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles={quote(title)}&rvslots=*&rvprop=content&formatversion=2&format=json"
        response = requests.get(api_url)
        data = response.json()
        try:
            content = data['query']['pages'][0]['revisions'][0]['slots']['main']['content']
            return content
        except:
            return ""
 
    def get_page_backlinks(title):
        backlinks = []
        bl_url = f"https://en.wikipedia.org/w/api.php?action=query&list=backlinks&bltitle={quote(title)}&bllimit=10&format=json"
        response = requests.get(bl_url).json()
        if 'query' in response:
            backlinks = [link['title'] for link in response['query']['backlinks']]
        return backlinks
 
    def build_pagerank_graph(base_title):
        G = nx.DiGraph()
        visited = set()
 
        def dfs(title, depth):
            if title in visited or depth > 1:
                return
            visited.add(title)
            links = get_page_backlinks(title)
            for linked_title in links:
                G.add_edge(linked_title, title)
                dfs(linked_title, depth + 1)
 
        dfs(base_title, 0)
        return G
 
    def get_latest_edit_comment(title):
        history_url = f"https://en.wikipedia.org/w/index.php?title={quote(title)}&action=history"
        response = requests.get(history_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        latest_edit = soup.find('li', class_='mw-history-histlinks')
        if latest_edit:
            comment = latest_edit.find('span', class_='comment')
            if comment:
                return comment.text.strip()
        return "No comment found"
 
    # Extract title
    page_title = extract_page_title(wiki_url)
 
    # Get latest edit comment
    latest_comment = get_latest_edit_comment(page_title)
    st.markdown(f"**Latest Edit Comment:** {latest_comment}")
 
    # Predict vandalism using latest comment
    loaded_model = joblib.load("vandalism_model.pkl")
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    comment_vec = loaded_vectorizer.transform([latest_comment])
 
    prediction = loaded_model.predict(comment_vec)[0]
    st.write("### Prediction:", "🛡️ Regular" if prediction == 0 else "⚠️ Vandalism")
 
    if prediction == 1:
        st.markdown("## 🔍 Vandalism Detected – Recovery Recommendations")
 
    # Trusted Editors
    trusted_editors = get_recent_editors(page_title)
    if trusted_editors:
        st.markdown("### 👥 Trusted Editors (Reverted Changes):")
        st.write(trusted_editors)
 
    # Safe Related Pages
    backlinks = get_page_backlinks(page_title)
    safe_pages = recommend_safe_pages(backlinks)
    if safe_pages:
        st.markdown("### 🧭 Safe Related Articles:")
        for title, count in safe_pages:
            st.markdown(f"- [{title}](https://en.wikipedia.org/wiki/{quote(title)}) – {count} edits")
 
    # Safe Past Revisions
    current_text = get_page_content(page_title)
    safe_versions = recommend_safe_versions(page_title, current_text)
    if safe_versions:
        st.markdown("### ⏪ Previous Safe Revisions:")
        for rev_id, ts, score in safe_versions:
            rev_link = f"https://en.wikipedia.org/w/index.php?oldid={rev_id}"
            st.markdown(f"- [Revision {rev_id}]({rev_link}) ({score*100:.1f}% match) – {ts}")
 
    # Compare revisions
    existing_text = "This is an example of the older version of the article."
    differences = list(difflib.ndiff(existing_text.split(), current_text.split()))
    st.markdown("### Differences (example):")
    st.code(" ".join(differences[:50]))
 
    # Similarity score
    matcher = difflib.SequenceMatcher(None, existing_text, current_text)
    accuracy_score = matcher.ratio()
    st.metric(label="Similarity with Previous Version", value=f"{accuracy_score*100:.2f}%")
 
    # Real PageRank Recommender
    st.markdown("### Recommended Pages via PageRank")
    G = build_pagerank_graph(page_title)
    pr_scores = nx.pagerank(G)
    ranked_pages = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    df_ranked = pd.DataFrame(ranked_pages, columns=["Page", "PageRank Score"])
    st.dataframe(df_ranked)
 
    # Optional: Save user query to local file
    with open("user_history_log.txt", "a") as log:
        log.write(f"{page_title}, {prediction}, {accuracy_score:.2f}\n")