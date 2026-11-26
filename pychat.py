import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PyChat AI", page_icon="🐍")
st.title("🐍 PyChat - Python AI Assistant")
st.caption("Ask me any beginner Python question (e.g., 'what is a list?', 'explain functions')")

df = pd.read_csv("clean_lessons.csv")

STOPWORDS = {"what", "is", "a", "an", "the", "explain", "tell", "me", "about", "how", "to", "in", "python"}

def clean_and_tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

df["processed_text"] = (df["topic"] + " " + df["keywords"]).apply(clean_and_tokenize)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

def get_pychat_reply(user_query):
    clean_query = clean_and_tokenize(user_query)
    if not clean_query:
        return "Could you ask about a specific Python topic like **variables, lists, loops, or functions**?", None

    query_vec = vectorizer.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = scores.argmax()
    best_score = scores[best_idx]

    if best_score < 0.15:
        return "I'm not sure about that yet! Try asking about **Python, variables, data types, arrays, lists, tuples, sets, dictionaries, loops, functions, or classes**.", None

    row = df.iloc[best_idx]
    code = row["example_code"].replace("\\n", "\n")
    return f"### {row['topic']}\n{row['explanation']}", code

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am **PyChat**. Ask me anything about Python basics!", "code": None}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("code"):
            st.code(msg["code"], language="python")

if prompt := st.chat_input("Ask a Python question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "code": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply_text, reply_code = get_pychat_reply(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply_text, "code": reply_code})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
        if reply_code:
            st.code(reply_code, language="python")
