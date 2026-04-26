import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api")
USER_ID = "test_user_1"

st.set_page_config(page_title="Enterprise RAG Copilot", page_icon="🕵️‍♂️")
st.title("Enterprise RAG Copilot 🕵️‍♂️")

with st.sidebar:
    st.header("⚙️ Управление")
    st.write(f"**Текущий пользователь:** `{USER_ID}`")

    if st.button("🗑 Сбросить контекст", use_container_width=True):
        try:
            response = httpx.delete(f"{API_URL}/chat/context/{USER_ID}", timeout=None)
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("История чата успешно удалена из базы!")
            else:
                st.error(f"Ошибка при удалении: {response.text}")
        except Exception as e:
            st.error(f"Сервер недоступен: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        # we get last 5 messages from the history to provide some context to the agent
        res = httpx.get(f"{API_URL}/chat/history/{USER_ID}?limit=5", timeout=None)
        if res.status_code == 200:
            history = res.json()
            for doc in history:
                st.session_state.messages.append(
                    {"role": "user", "content": doc["query"]}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": doc["answer"]}
                )
    except Exception as e:
        print(f"Error while fetching chat history: {e}")

# show the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# handle new user input
if prompt := st.chat_input("Ask me anything about SEC 10-K reports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent's turn to respond
    with st.chat_message("assistant"):
        with st.spinner(
            "Analyzing the database and going through the critic's review..."
        ):
            try:
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={"user_id": USER_ID, "query": prompt},
                    timeout=None,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer available")
                    revisions = data.get("revisions_needed", 0)

                    st.markdown(answer)

                    # If the critic forced the agent to revise the answer
                    #  show a caption about it
                    if revisions > 0:
                        st.caption(
                            f"""🔄 The critic forced the agent 
                            to revise the answer {revisions} time(s)."""
                        )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(
                    f"""Error connecting to FastAPI. 
                      Make sure the server is running. Error: {e}"""
                )
