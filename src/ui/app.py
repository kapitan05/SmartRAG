import json
import os
import threading
import time
from typing import Any

import httpx
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api")

st.set_page_config(page_title="Enterprise RAG Copilot", page_icon="🕵️‍♂️")
st.title("Enterprise RAG Copilot 🕵️‍♂️")

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "use_critic" not in st.session_state:
    st.session_state.use_critic = False

if "use_planner" not in st.session_state:
    st.session_state.use_planner = False


def get_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.access_token}"}


if st.session_state.access_token is None:
    st.title("Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                res = httpx.post(
                    f"{API_URL}/token", data={"username": user, "password": pw}
                )
                if res.status_code == 200:
                    st.session_state.access_token = res.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("reg_form"):
            new_user = st.text_input("New Username")
            new_pw = st.text_input("New Password", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                res = httpx.post(
                    f"{API_URL}/register",
                    data={"username": new_user, "password": new_pw},
                )
                if res.status_code == 201:
                    st.success("Account created! You can now login.")
                else:
                    try:
                        error_msg = res.json().get("detail", "Registration failed")
                        st.error(f"Error: {error_msg}")
                    except json.JSONDecodeError:
                        st.error(f"Server Error ({res.status_code}): {res.text}")
    st.stop()


# SIDEBAR LOGIC
with st.sidebar:
    st.header("⚙️ Management")

    if st.session_state.is_generating:
        st.warning("⏳ Generation in progress...")

    if st.button(
        "🚪 Logout", use_container_width=True, disabled=st.session_state.is_generating
    ):
        st.session_state.access_token = None
        st.session_state.messages = []
        st.rerun()

    # Toggles now stay visible but gray out when generating
    st.session_state.use_planner = st.toggle(
        "Enable Planner",
        value=st.session_state.use_planner,
        help="If enabled, the AI will use the planner node to break down complex queries.",
        disabled=st.session_state.is_generating,
    )

    st.session_state.use_critic = st.toggle(
        "Enable Critic",
        value=st.session_state.use_critic,
        help="If disabled, the AI will answer faster but without self-reflection.",
        disabled=st.session_state.is_generating,
    )

    if st.button(
        "🗑 Reset Context",
        use_container_width=True,
        disabled=st.session_state.is_generating,
    ):
        try:
            response = httpx.delete(
                f"{API_URL}/chat/context", headers=get_headers(), timeout=None
            )
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("Chat history successfully deleted from the database!")
            else:
                st.error(f"Error occurred while deleting: {response.text}")
        except Exception as e:
            st.error(f"Server is not available: {e}")


# CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        res = httpx.get(f"{API_URL}/chat/history?limit=5", headers=get_headers())
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# CHAT INPUT
if prompt := st.chat_input(
    "Ask me anything about SEC 10-K reports...", disabled=st.session_state.is_generating
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.session_state.is_generating = True
    st.rerun()

# BACKGROUND GENERATION LOGIC
if st.session_state.is_generating and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt

    with st.chat_message("assistant"):
        # 1. Create a stable area for the Stop button and Status
        stop_col, status_col = st.columns([1, 3])

        with stop_col:
            stop_clicked = st.button(
                "🛑 Stop", key="stop_btn", use_container_width=True
            )

        with status_col:
            # Using an empty placeholder instead of st.spinner to prevent flickering
            status_placeholder = st.empty()

        if stop_clicked:
            st.session_state.is_generating = False
            st.session_state.pending_prompt = None
            if "api_result" in st.session_state:
                del st.session_state["api_result"]
            st.rerun()

        # 2. Handle the API Thread
        if "api_result" not in st.session_state:
            st.session_state.api_result = {
                "data": None,
                "error": None,
                "status": None,
                "done": False,
            }

            payload = {
                "query": prompt,
                "use_critic": st.session_state.get("use_critic", False),
                "use_planner": st.session_state.get("use_planner", False),
            }

            def fetch_data() -> None:
                try:
                    response = httpx.post(
                        f"{API_URL}/chat",
                        json=payload,
                        headers=get_headers(),
                        timeout=None,
                    )
                    st.session_state.api_result["status"] = response.status_code
                    if response.status_code == 200:
                        st.session_state.api_result["data"] = response.json()
                    else:
                        st.session_state.api_result["error"] = response.text
                except Exception as e:
                    st.session_state.api_result["error"] = str(e)
                finally:
                    st.session_state.api_result["done"] = True

            thread = threading.Thread(target=fetch_data)
            add_script_run_ctx(thread)
            thread.start()

        # 3. The Stable Wait Loop
        if not st.session_state.api_result["done"]:
            # Show a static message instead of a wrapping spinner
            status_placeholder.markdown(
                "🔍 *Analyzing reports and consulting critic...*"
            )
            time.sleep(0.5)
            st.rerun()

        # 4. Finalizing
        status_placeholder.empty()  # Clear the "Analyzing" text

        api_result = st.session_state.api_result
        if api_result["done"]:
            if api_result["status"] == 200 and api_result["data"]:
                data = api_result["data"]
                answer = data.get("answer", "No answer available")
                revisions = data.get("revisions_needed", 0)

                st.markdown(answer)
                if revisions > 0:
                    st.caption(f"🔄 Revised {revisions} time(s) by critic.")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            elif api_result["error"]:
                st.error(f"Error: {api_result['error']}")

            # Cleanup
            st.session_state.is_generating = False
            st.session_state.pending_prompt = None
            del st.session_state["api_result"]
            st.rerun()
