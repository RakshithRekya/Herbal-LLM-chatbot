import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chat.chat import build_chain

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Herbal Assistant",
    page_icon="🌿",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🌿 Herbal Assistant")
st.caption("Ask me anything about herbal remedies, dosages, and properties.")
st.divider()

# ── Load chain once and cache it ───────────────────────────────────────────
@st.cache_resource
def get_chain():
    return build_chain()

chain = get_chain()

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about herbs..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
