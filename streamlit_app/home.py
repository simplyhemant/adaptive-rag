
"""
Home page for the Adaptive RAG application.
"""

import uuid
import streamlit as st

st.set_page_config(
    page_title="Adaptive RAG Assistant",
    page_icon="💬",
    layout="wide"
)

# Hide Streamlit's default sidebar navigation
hide_sidebar_style = """
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

st.title("💬 Welcome to Adaptive RAG Assistant")

st.write(
    "Upload documents and ask questions using "
    "your Adaptive RAG pipeline."
)

# Initialize a unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

st.success("Your chat session is ready.")

if st.button("Start Chat", type="primary"):
    st.switch_page("pages/chat.py")
