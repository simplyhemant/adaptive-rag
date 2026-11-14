
"""
Chat page for the Adaptive RAG application.
"""

import uuid
import streamlit as st

from utils.api_client import (
    query_backend,
    document_upload_rag
)

st.set_page_config(
    page_title="Adaptive RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None
    }
)

# Initialize session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Header
col1, col2 = st.columns([10, 2])

with col1:
    st.title("💬 Adaptive RAG Chat")

with col2:
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# Document upload section
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        file_description = st.text_input(
            "📄 Describe your document",
            max_chars=300,
            placeholder="E.g. LangGraph tutorial with workflows and code examples"
        )

        if st.button("Upload Document"):
            if not file_description.strip():
                st.warning("Please describe your document.")
            else:
                file_key = (
                    f"{uploaded_file.name}_{file_description.strip()}"
                )

                if file_key in st.session_state.uploaded_files:
                    st.info("This document has already been uploaded.")
                else:
                    with st.spinner("Uploading document..."):
                        success = document_upload_rag(
                            uploaded_file,
                            file_description.strip()
                        )

                    if success:
                        st.success(
                            f"Uploaded: {uploaded_file.name}"
                        )
                        st.session_state.uploaded_files[file_key] = True
                    else:
                        st.error("Document upload failed.")

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# User input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Display user message
    st.session_state.chat_history.append(
        ("user", user_input)
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Adaptive RAG is thinking..."):
            response = query_backend(
                user_input,
                st.session_state.session_id
            )

        st.write(response)

    # Save assistant response
    st.session_state.chat_history.append(
        ("assistant", response)
    )
