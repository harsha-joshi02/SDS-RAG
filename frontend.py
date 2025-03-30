import streamlit as st
import requests
from typing import List
import os

API_URL = "http://localhost:8000"  # into .env

def upload_sds(files):
    if not files:
        st.warning("No files selected.")
        return
    files_data = [("files", (file.name, file.read(), file.type)) for file in files]
    try:
        response = requests.post(f"{API_URL}/upload-sds/", files=files_data)
        if response.status_code == 200:
            st.success("Files uploaded successfully!")
            st.json(response.json())
        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")

def submit_url(url: str):
    if not url:
        st.warning("Please enter a URL.")
        return
    try:
        response = requests.post(f"{API_URL}/submit-url/?url={url}")
        if response.status_code == 200:
            st.success("URL submitted successfully!")
            st.json(response.json())
        else:
            st.error(f"URL submission failed: {response.text}")
    except Exception as e:
        st.error(f"Error submitting URL: {str(e)}")

def query_rag(question: str):
    if not question:
        st.warning("Please enter a question.")
        return
    try:
        response = requests.post(f"{API_URL}/query/?question={question}")
        if response.status_code == 200:
            result = response.json()
            st.success("Query answered!")
            st.markdown("### Answer")
            st.write(result["answer"])
        else:
            st.error(f"Query failed: {response.text}")
    except Exception as e:
        st.error(f"Error querying: {str(e)}")

def main():
    st.title("Retrieval Augmented Generation System")
    st.write("Upload Documents, submit URLs, or query the system.")

    tab1, tab2, tab3 = st.tabs(["Upload Document", "Submit URL", "Query"])

    with tab1:
        st.header("Upload Single/Multiple Documents")
        uploaded_files = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
        if st.button("Upload Files"):
            upload_sds(uploaded_files)

    with tab2:
        st.header("Submit a URL")
        url = st.text_input("Enter URL (e.g., https://example.com)")
        if st.button("Submit URL"):
            submit_url(url)

    with tab3:
        st.header("Ask a Question")
        question = st.text_input("Enter your question")
        if st.button("Submit Question"):
            query_rag(question)

if __name__ == "__main__":
    main()