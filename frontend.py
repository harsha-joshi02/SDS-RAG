import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")

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
    try:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/query/?question={question}")
            if response.status_code == 200:
                result = response.json()
                return result["answer"]
            else:
                return f"Error: {response.text}"
    except Exception as e:
        return f"Error querying: {str(e)}"

def main():
    st.title("Retrieval Augmented Generation System")
    st.write("Upload Documents, submit URLs, or chat with the system.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
        st.header("Chat with the System")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask your question here...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            answer = query_rag(question)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    

if __name__ == "__main__":
    main()