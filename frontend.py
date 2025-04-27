import streamlit as st
import requests
import os
import pandas as pd
import json
from dotenv import load_dotenv

st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #000000;
    }
    
    /* Text and headers - neon blue for headers, white for text */
    h1, h2, h3, h4 {
        color: #00BFFF !important;
    }
    
    p, div, span, label {
        color: #FFFFFF;
    }
    
    /* Button styling - subtle glow on hover */
    .stButton > button {
        background-color: #00BFFF;
        color: #FFFFFF;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #00BFFF;
        filter: brightness(110%);
        box-shadow: 0 0 5px #00BFFF;
    }
    
    /* File uploader styling - fix hover color */
    .stFileUploader > div[data-testid="stFileUploadDropzone"] {
        color: #FFFFFF;
    }
    
    .stFileUploader > div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #00BFFF !important;
        color: #00BFFF !important;
    }
    
    /* Make sure file upload browse button is neon blue and no reddish outline */
    .stFileUploader button[kind="secondary"] {
        background-color: #00BFFF !important;
        color: #FFFFFF !important;
        border: none !important;
        outline: none !important;
    }
    
    .stFileUploader button[kind="secondary"]:hover,
    .stFileUploader button[kind="secondary"]:focus,
    .stFileUploader button[kind="secondary"]:active {
        background-color: #00BFFF !important;
        filter: brightness(110%);
        box-shadow: 0 0 5px #00BFFF;
        outline: none !important;
        border: none !important;
    }

    /* Style for tabs */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00BFFF !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stInfo {
        background-color: rgba(0, 191, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    
    .stError, .stWarning {
        background-color: rgba(255, 99, 71, 0.1) !important;
        color: #FFFFFF !important;
    }
    
    /* Chat styling */
    .stChatMessage [data-testid="stChatMessageContent"] {
        color: #FFFFFF;
    }
    
    /* Previously uploaded files */
    .file-uploaded {
        color: #00BFFF !important;
    }
</style>
""", unsafe_allow_html=True)

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
            st.session_state.document_files = files  # Store in session_state
        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")

def upload_excel(file):
    if not file:
        st.warning("No file selected.")
        return
    
    try:
        files_data = {"file": (file.name, file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(f"{API_URL}/upload-excel/", files=files_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Excel file uploaded and processed successfully!")
            st.write(f"Created tables: {', '.join(result['tables_created'])}")
            st.session_state.excel_file = file  # Store in session_state
            show_table_preview()
        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Error uploading Excel file: {str(e)}")

def show_table_preview():
    try:
        response = requests.get(f"{API_URL}/excel-tables/")
        if response.status_code == 200:
            tables = response.json()["tables"]
            
            if not tables:
                st.info("No tables available in the database.")
                return
            for table_name, table_info in tables.items():
                with st.expander(f"Table: {table_name} ({table_info['row_count']} rows)"):
                    if table_info['sample_data']:
                        df = pd.DataFrame(
                            table_info['sample_data'], 
                            columns=table_info['columns']
                        )
                        st.write("Sample data:")
                        st.dataframe(df)
                    
                    st.write("Columns:")
                    col_info = pd.DataFrame({
                        "Column": table_info['columns'],
                        "Type": table_info['types']
                    })
                    st.dataframe(col_info)
        else:
            st.error(f"Failed to retrieve table information: {response.text}")
    except Exception as e:
        st.error(f"Error retrieving table information: {str(e)}")

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

def query_sql(question: str):
    try:
        with st.spinner("Analyzing data..."):
            response = requests.post(f"{API_URL}/sql-query/?query={question}")
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                return f"Error: {response.text}"
    except Exception as e:
        return f"Error querying SQL: {str(e)}"

def main():
    st.title("Retrieval Augmented Generation System")
    st.write("Upload Documents/Excel files, submit URLs, or chat with the system.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "sql_chat_history" not in st.session_state:
        st.session_state.sql_chat_history = []

    tab1, tab2, tab3 = st.tabs([
        "Upload", 
        "Submit URL", 
        "Query"
    ])

    with tab1:
        st.header("Upload Files")
        file_type = st.selectbox("Select file type", ["Document", "Excel"])

        if file_type == "Document":
            if "document_files" in st.session_state:
                st.write("Previously uploaded documents:")
                for uploaded_file in st.session_state.document_files:
                    st.markdown(f'<p class="file-uploaded">{uploaded_file.name}</p>', unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Choose PDF or DOCX files", 
                type=["pdf", "docx"], 
                accept_multiple_files=True,
                key="doc_uploader"
            )
            if st.button("Upload Documents"):
                upload_sds(uploaded_files)
        
        elif file_type == "Excel":
            if "excel_file" in st.session_state:
                st.markdown('Previously uploaded Excel file:', unsafe_allow_html=True)
                st.markdown(f'<p class="file-uploaded">{st.session_state.excel_file.name}</p>', unsafe_allow_html=True)
            
            excel_file = st.file_uploader(
                "Choose Excel file", 
                type=["xlsx", "xls"],
                key="excel_uploader"
            )
            if st.button("Upload Excel"):
                upload_excel(excel_file)

    with tab2:
        st.header("Submit a URL")
        url = st.text_input("Enter URL (e.g., https://example.com)")
        if st.button("Submit URL"):
            submit_url(url)

    with tab3:
        st.header("Query the System")
        query_type = st.selectbox("Select query type", ["Chat with Documents", "Query Excel Data"])
        
        if query_type == "Chat with Documents":
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            doc_question = st.chat_input("Ask about your documents...", key="doc_chat")
            if doc_question:
                st.session_state.chat_history.append({"role": "user", "content": doc_question})
                with st.chat_message("user"):
                    st.markdown(doc_question)

                answer = query_rag(doc_question)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

        elif query_type == "Query Excel Data":
            for message in st.session_state.sql_chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            sql_question = st.chat_input("Ask about your data...", key="sql_chat")
            if sql_question:
                st.session_state.sql_chat_history.append({"role": "user", "content": sql_question})
                with st.chat_message("user"):
                    st.markdown(sql_question)

                answer = query_sql(sql_question)
                st.session_state.sql_chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

if __name__ == "__main__":
    main()

