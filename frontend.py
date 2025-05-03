import streamlit as st
import requests
import os
import pandas as pd
from dotenv import load_dotenv
from app.config import CONFIG

st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    
    h1, h2, h3, h4 {
        color: #00BFFF !important;
    }
    
    p, div, span, label {
        color: #FFFFFF;
    }
    
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
    
    .stFileUploader > div[data-testid="stFileUploadDropzone"] {
        color: #FFFFFF;
    }
    
    .stFileUploader > div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #00BFFF !important;
        color: #00BFFF !important;
    }
    
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

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00BFFF !important;
    }
    
    .stSuccess, .stInfo {
        background-color: rgba(0, 191, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    
    .stError, .stWarning {
        background-color: rgba(255, 99, 71, 0.1) !important;
        color: #FFFFFF !important;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        color: #FFFFFF;
    }
    
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
            if "document_files" not in st.session_state:
                st.session_state.document_files = []
            st.session_state.document_files.extend(files) 
        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Error uploading files: {str(e)}")

def upload_excel(file):
    if not file:
        st.warning("No file selected.")
        return None, None
    
    try:
        files_data = {"file": (file.name, file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(f"{API_URL}/upload-excel/", files=files_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Excel file uploaded successfully!")
            st.write(f"Created tables: {', '.join(result['tables_created'])}")
            st.session_state.excel_file = file   
            show_table_preview()
            return file.name, result["tables_created"]
        else:
            st.error(f"Upload failed: {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Error uploading Excel file: {str(e)}")
        return None, None

def submit_schema(schema_name, file_name, tables):
    if not schema_name:
        st.warning("Please enter a schema name.")
        return
    st.write(f"Debug: Submitting schema '{schema_name}' for file '{file_name}' with tables: {tables}")
    try:
        schema_response = requests.post(
            f"{API_URL}/set-schema/",
            json={"schema_name": schema_name, "tables": tables, "file_name": file_name}
        )
        if schema_response.status_code == 200:
            st.success(f"Schema '{schema_name}' set successfully!")
            if "schemas" not in st.session_state:
                st.session_state.schemas = {}
            st.session_state.schemas[schema_name] = tables
            #st.write(f"Debug: Current schemas in session state: {st.session_state.schemas}")
        else:
            st.error(f"Failed to set schema: {schema_response.text}")
    except Exception as e:
        st.error(f"Error submitting schema: {str(e)}")

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

def query_rag(question: str, doc_path: str):
    try:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/query/?question={question}&sds_paths={doc_path}")
            if response.status_code == 200:
                result = response.json()
                return result["answer"]
            else:
                return f"Error: {response.text}"
    except Exception as e:
        return f"Error querying: {str(e)}"

def query_sql(question: str, schema_name: str):
    try:
        with st.spinner("Analyzing data..."):
            response = requests.post(f"{API_URL}/sql-query/?query={question}&schema_name={schema_name}")
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                return f"Error: {response.text}"
    except Exception as e:
        return f"Error querying SQL: {str(e)}"

def query_web(question: str):
    try:
        with st.spinner("Searching the web..."):
            response = requests.post(f"{API_URL}/web-search/?question={question}")
            if response.status_code == 200:
                result = response.json()
                return result["answer"]
            else:
                return f"Error: {response.text}"
    except Exception as e:
        return f"Error querying web: {str(e)}"

def main():
    st.title("Retrieval Augmented Generation System")
    st.write("Upload Documents/Excel files, submit URLs, or chat with the system.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "sql_chat_history" not in st.session_state:
        st.session_state.sql_chat_history = []
    
    if "web_chat_history" not in st.session_state:
        st.session_state.web_chat_history = []
    
    if "document_files" not in st.session_state:
        st.session_state.document_files = []
    
    if "schemas" not in st.session_state:
        st.session_state.schemas = {}
    
    if "excel_upload_result" not in st.session_state:
        st.session_state.excel_upload_result = None

    with st.sidebar:
        st.subheader("Upload Documents")
        st.markdown("""
        - Upload a PDF or DOCX file to process documents.
        """)

        st.subheader("Upload Tables")
        st.markdown("""
        - Upload an Excel file where each sheet contains one table.
        - Tables are converted to SQL tables.
        - Enter a schema name after uploading.
        """)

        st.subheader("Query")
        st.markdown("""
        - Select a Schema name, Document name, or Web Search to chat with the respective data source.
        """)

    tab1, tab2, tab3 = st.tabs([
        "Upload", 
        "Submit URL", 
        "Query"
    ])

    with tab1:
        st.header("Upload Files")
        file_type = st.selectbox("Select file type", ["Document", "Excel"])

        if file_type == "Document":
            if st.session_state.document_files:
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
                file_name, tables = upload_excel(excel_file)
                if file_name and tables:
                    st.session_state.excel_upload_result = {"file_name": file_name, "tables": tables}
            
            if st.session_state.excel_upload_result:
                st.write("Enter schema name for the uploaded Excel file:")
                schema_name = st.text_input("Schema name:", key="schema_input")
                if st.button("Submit Schema", key="submit_schema"):
                    st.write(f"Debug: 'Submit Schema' button clicked")
                    submit_schema(
                        schema_name,
                        st.session_state.excel_upload_result["file_name"],
                        st.session_state.excel_upload_result["tables"]
                    )

    with tab2:
        st.header("Submit a URL")
        url = st.text_input("Enter URL (e.g., https://example.com)")
        if st.button("Submit URL"):
            submit_url(url)

    with tab3:
        st.header("Query the System")
        #st.write(f"Debug: Current schemas in session state: {st.session_state.schemas}")
        
        query_options = ["Web Search"]  
        query_type_map = {"Web Search": {"type": "web", "value": None}}  
        
        for schema_name in st.session_state.schemas.keys():
            query_options.append(f"Schema: {schema_name}")
            query_type_map[f"Schema: {schema_name}"] = {"type": "schema", "value": schema_name}
        
        for doc_file in st.session_state.document_files:
            doc_path = os.path.join(CONFIG["app"]["upload_dir"], doc_file.name)
            query_options.append(f"Document: {doc_file.name}")
            query_type_map[f"Document: {doc_file.name}"] = {"type": "document", "value": doc_path}
        
        if not query_options:
            st.warning("No schemas, documents, or web search available. Please upload files or use web search.")
            return
        
        selected_option = st.selectbox("Select data to query", query_options)
        
        query_type = query_type_map[selected_option]["type"]
        query_value = query_type_map[selected_option]["value"]
        
        if query_type == "schema":
            chat_history = st.session_state.sql_chat_history
        elif query_type == "web":
            chat_history = st.session_state.web_chat_history
        else:
            chat_history = st.session_state.chat_history
        
        for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(f"Ask about {selected_option}...", key=f"chat_{selected_option}")
        if question:
            chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            if query_type == "schema":
                answer = query_sql(question, query_value)
                st.session_state.sql_chat_history.append({"role": "assistant", "content": answer})
            elif query_type == "web":
                answer = query_web(question)
                st.session_state.web_chat_history.append({"role": "assistant", "content": answer})
            else:
                answer = query_rag(question, query_value)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            with st.chat_message("assistant"):
                st.markdown(answer)

if __name__ == "__main__":
    main()