import streamlit as st
import requests
import os
import pandas as pd
from dotenv import load_dotenv
from app.config import CONFIG
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("frontend.log")
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
API_URL = os.getenv("API_URL")

def upload_sds(files):
    """
    Uploads selected SDS files to the backend API and updates the session state with uploaded files.

    Args:
        files (list): List of uploaded file objects from Streamlit's file uploader.
    """

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
    """
    Uploads an Excel file to the backend API, displays the result in Streamlit, and updates session state.

    Args:
        file: The uploaded Excel file object from Streamlit.

    Returns:
        Tuple[str, list] or (None, None): File name and list of created tables if successful, otherwise None values.
    """

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
    """
    Submits a schema definition to the backend API and updates Streamlit session state.

    Args:
        schema_name (str): Name of the schema to be created.
        file_name (str): Name of the Excel file associated with the schema.
        tables (list): List of tables to include in the schema.

    Returns:
        None
    """

    if not schema_name:
        st.warning("Please enter a schema name.")
        return
    logger.info(f"Submitting schema '{schema_name}' for file '{file_name}' with tables: {tables}")
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
        else:
            st.error(f"Failed to set schema: {schema_response.text}")
    except Exception as e:
        st.error(f"Error submitting schema: {str(e)}")

def show_table_preview():
    """
    Fetches and displays a preview of tables from the uploaded Excel file using the backend API.

    Shows sample data and column metadata for each table in an expandable Streamlit UI.

    Returns:
        None
    """

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

def query_rag(question: str, doc_path: str, evaluate_metrics: bool):
    """
    Sends a RAG query to the backend API and optionally evaluates response metrics.

    Args:
        question (str): The user query.
        doc_path (str): Path to the document(s) to retrieve context from.
        evaluate_metrics (bool): Whether to evaluate the response using metrics like hallucination or precision.

    Returns:
        Tuple[str, dict, list]: A tuple containing the answer string, a dictionary of evaluation metrics, and a list of source documents.
    """

    try:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/query/?question={question}&sds_paths={doc_path}&evaluate_metrics={evaluate_metrics}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received RAG query response: question={question}, answer={result['answer'][:50]}..., metrics={result['metrics']}")
                return result["answer"], result.get("metrics", {}), result.get("sources", [])
            else:
                logger.error(f"RAG query failed: {response.text}")
                return f"Error: {response.text}", {}, []
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return f"Error querying: {str(e)}", {}, []

def query_sql(question: str, schema_name: str, evaluate_metrics: bool):
    """
    Sends a natural language query to the SQL agent backend and optionally evaluates the response.

    Args:
        question (str): The user's natural language question about the data.
        schema_name (str): The schema to query within the SQL database.
        evaluate_metrics (bool): Whether to evaluate the SQL response using metrics.

    Returns:
        Tuple[str, dict, list]: A tuple containing the query response, evaluation metrics (if any), and list of sources used.
    """

    try:
        with st.spinner("Analyzing data..."):
            response = requests.post(f"{API_URL}/sql-query/?query={question}&schema_name={schema_name}&evaluate_metrics={evaluate_metrics}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received SQL query response: question={question}, response={result['response'][:50]}...")
                return result["response"], result.get("metrics", {}), result.get("sources", [])
            else:
                logger.error(f"SQL query failed: {response.text}")
                return f"Error: {response.text}", {}, []
    except Exception as e:
        logger.error(f"Error querying SQL: {str(e)}")
        return f"Error querying SQL: {str(e)}", {}, []

def query_web(question: str, evaluate_metrics: bool):
    """
    Queries the web using the given question and optionally evaluates the response.

    Args:
        question (str): The user's natural language query.
        evaluate_metrics (bool): Whether to evaluate the generated answer using predefined metrics.

    Returns:
        Tuple[str, dict, list]: A tuple containing the generated answer, evaluation metrics (if any), and source URLs used.
    """

    try:
        with st.spinner("Searching the web..."):
            response = requests.post(f"{API_URL}/web-search/?question={question}&evaluate_metrics={evaluate_metrics}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received web query response: question={question}, answer={result['answer'][:50]}..., metrics={result['metrics']}")
                return result["answer"], result.get("metrics", {}), result.get("sources", [])
            else:
                logger.error(f"Web query failed: {response.text}")
                return f"Error: {response.text}", {}, []
    except Exception as e:
        logger.error(f"Error querying web: {str(e)}")
        return f"Error querying web: {str(e)}", {}, []

def main():
    st.title("Retrieval Augmented Generation System")
    st.write("Upload Documents/Excel files or chat with the system.")

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

    tab1, tab2 = st.tabs([
        "Upload",
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
                    logger.info(f"'Submit Schema' button clicked for schema: {schema_name}")
                    submit_schema(
                        schema_name,
                        st.session_state.excel_upload_result["file_name"],
                        st.session_state.excel_upload_result["tables"]
                    )

    with tab2:
        st.header("Query the System")
        
        evaluate_metrics = st.toggle("Show Evaluation Metrics", value=False, key="evaluate_metrics_toggle")
        
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
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    answer = message["content"]
                    citations = ""
                    if "Citations:" in answer:
                        answer_parts = answer.split("Citations:", 1)
                        answer = answer_parts[0].strip()
                        citations = answer_parts[1].strip()
                    st.markdown(answer)
                    
                    if evaluate_metrics and "metrics" in message and message["metrics"]:
                        with st.expander("Evaluation Metrics"):
                            metrics = message["metrics"]
                            logger.info(f"Rendering metrics for message: {metrics}")
                            metrics_df = pd.DataFrame([
                                {"Metric": "Hallucination", "Value": f"{metrics.get('hallucination', 0.0):.2f}"},
                                {"Metric": "Context Precision", "Value": f"{metrics.get('context_precision', 0.0):.2f}"}
                            ])
                            st.markdown('<div class="metrics-table">', unsafe_allow_html=True)
                            st.table(metrics_df.to_dict('records'))
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    sources = message.get("sources", [])
                    if citations or sources:
                        with st.expander("Citations and Sources"):
                            if citations:
                                st.markdown("**Citations:**")
                                st.markdown(citations)
                            if sources:
                                st.markdown("**Web Sources:**")
                                for source in sources:
                                    st.markdown(f"- [{source}]({source})")

        question = st.chat_input(f"Ask about {selected_option}...", key=f"chat_{selected_option}")
        if question:
            chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            if query_type == "schema":
                answer, metrics, sources = query_sql(question, query_value, evaluate_metrics)
                citations = ""
            elif query_type == "web":
                answer, metrics, sources = query_web(question, evaluate_metrics)
                citations = ""
            else:
                answer, metrics, sources = query_rag(question, query_value, evaluate_metrics)
                citations = ""
                if "Citations:" in answer:
                    answer_parts = answer.split("Citations:", 1)
                    answer = answer_parts[0].strip()
                    citations = answer_parts[1].strip()
            
            chat_history.append({
                "role": "assistant",
                "content": answer + ("\n\nCitations:\n" + citations if citations else ""),
                "metrics": metrics,
                "sources": sources
            })
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                logger.info(f"Displaying new answer with metrics: {metrics}")
                
                if evaluate_metrics and metrics:
                    with st.expander("Evaluation Metrics"):
                        metrics_df = pd.DataFrame([
                            {"Metric": "Hallucination", "Value": f"{metrics.get('hallucination', 0.0):.2f}"},
                            {"Metric": "Context Precision", "Value": f"{metrics.get('context_precision', 0.0):.2f}"}
                        ])
                        st.markdown('<div class="metrics-table">', unsafe_allow_html=True)
                        st.table(metrics_df.to_dict('records'))
                        st.markdown('</div>', unsafe_allow_html=True)
                
                if citations or sources:
                    with st.expander("Citations and Sources"):
                        if citations:
                            st.markdown("**Citations:**")
                            st.markdown(citations)
                        if sources:
                            st.markdown("**Web Sources:**")
                            for source in sources:
                                st.markdown(f"- [{source}]({source})")

if __name__ == "__main__":
    main()