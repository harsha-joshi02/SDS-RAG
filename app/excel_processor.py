import json
import pandas as pd
import sqlite3
import os
import logging
import time
from typing import List, Dict, Any, Tuple
from groq import Groq
from app.config import CONFIG
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def detect_table_types(df):
    type_mapping = {}
    
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            type_mapping[column] = "INTEGER"
        elif pd.api.types.is_float_dtype(df[column]):
            type_mapping[column] = "REAL"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            type_mapping[column] = "DATETIME"
        else:
            type_mapping[column] = "TEXT"
    
    return type_mapping

class ExcelToSQLProcessor:
    def __init__(self, db_path="excel_data.sqlite"):
        self.db_path = db_path
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.conn = sqlite3.connect(db_path)
        self.table_schemas = {}
    
    def process_excel_file(self, file_path: str) -> List[str]:
        logger.info(f"Processing Excel file: {file_path}")
        start_time = time.time()
        
        try:
            excel_file = pd.ExcelFile(file_path)
            table_names = []
            
            for sheet_name in excel_file.sheet_names:
                table_name = self._sanitize_table_name(sheet_name)
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                df.columns = [self._sanitize_column_name(col) for col in df.columns]
                self._create_sql_table(df, table_name)
                table_names.append(table_name)
                
                self.table_schemas[table_name] = {
                    "columns": list(df.columns),
                    "types": detect_table_types(df),
                    "rows": len(df)
                }
                
                logger.info(f"Created table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
            
            end_time = time.time()
            logger.info(f"Excel Processing Time: {end_time - start_time:.2f} sec")
            
            return table_names
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}", exc_info=True)
            raise
    
    def _sanitize_table_name(self, name: str) -> str:
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in name.replace(' ', '_'))
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'table_' + sanitized
        return sanitized.lower()
    
    def _sanitize_column_name(self, name: str) -> str:
        if isinstance(name, (int, float)):
            return f"col_{name}"
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in str(name).replace(' ', '_'))
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = 'col_' + sanitized
        return sanitized.lower()
    
    def _create_sql_table(self, df: pd.DataFrame, table_name: str) -> None:
        try:
            type_mapping = detect_table_types(df)
            column_defs = []
            for col in df.columns:
                column_defs.append(f'"{col}" {type_mapping[col]}')
            
            create_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_defs)})'
            
            with self.conn:
                self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                self.conn.execute(create_query)
                
                df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        except Exception as e:
            logger.error(f"Error creating SQL table: {str(e)}", exc_info=True)
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        tables = {}
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table in table_names:
                cursor.execute(f"PRAGMA table_info('{table}');")
                columns = cursor.fetchall()
                
                cursor.execute(f"SELECT * FROM '{table}' LIMIT 3;")
                sample_rows = cursor.fetchall()
                
                tables[table] = {
                    "columns": [col[1] for col in columns],
                    "types": [col[2] for col in columns],
                    "sample_data": sample_rows,
                    "row_count": self._get_row_count(table)
                }
            
            return tables
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}", exc_info=True)
            return {}
    
    def _get_row_count(self, table_name: str) -> int:
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting rows: {str(e)}", exc_info=True)
            return 0
    
    def translate_to_sql(self, natural_language_query: str) -> str:
        start_time = time.time()
        logger.info(f"Translating query to SQL: {natural_language_query}")
    
        tables_info = self.get_table_info()
        schema_desc = []
        for table_name, info in tables_info.items():
            col_desc = []
            for col, dtype in zip(info["columns"], info["types"]):
                col_desc.append(f"{col} ({dtype})")
            
            schema_desc.append(f"Table: {table_name}")
            schema_desc.append(f"Columns: {', '.join(col_desc)}")
            schema_desc.append(f"Row count: {info['row_count']}")
            
            if info['sample_data']:
                sample_rows = []
                for row in info['sample_data'][:2]: 
                    sample_rows.append(", ".join([str(val) for val in row]))
                schema_desc.append(f"Sample data: [{' | '.join(sample_rows)}]")
            
            schema_desc.append("") 
        
        schema_text = "\n".join(schema_desc)
        
        prompt = f"""
        You are an expert SQL developer. Convert this natural language query to a valid SQLite SQL query.
        
        Database Schema:
        {schema_text}
        
        Natural Language Query: {natural_language_query}
        
        Rules:
        1. Return ONLY the SQL query, nothing else.
        2. Use valid SQLite syntax.
        3. Use double quotes for table and column names.
        4. Make sure to handle JOINs appropriately if needed.
        5. If the query cannot be translated, return "ERROR: " followed by a brief explanation.
        
        SQL Query:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=CONFIG["groq"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]  
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            if sql_query.startswith("ERROR:"):
                logger.warning(f"LLM couldn't translate query: {sql_query}")
                return sql_query
                
            end_time = time.time()
            logger.info(f"SQL Translation Time: {end_time - start_time:.2f} sec")
            logger.info(f"Generated SQL: {sql_query}")
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error translating to SQL: {str(e)}", exc_info=True)
            return f"ERROR: Failed to translate query - {str(e)}"
    
    def execute_sql_query(self, sql_query: str) -> Tuple[List[Dict[str, Any]], str]:
        start_time = time.time()
        logger.info(f"Executing SQL query: {sql_query}")
        
        if sql_query.startswith("ERROR:"):
            return [], sql_query
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
        
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
        
            results = []
            for row in rows:
                result_dict = {}
                for i, col in enumerate(column_names):
                    result_dict[col] = row[i]
                results.append(result_dict)
            
            end_time = time.time()
            logger.info(f"SQL Execution Time: {end_time - start_time:.2f} sec")
            logger.info(f"Query returned {len(results)} rows")
            
            return results, ""
            
        except Exception as e:
            error_message = f"ERROR: {str(e)}"
            logger.error(f"Error executing SQL query: {str(e)}", exc_info=True)
            return [], error_message
    
    def format_result_with_llm(self, natural_language_query: str, sql_query: str, results: List[Dict[str, Any]], error: str = "") -> str:
        start_time = time.time()
        
        if error:
            prompt = f"""
            You are a helpful AI assistant with SQL expertise.
            
            User's question: {natural_language_query}
            
            I tried to convert this to SQL: {sql_query}
            
            But encountered this error: {error}
            
            Please explain what went wrong in a helpful way and suggest corrections.
            """
        elif not results:
            prompt = f"""
            You are a helpful AI assistant with SQL expertise.
            
            User's question: {natural_language_query}
            
            I converted this to SQL: {sql_query}
            
            The query executed successfully but returned no results.
            
            Please explain this in a natural way to the user.
            """
        else:
            results_str = json.dumps(results[:5], indent=2) 
            total_results = len(results)
            
            prompt = f"""
            You are a helpful AI assistant with SQL expertise.
            
            User's question: {natural_language_query}
            
            I converted this to SQL: {sql_query}
            
            The query returned {total_results} results. Here are the first few:
            {results_str}
            
            Please answer the user's original question in a natural way based on these results.
            Include key insights, numbers, and trends if applicable.
            If there are more than 5 results, mention the total count.
            Ensure your response is helpful and directly addresses the user's question.
            """
        
        try:
            response = self.client.chat.completions.create(
                model=CONFIG["groq"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            formatted_response = response.choices[0].message.content.strip()
            
            end_time = time.time()
            logger.info(f"Response Formatting Time: {end_time - start_time:.2f} sec")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}", exc_info=True)
            if error:
                return f"Error in SQL query: {error}"
            elif not results:
                return "The query executed successfully but didn't return any results."
            else:
                return f"Found {len(results)} results, but couldn't format the response."
    
    def process_natural_language_query(self, query: str) -> str:
        sql_query = self.translate_to_sql(query)
        results, error = self.execute_sql_query(sql_query)
        response = self.format_result_with_llm(query, sql_query, results, error)
        return response
    
    def close(self):
        if self.conn:
            self.conn.close()

