import os
import shutil

def clear_data_folder():
    data_path = "data"
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(data_path)
    print('"data" has been cleared.')

def delete_faiss_index_folder():
    faiss_path = "faiss_index"
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
        print('"faiss_index" has been deleted.')
    else:
        print('"faiss_index" folder does not exist.')

def delete_excel_data_sqlite():
    sqlite_path = "excel_data.sqlite"
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
        print('"excel_data.sqlite" has been deleted.')
    else:
        print('"excel_data.sqlite" does not exist.')

def cleanup_all():
    clear_data_folder()
    delete_faiss_index_folder()
    delete_excel_data_sqlite()

if __name__ == "__main__":
    cleanup_all()
