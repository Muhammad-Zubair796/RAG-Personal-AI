from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    Docx2txtLoader, UnstructuredExcelLoader, JSONLoader
)

def load_all_documents(data_dir: str) -> List[Any]:
    data_path = Path(data_dir).resolve()
    documents = []
    
    # Define which loader to use for each file type
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    for ext, loader_cls in loaders.items():
        files = list(data_path.glob(f"**/*{ext}"))
        for file_path in files:
            try:
                print(f"[DEBUG] Loading: {file_path.name}")
                loader = loader_cls(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")

    # Special handling for JSON as it usually needs a schema
    for json_file in data_path.glob('**/*.json'):
        try:
            loader = JSONLoader(str(json_file), jq_schema=".[]", text_content=False)
            documents.extend(loader.load())
        except:
            pass 

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents