from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader, Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from typing import List, Any

def load_all_documents(dir: str) -> List[Any]:

    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """

    dir_path = Path(dir).resolve()
    print(f"Data Path: {dir_path}")
    documents = []

    # load pdf files
    pdf_files = list(dir_path.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"Loading PDF: {str(pdf_file)}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_documents = loader.load()
            print(f"Loaded {len(pdf_documents)} PDF docs from {pdf_file}")
            documents.extend(pdf_documents)
        except Exception as e:
            print(f"Failed to load PDF {pdf_file}: {e}")


    # load text files
    txt_files = list(dir_path.glob("**/*.txt"))
    print(f"Found {len(txt_files)} text files: {[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"Loading text file: {str(txt_file)}")
        try:
            loader = TextLoader(str(txt_file))
            txt_documents = loader.load()
            print(f"Loaded {len(txt_documents)} PDF docs from {txt_file}")
            documents.extend(txt_documents)
        except Exception as e:
            print(f"Failed to load Text file {txt_file}: {e}")

    
    # load CSV files
    csv_files = list(dir_path.glob("**/*.csv"))
    print(f"Loaded {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")

    for csv_file in csv_files:
        print(f"Loading CSV: {str(csv_file)}")
        try:
            loader = CSVLoader(str(csv_file))
            csv_documents = loader.load()
            print(f"Loaded {len(csv_documents)} CSV files from {csv_file}")
            documents.extend(csv_documents)
        except Exception as e:
            print(f"Failed to load CSV {csv_file}: {e}")


    # load Excel files
    excel_files = list(dir_path.glob("**/*.xlsx"))
    print(f"Loaded {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")

    for excel_file in excel_files:
        print(f"Loading Excel file: {str(excel_file)}")
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            excel_documents = loader.load()
            print(f"Loaded {len(excel_documents)} Excel files from {excel_file}")
            documents.extend(excel_documents)
        except Exception as e:
            print(f"Failed to load Excel file {excel_file}: {e}")


    # load word files
    word_files = list(dir_path.glob("**/*.docx"))
    print(f"Loaded {len(word_files)} Word files: {[str(f) for f in word_files]}")

    for word_file in word_files:
        print(f"Loading Word file: {str(word_file)}")
        try:
            loader = Docx2txtLoader(str(word_file))
            word_documents = loader.load()
            print(f"Loaded {len(word_documents)} Word files from {word_file}")
            documents.extend(word_documents)
        except Exception as e:
            print(f"Failed to load Word file {word_file}: {e}")

    
    # load JSON files
    json_files = list(dir_path.glob("**/*.json"))
    print(f"Loaded {len(json_files)} JSON files: {[str(f) for f in json_files]}")

    for json_file in json_files:
        print(f"Loading JSON file: {str(json_file)}")
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[]",
                text_content=False
            )
            json_documents = loader.load()
            print(f"Loaded {len(json_documents)} JSON files from {json_file}")
            documents.extend(json_documents)
        except Exception as e:
            print(f"Failed to load JSON file {json_file}: {e}")


    # load SQL files
    sql_files = list(dir_path.glob("**/*.sql"))
    print(f"Loaded {len(sql_files)} SQL files: {[str(f) for f in sql_files]}")

    for sql_file in sql_files:
        print(f"Loading SQL file: {str(sql_file)}")
        try:
            loader = TextLoader(str(sql_file))
            sql_documents = loader.load()
            print(f"Loaded {len(sql_documents)} SQL files from {sql_file}")
            documents.extend(sql_documents)
        except Exception as e:
            print(f"Failed to load SQL file {sql_file}: {e}")

    print (f"Total loaded documents: {len(documents)}")
    return documents

if __name__ == "__main__" :
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document: ", docs[0] if docs else None)

