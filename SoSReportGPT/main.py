from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter




def read_log_files(directory):
    text = ""
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out only the log files
    log_files = [file for file in files if file.endswith('.log')]

    # Iterate over each log file and read its contents
    for log_file in log_files:
        file_path = os.path.join(directory, log_file)
        with open(file_path, 'r') as file:
            print(f"Reading log file: {file_path}")
            # Process the content of the log file here, for example:
            for line in file:
                print(line.strip())  # Print each line of the log file
                text = text + line.strip()
    return text
# Example usage:
log_data=read_log_files('./data/')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=20
)
documents = text_splitter.split_text(log_data)
print(documents)
print(f'You have {len(documents)} document(s) in your data folder.')



embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db = Chroma.from_texts(documents, embeddings, persist_directory="./chroma_db/")
db.persist()