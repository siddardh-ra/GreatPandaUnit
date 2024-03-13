from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader

# pdfLoader = PyPDFDirectoryLoader('./data/')
# documents = []
# documents.extend(pdfLoader.load())


pdfLoader = PyPDFDirectoryLoader('./data/')
documents = []
documents.extend(pdfLoader.load())


print(f'You have {len(documents)} document(s) in your data folder.')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
print(len(documents))

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db/")
db.persist()