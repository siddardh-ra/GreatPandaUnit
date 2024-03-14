# Import necessary libraries.
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceEndpoint
import requests
import os

# Set web page title and icon.
st.set_page_config(
    page_title="Chat with Reports",
    page_icon=":robot:"
)

st.title('SoS Report Bot 💬 ')

st.markdown(
    """
    This is a chatbot for Q & A for SoS Reports and logs
    """
)

# Define a function to get user input.
def get_input_text():
    input_text = st.text_input("Ask a question about your log:")
    return input_text

# Define to variables to use "sentence-transformers/all-MiniLM-L6-v2" embedding model from HuggingFace.
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define the Chroma vector store and function to generate embeddings.
db = Chroma(persist_directory="./chroma_db/", embedding_function=embeddings)

# Get user input.
user_input = get_input_text()

# Initialize the Azure OpenAI ChatGPT model.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def format_llama_prompt(user_prompt):
    prompt = """\
<s>[INST] 

{user_prompt}[/INST]\
"""
    return prompt.format(user_prompt=user_prompt)



# Define the function to get the response.
if user_input:


    URL = "https://llama-2-7b-chat-perfconf-hackathon.apps.dripberg-dgx2.rdu3.labs.perfscale.redhat.com"

    endpoint = "/generate"

    headers = {
        "Content-Type": "application/json"
    }

    # Perform similarity search for the user input.
    docs = db.similarity_search(user_input)


    # res_string = docs[0].page_content
    # metadata = docs[0].metadata
    #
    # res_string = res_string.replace("\n"," ")
    # res_string=res_string + f'The above content was taken from the file {metadata["source"]}and page number is {metadata["page"]}'

    res_string = docs[0]

    data = {
        "inputs": format_llama_prompt(res_string),
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.9,  # Just an example
            "repetition_penalty": 1.03,  # Just an example
            "details": False
        }
    }

    response = requests.post(f"{URL}{endpoint}", headers=headers, json=data, verify=False)
    st.write(response.json().get("generated_text"))
