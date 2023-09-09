#coded in PyCharm
#.env file OPENAI_API_KEY="YOUR KEY"
#run code using the command from command line / terminal "streamlit run PDFChatBotKv1.py"
#The code creates a Streamlit web application that allows users to upload one or multiple PDF files.
# Users can then ask questions related to the content of the uploaded PDFs, and the application will provide answers using OpenAI's model.
#The response is displayed on the Streamlit interface, and the chat history is updated.
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ... [previous imports]

# Global variables to store chat history and multiple documents
chat_history = []
pdf_data = {}

# sidebar contents
with st.sidebar:
    st.title('OpenAI LLM based chatbot on your PDF file')
    st.markdown('''
    ## The application is created using:

    - [OpenAI](https://openai.com/)
    - [Langchain](https://docs.langchain.com/docs/) 
    - [Streamlit](https://streamlit.io/)
    - [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
    - [PyPDF2](https://pypdf2.readthedocs.io/)

    ## About me:
    - [Linkedin](https://www.linkedin.com/in/vkrisvasan/)

    ''')

# ... [previous imports and sidebar code]

def main():
    # Initialize session state variables if they don't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDF files")

    # upload your pdf files
    pdfs = st.file_uploader("Upload your PDF files", type='pdf', accept_multiple_files=True)

    combined_text = ""
    store_name = "combined_pdf_context"

    if pdfs:
        for single_pdf in pdfs:
            pdf_reader = PdfReader(single_pdf)
            for page in pdf_reader.pages:
                combined_text += page.extract_text()

    # If there's combined text from the uploaded PDFs
    if combined_text:
        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=combined_text)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
        else:
            # embedding (Openai methods)
            embeddings = OpenAIEmbeddings()

            # Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

    # Accept user questions/query
    query = st.text_input("Ask a question related to your uploaded pdf files")

    if query and combined_text:
        docs = vectorstore.similarity_search(query=query, k=3)

        # openai rank lnv process
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

        # Update chat history in session state
        st.session_state.chat_history.append({"question": query, "answer": response})

    # Display chat history from session state
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"Q: {chat['question']}")
        st.write(f"A: {chat['answer']}")

if __name__ == "__main__":
    main()
