#coded in PyCharm
#.env file OPENAI_API_KEY="YOUR KEY"
#run code using the command from command line / terminal "streamlit run PDFChatBotKv1.py"
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

def main():
    st.header("Chat with your PDF file")

    # upload a your pdf file
    pdf = st.file_uploader("Upload your PDF file", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # store pdf name
        store_name = pdf.name[:-4]

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
        query = st.text_input("Ask question related to your uploaded pdf file")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)

            # openai rank lnv process
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()