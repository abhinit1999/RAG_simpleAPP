import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_KEY")
model = ChatGroq(model='gemma2-9b-it',groq_api_key=groq_api_key)


st.title("RAG application By Verto(Abhinit)")

uploade_file = st.file_uploader("Upload you PDF", type=['PDF'])

if uploade_file:
    with st.spinner("processing document...."):
        # saving file locally
        temp_path2 = os.path.join("temp", uploade_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path2, "wb") as f:
            f.write(uploade_file.getbuffer())
        # load and spliting the document
        loader = PyPDFLoader(temp_path2)
        docs = loader.load()

        spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splited_chunks = spliter.split_documents(docs)

        # embedding and storing into FAISS
        embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        DB = FAISS.from_documents(splited_chunks,embeddings)
        # creating retriever 
        retriever = DB.as_retriever(search_type="similarity",search_kwargs={'k':4})

        st.success("document has been loaded successfully.. Start asking questions....")

# creating the prompt
prompt = PromptTemplate(
    template='''

            You are a helpful assistant. Your job is to provide transcript context based on the provided context only.
            If context is insufficient, just say "Sorry I don't know about it
            context:{context}
            question:{question}
        ''',
        input_variables=['context','question']
)

        
# user question and Input button

st.subheader("Ask the question about the document...")
input_question = st.text_input("Enter you question here")
ask_button = st.button("ask")

# handiling the ask button
if input_question and ask_button:
    with st.spinner("Generating answer...."):
        retrivew_docs = retriever.invoke(input_question)
        final_prompt = prompt.invoke({'context':retrivew_docs,'question':input_question})
        answer = model.invoke(final_prompt)
        st.markdown(f"Answer:-> {answer.content}")