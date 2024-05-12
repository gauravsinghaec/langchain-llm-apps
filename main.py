#!/usr/bin/env python
import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# Load
loader = WebBaseLoader("https://finance.yahoo.com/news/accern-emerges-premier-nlp-leader-124500667.html")
docs = loader.load()
print(f"Total input docs: {len(docs)}")

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
print(f"Total chunks: {len(documents)}")

# Add to vectorDB
embeddings = OpenAIEmbeddings()
vector_db = None
# This load_local and save_local are optional, it is just to avoid calculating embedding 
# and lossing some of your open AI credits unnecessarily
try:
    # Optional
    vector_db = FAISS.load_local(folder_path="./database/faiss_db",embeddings=embeddings,index_name="gks-faiss-index",allow_dangerous_deserialization=True)
    print("found the faiss locally")
except Exception:
    print("Loading FAISS index to local")
    vector_db = FAISS.from_documents(documents, embeddings)
    # Optional
    vector_db.save_local(folder_path="./database/faiss_db", index_name="gks-faiss-index")

retriever = vector_db.as_retriever()

retrieval_prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context,if you dont know the answer, please think outside of context then answer.:
    <context>
    {context}
    </context>
Question: {input}""")

# LLM
model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Parser
output_parser = StrOutputParser()

# # **** Method 1(Not Working) *******
# # Doc Chain: for passing a list of Documents to the model.
# combine_docs_chain = create_stuff_documents_chain(model, retrieval_prompt)
# # RAG Chain
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain) 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# # **** Method 2 *******
setup_and_retrieval = RunnableParallel(
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
)
retrieval_chain = setup_and_retrieval | retrieval_prompt | model | output_parser


# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route
add_routes(
    app,
    retrieval_chain,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)