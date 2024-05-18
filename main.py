#!/usr/bin/env python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.pydantic_v1 import BaseModel
from langserve import add_routes
from langchain_community.utilities import StackExchangeAPIWrapper
from tools.custom import char_count_tool
from tools.built_in import built_in_tools
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

stackexchange = StackExchangeAPIWrapper()

# 1. Load Retriever
loader = WebBaseLoader(
    "https://finance.yahoo.com/news/accern-emerges-premier-nlp-leader-124500667.html"
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "accern_news_search",
    "Search for information about Accern. For any questions about Accern's latest partnership,if you dont know the answer, please think outside of context then answer",
)


tools = [retriever_tool, char_count_tool] + built_in_tools

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.


class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
