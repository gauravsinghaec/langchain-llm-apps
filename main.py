#!/usr/bin/env python

import os
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langserve import add_routes

# 1 Simple app using Lang
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
prompt = ChatPromptTemplate.from_messages([
    # ("system", "You are the famous Indian story teller Munsi Premchand"),
    ("system", "You are the world database, the keeper of all the knowledge."),
    ("user", "{input}")
])

chain = prompt | llm 

# 2. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    chain
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)