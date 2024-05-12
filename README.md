App built with Langchain Framework

# Playing with Langchain Framework - From a Simple LLM APP to building with RAG and Agent

## Navigations

[Langchain APP](#app-built-with-langchain-framework)

- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Startup Guide](#project-startup-guide)
  - [Installation](#installation)
  - [Testing](#testing)
- [Resources](#resources)

## Folder Structure

- tools/
  -May Contains tools for the agents
- main.py
- pyproject.toml
- poetry.lock
- .gitignore
- README.md

## Prerequisites

**python 3.11**

## Startup Guide

- Clone the repo and
- Install the dependencies to setup your system to run the app
  ```
    poetry install
  ```
- Run the app
  ```
    poetry run python main.py
  ```
- Checkout the respective branch as you follow along the article

- To install new package
  ```
    poetry add <package name>
  ```

### Testing

- To test
  ```
    poetry run pytest
  ```

## Resources

- [Langchain Quickstart](https://python.langchain.com/v0.1/docs/get_started/quickstart/)
- [Poetry package tool for python](https://python-poetry.org/docs/)
