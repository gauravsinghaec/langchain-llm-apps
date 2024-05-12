from langchain.agents import load_tools

built_in_tools = load_tools(
    ["graphql", "stackexchange"],
    graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index",
)
