from smolagents import (
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)

from _tools.initialize_model import Initialize_Client
from _tools.initialize_agents import Initialize_Agent
from _tools.hugging_face import model_download_tool


# Get the LLM client instance
client = Initialize_Client().initialize_HFApi_client()


# Basic Use case for code agentic run.
agent = Initialize_Agent(
    model=client,
    verbose=True,
).create_code_agent()

task = """
Could you give me the 118th number in the Fibonacci sequence?
"""


# # Basic use case for code agent run with additonal imports.
# agent = Initialize_Agent(
#     model=client,
#     verbose=True,
# ).create_code_agent(
#     authorized_imports=['requests', 'bs4'],
# )

# task = """
# Could you get me the title of the page at url 'https://huggingface.co/blog'?
# """


# # Basic use case for tool calling agent.
# agent = Initialize_Agent(
#     model=client,
#     tools = [DuckDuckGoSearchTool(), VisitWebpageTool()], # If not passed, the task will fail
#     verbose=True,
# ).create_tools_agent()

# task = """
# Could you get me the title of the page at url 'https://huggingface.co/blog'?
# """


# # Basic use case for manager agent.
# web_agent = Initialize_Agent(
#     model=client,
#     tools = [DuckDuckGoSearchTool(max_results=5)],
#     verbose=True,
# ).create_tools_agent()


# managed_web_agent = Initialize_Agent(
#     model=client,
#     verbose=True,
# ).create_managed_agent(
#     agent=web_agent,
#     name="web_search",
#     description="Runs web searches for you. Give it your query as an argument."
# )

# manager_agent = Initialize_Agent(
#     model=client,
#     verbose=True,
# ).create_code_agent(
#     managed_agents=[managed_web_agent],
#     authorized_imports=['requests', 'bs4'],
# )

# task = """
# List all CEOs of Apple for years 2000-2024.
# """


# Run the agent to fetch information about the given query
agent.run(task)
# manager_agent.run(task)

