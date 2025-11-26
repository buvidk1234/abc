from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel

from src.agent.model import model

tavily_tool = TavilySearch(max_results=5)

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()

class State(TypedDict):
    messages: list[str]
    next: str

class Result(BaseModel):
    """Result of the agent's work."""
    result: str

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

script_run_agent = create_agent(
    model=model,
    tools=[python_repl_tool],
    system_prompt="You can only generate charts. You are working with a researcher colleague.",
    response_format=Result
)