from langchain_core.tools import tool

from src.agent.backing import backing

@tool
def solve_problem(question: str) -> str:
    """have strong power to complete complex and knowledgeable questions."""
    return backing.invoke({"messages": [{"role": "user", "content": question}]})

