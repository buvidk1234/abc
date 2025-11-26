import operator
from typing import TypedDict, Literal, Callable, Any, Annotated, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.constants import END
from langgraph.graph import MessagesState, add_messages
from langgraph.types import Command

from src.agent.model import model as llm

class State(TypedDict):
    # messages: 保存对话历史
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # next:
    next: str

def make_supervisor_node(members: list[str],llm: BaseChatModel=llm) -> Callable[[State], Command[Any]]:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH. respond json format"
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router, method="function_calling").invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node



