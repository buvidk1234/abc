"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Annotated, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from src.agent.model import model

class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    result = await model.ainvoke(state.messages)

    return {
        "messages": [result]
    }


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)
    .add_edge("__start__", "call_model")
    .compile(name="New Graph")
)


async def main() -> None:
    """Main function to run the graph."""
    result = await graph.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        },
        context={
            "my_configurable_param": "example_value"
        }
    )
    print(result)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
