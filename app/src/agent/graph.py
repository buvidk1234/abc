from operator import add
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


from src.agent.model import model, embeddings

"""
短期记忆：raw
中期记忆：summary
长期记忆：retrieval
"""

# TODO: 持久化
# 短中期记忆
checkpointer = InMemorySaver()
# 长期记忆
# 存储全量会话vector or 保存重要信息
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str | None

def update_long_term_memory(state: GraphState, config: RunnableConfig, *, store: BaseStore):

    # 获取线程 ID 作为用户标识
    thread_id = config["configurable"]["thread_id"]
    # 命名空间
    namespace = (thread_id, "memories")

    # 分析对话并创建新记忆（此处为示例，实际应用中应有更复杂的逻辑）
    user_message = state["messages"][-2].content
    memory = f"记住用户说过: {user_message}"

    # 创建新记忆 ID
    import uuid
    memory_id = str(uuid.uuid4())

    # 存储新记忆
    store.put(namespace, memory_id, {"memory": memory})

    print(f"Memory updated for thread {thread_id}: {memory}")

def summarize_conversation(state: GraphState):
    messages = state["messages"]
    # 检查 Token
    current_tokens = model.get_num_tokens_from_messages(messages)

    if current_tokens <= 4000 and len(messages) <= 20:
        return None

    summary = state.get("summary", "")
    messages = state["messages"]

    # 如果有旧摘要，加进去一起总结
    if summary:
        summary_message = (
            f"这是之前的对话摘要: {summary}\n\n"
            "请将上面的摘要和以下新的对话内容合并，生成一个新的摘要。"
        )
    else:
        summary_message = "请将以下对话内容总结成一段简短的摘要。"

    # 调用 LLM 生成新摘要
    messages_to_summarize = [HumanMessage(content=summary_message)] + messages
    response = model.invoke(messages_to_summarize)
    new_summary = response.content

    # --- 关键步骤：删除旧消息 ---
    current_tokens = model.get_num_tokens_from_messages(messages[-10:])
    if current_tokens >= 2000:
        messages_to_delete = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    else:
        messages_to_delete = [RemoveMessage(id=m.id) for m in messages[:-10]]

    return {
        "summary": new_summary,
        "messages": messages_to_delete  # 这里返回的是删除指令
    }

from src.agent.tools import solve_problem
llm = create_agent(
    model=model,
    tools=[solve_problem],
    system_prompt="You chat as a friend, and if you encounter complex or knowledge based questions, use the 'backing' tool to help you answer the question."
)

def call_model(state: GraphState, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    # Namespace the memory
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = store.search(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # ... Use memories in the model call0
    res = llm.invoke({"messages": [SystemMessage(state.get("summary","")),*state["messages"], SystemMessage(info)]})
    return {"messages": [res["messages"][-1]]}

builder = StateGraph(GraphState)
builder.add_node("update_memory", update_long_term_memory)
builder.add_node("call_model", call_model)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "summarize_conversation")
builder.add_edge("call_model", "update_memory")
builder.add_edge("update_memory", END)
builder.add_edge("summarize_conversation", END)
# Receptionist
# chat_agent = builder.compile(checkpointer=checkpointer, store=store)
chat_agent = builder.compile()

if __name__ == "__main__":
    config = RunnableConfig(
        configurable={"thread_id": "1", "user_id": "1"}
    )
    res = chat_agent.invoke(
        {
        "messages": [HumanMessage(content="hi! i am Bob")]
        },
        config=config
    )
    print(res)
    res = chat_agent.invoke(
        {
        "messages": [HumanMessage(content="what is my name?")]
        },
        config=config
    )
    print(res)