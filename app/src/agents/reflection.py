from typing import TypedDict, Annotated, List

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

from langchain_openai import ChatOpenAI

from src.agent.model import model as llm


# 绑定一个带固定 system prompt 的反思模型

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# -------------------------
# 1. State 定义
# -------------------------

class State(TypedDict):
    """
    Graph 对外暴露的状态：
    - messages: 给前端 Agent Chat UI 用来渲染的主对话消息
    - process: 仅在后端使用，用来记录“生成 + 反思”的内部链路
    """
    messages: Annotated[List[BaseMessage], add_messages]
    process: Annotated[List[BaseMessage], add_messages]


# -------------------------
# 2. 节点定义
# -------------------------

async def generation_node(state: State) -> dict:
    """
    生成节点：
    - 如果是第一次调用：用 messages（用户输入等）作为上下文生成初稿
    - 之后的调用：用 process （前面所有生成+反思）作为上下文继续写作
    - 只把“生成结果”追加进 process
    - 是否要让用户实时看到这一步，由是否往 messages 写决定
    """
    process = state.get("process", [])

    if not process:
        # 第一次
        llm_reply: BaseMessage = await llm.ainvoke(state["messages"])
    else:
        # 后续：用 process 作为上下文（包括前面的生成和反思）
        llm_reply: BaseMessage = await llm.ainvoke(state["messages"] + process[-2:])

    # 内部链路：记录到 process
    updates: dict = {"process": [llm_reply]}

    return updates


async def reflection_node(state: State) -> dict:
    """
    反思节点：
    - 对 process 中最近一次生成(last_ai)做点评
    - 把点评结果（教师反馈）追加进 process
    - 可选择是否展示给前端（写入 messages）
    """
    process = state["process"]

    # 最近一次生成结果（假设是 AIMessage 或 BaseMessage）
    last_ai: BaseMessage = process[-1]

    original_user: BaseMessage = state["messages"][0]

    reflection_prompt: List[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a teacher grading an essay submission. "
                "Generate critique and recommendations for the user's submission. "
                "Provide detailed recommendations, including requests for length, "
                "depth, style, etc."
            )
        ),
        original_user,
        HumanMessage(content=last_ai.content),
    ]

    res: BaseMessage = await llm.ainvoke(reflection_prompt)

    feedback = HumanMessage(content=res.content)

    return {"process": [feedback]}


def should_continue(state: State) -> str:

    # 例如：process 长度达到一定阈值后结束
    # 这里的长度是“内部步骤总数”，你可以按自己的节奏调整
    # h a b a b a
    if len(state["process"]) >= 4:
        return "final"
    return "reflect"


async def final_node(state: State) -> dict:
    """
    最终汇总节点：
    - 根据 process 中的所有生成/反馈合成一个最终回答
    - 只追加一条 AIMessage 到 messages，不重写历史
    """

    body_messages = state["process"]

    combined_texts: List[str] = []
    for msg in body_messages:
        combined_texts.append(msg.content)

    combined = "\n\n".join(combined_texts)

    final_ai = AIMessage(content=combined)

    updates: dict = {
        "messages": [final_ai],
        "process": [],
    }

    return updates


# -------------------------
# 3. 构建并编译 Graph
# -------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(State)

    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("final", final_node)

    builder.add_edge(START, "generate")

    builder.add_conditional_edges(
        "generate",
        should_continue,
        {"final": "final", "reflect": "reflect"}
    )

    builder.add_edge("reflect", "generate")

    builder.add_edge("final", END)

    graph = builder.compile()
    return graph


graph = build_graph()

config = {"configurable": {"thread_id": "1"}}


# async for event in graph.astream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Generate an essay on the topicality of The Little Prince and its message in modern life"
#             )
#         ],
#     },
#     config,
# ):
#     print(event)
#     print("---")


async def main():
    initial_state: State = {
        "messages": [
            HumanMessage(content="写一段话，描述一个落魄的中年人"),
        ]
    }
    res = await graph.ainvoke(initial_state, config=config)
    print(res["messages"][-1].content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())