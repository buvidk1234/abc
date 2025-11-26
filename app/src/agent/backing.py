import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from src.agent.adaptive_rag import knowledge_agent
from src.agent.plan_and_execute import plan_execute_agent
from src.agent.script_run import script_run_agent
from src.agent.supervisor import make_supervisor_node


# ==========================================
# 1. 定义状态 (必须与 Supervisor 中的 State 兼容)
# ==========================================
class AgentState(TypedDict):
    # messages: 保存对话历史
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # next: 虽然 Command 自动处理了跳转，但保留这个字段用于状态追踪也是好的
    next: str


# ==========================================
# 2. 定义 节点封装
# ==========================================
supervisor_node = make_supervisor_node(members=["knowledge_agent", "plan_execute_agent","script_run_agent"])

def knowledge_node(state: AgentState):
    """知识库节点"""
    response = knowledge_agent.invoke({"question": state["messages"][-1].content})
    return {
        "messages": [AIMessage(content=str(response), name="knowledge_agent")]
    }


def plan_execute_node(state: AgentState):
    """规划执行节点"""
    response = plan_execute_agent.invoke({"input": state["messages"][-1]})
    return {
        "messages": [AIMessage(content=str(response), name="plan_execute_agent")]
    }


def script_run_node(state: AgentState):
    """代码执行节点"""
    response = script_run_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [AIMessage(content=str(response), name="script_run_agent")]
    }


# ==========================================
# 3. 构建图 (Build the Graph)
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("knowledge_agent", knowledge_node)
workflow.add_node("plan_execute_agent", plan_execute_node)
workflow.add_node("script_run_agent", script_run_node)

workflow.add_edge(START, "Supervisor")
workflow.add_edge("knowledge_agent", "Supervisor")
workflow.add_edge("plan_execute_agent", "Supervisor")
workflow.add_edge("script_run_agent", "Supervisor")

# ==========================================
# 4. 编译应用
# ==========================================
backing = workflow.compile()

# ==========================================
# 5. 测试运行
# ==========================================
if __name__ == "__main__":
    print("--- 🚀 System Initialized (Command Mode) ---")

    user_input = "请查一下2024年诺贝尔文学奖的获奖者是谁。"

    # 这里的 config 可以用于调试
    config = {"recursion_limit": 15}

    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }

    for output in backing.stream(inputs, config=config):
        for key, value in output.items():
            print(f"\n🔹 [Node]: {key}")

            # 打印 Supervisor 的路由决定
            if key == "Supervisor" and "next" in value:
                print(f"   👉 Routing to: {value['next']}")

            # 打印 Worker 的输出
            if "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, "name"):
                    print(f"   Worker ({last_msg.name}) says: {str(last_msg.content)[:100]}...")