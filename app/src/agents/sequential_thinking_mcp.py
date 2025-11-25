from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


from src.agent.model import model

client = MultiServerMCPClient(
    {
        "sequential-thinking": {
          "transport": "stdio",
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-sequential-thinking"
          ]
        }
    }
)

async def main():
    tools = await client.get_tools()
    agent = create_agent(
        model=model,
        tools=tools
    )
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "写一份报告关于rag系统的架构设计，包括数据处理、检索和重排序模块的实现细节。"}]}
    )
    print(math_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())