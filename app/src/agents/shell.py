"""使用 Shell 工具的代理示例（Linux 环境）"""



from langchain_openai import ChatOpenAI

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
)

from src.agent.model import model


# 创建代理，注册 Shell 工具
agent = create_agent(
    model=model,
    tools=[],   # 可以加其他工具，这里只用 Shell
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",          # Linux 路径，确保目录存在
            execution_policy=HostExecutionPolicy(),  # 在宿主机直接执行
            shell_command="/bin/bash",            # Linux 默认 shell
            startup_commands=[
                "echo 'Shell session started on Linux'"
            ],
            shutdown_commands=[
                "echo 'Shell session shutting down'"
            ],
        ),
    ],
)

if __name__ == "__main__":
    # 用户消息：创建文件并读取内容
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a file named test.txt with content 'Hello, World!' and then read its content"
                }
            ]
        }
    )

    # 输出结果
    pretty = getattr(result, "pretty_print", None)
    if callable(pretty):
        pretty()
    else:
        print(result)


