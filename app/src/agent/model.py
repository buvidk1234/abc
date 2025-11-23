from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-flash",
    tiktoken_model_name="gpt-4o"
)
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
)