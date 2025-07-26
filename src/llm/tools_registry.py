from dotenv import load_dotenv

from src.tools import all_tools

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.types import interrupt

@tool
def human_assistance(query: str) -> str:
    """当有问题时，可以请求人工协助"""
    human_response = interrupt({"query": query})
    s = input("请协助处理")
    return human_response["data"]

# 创建LLM实例
llm = ChatOpenAI(
    model="deepseek-chat",  # e.g., "deepseek-coder", "deepseek-chat"
    base_url="https://api.deepseek.com/v1",  # Replace with DeepSeek's actual API base URL
    temperature=0.7
)
# phi_llm = get_llm()

# 绑定工具到LLM
llm_with_tools = llm.bind_tools(all_tools)
