from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def human_assistance(query: str) -> str:
    """当有问题时，可以请求人工协助"""
    human_response = interrupt({"query": query})
    s = input("请协助处理")
    return human_response["data"]