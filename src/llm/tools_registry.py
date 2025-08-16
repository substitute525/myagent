from dotenv import load_dotenv

from src.assistant.qwen_assistant import QwenAssistant
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
# llm = ChatOpenAI(
#     model="deepseek-chat",  # e.g., "deepseek-coder", "deepseek-chat"
#     base_url="https://api.deepseek.com/v1",  # Replace with DeepSeek's actual API base URL
#     temperature=0.7
# )
# phi_llm = get_llm()

# 绑定工具到LLM
# llm_with_tools = llm.bind_tools(all_tools)

llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    # 'model': 'qwen-max-latest',
    # 'model_type': 'qwen_dashscope',
    # 'api_key': 'YOUR_DASHSCOPE_API_KEY',
    # 如果这里没有设置 'api_key'，它将读取 `DASHSCOPE_API_KEY` 环境变量。

    # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
    'model': 'qwen',
    'base_url': 'http://localhost:8000/v1',  # base_url，也称为 api_base
    'api_key': 'EMPTY',

    # （可选） LLM 的超参数：
    'generate_cfg': {
        'top_p': 0.8
    }
}
assistant = QwenAssistant(function_list=all_tools, llm=llm_cfg, name='qwen')

for msg_batch in assistant.run("list_session中，当前有哪些session可用?", stream=True, usage=True):
    # msg_batch 是 List[BaseMessage]
    for msg in msg_batch:
        # 打印 role/type 和内容
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            print(msg.content)
            print(msg.usage_metadata)
        else:
            print(msg.content, end='')