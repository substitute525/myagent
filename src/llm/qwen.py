# 不自动调用工具，由外部统一调用
import copy
from typing import List, Iterator

from qwen_agent import Agent
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from qwen_agent.utils.output_beautify import typewriter_print
from src.tools import qwen_adaptation_init

# langchain tool适配qwen-agent
qwen_adaptation_init()


class LocalQwenAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        response = []
        extra_generate_cfg = {'lang': lang}
        output_stream = self._call_llm(messages=messages,
                                       functions=[func.function for func in self.function_map.values()],
                                       extra_generate_cfg=extra_generate_cfg)
        output: List[Message] = []
        for output in output_stream:
            if output:
                yield response + output
        if output:
            response.extend(output)
        yield response


# 步骤 2：配置您所使用的 LLM。
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    # 'model': 'qwen-max-latest',
    # 'model_type': 'qwen_dashscope',
    # 'api_key': 'YOUR_DASHSCOPE_API_KEY',
    # 如果这里没有设置 'api_key'，它将读取 `DASHSCOPE_API_KEY` 环境变量。

    # 使用与 OpenAI API 兼容的模型服务，例如 vLLM 或 Ollama：
    'model': 'qwen',
    'model_server': 'http://localhost:8000/v1',  # base_url，也称为 api_base
    'api_key': 'EMPTY',

    # （可选） LLM 的超参数：
    'generate_cfg': {
        'top_p': 0.8,
        'extra_body': {
            'chat_template_kwargs': {'enable_thinking': True}
        },
    }
}

system_instruction = '''你需要协助用户完成命令执行。'''


tools = ['code_interpreter', 'human_assistance', 'execute_command', 'list_sessions', 'read_output', 'create_session', 'query_url', 'search_web']  # `code_interpreter` 是框架自带的工具，用于执行代码。
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                )
llm = LocalQwenAgent(llm=llm_cfg,
                function_list=tools,)

def chat():
    # 步骤 4：作为聊天机器人运行智能体。
    messages = []  # 这里储存聊天历史。
    while True:
        # 例如，输入请求 "绘制一只狗并将其旋转 90 度"。
        query = input('\n用户请求: ')
        # 将用户请求添加到聊天历史。
        messages.append({'role': 'system', 'content': '''你需要协助用户完成命令执行。
你是一个用于软件开发支持的前置知识分析助手。
用户希望你分析当前任务中可能需要的基础前置信息，例如文件结构、配置、关键代码文件、依赖说明等。并通过调用工具获取必要的信息。
前置信息通常仅作为或许的基础知识背景或处理依据，因此需要提取的是必要的信息，而不需要提取详细、全面的信息。
若用户有明确要求调用工具完成什么命令的，则你可以直接完成。
*当你完成工具调用并收到结果后，你需要将这些信息整理为背景知识，以供下一个模型更好的做出规划。且你仅能输出背景知识，其他的一概禁止输出，包括但不限于如果需要进一步操作请告诉我！*
**注意**
- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具
- 大多数情况你都需要调用list_sessions工具列举出当前存活的session，以供后续使用'''})
        messages.append({'role': 'user', 'content': query})
        response = []
        response_plain_text = ''
        print('机器人回应:')
        nonstream = bot.run_nonstream(messages=messages)
        print(f"{nonstream}")
        print("---------------------")
        for response in bot.run(messages=messages):
            # 流式输出。
            response_plain_text = typewriter_print(response, response_plain_text)
        # 将机器人的回应添加到聊天历史。
        messages.extend(response)

if __name__ == '__main__':
    # result = TOOL_REGISTRY['human_assistance']().call(params='{"query":"当前任务是什么？"}')
    # print(f"结果{result}")
    chat()