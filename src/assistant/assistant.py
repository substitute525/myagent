import copy
import json
import traceback
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Iterator, Tuple

import json5
import qwen_agent.tools
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk, SystemMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.tools import BaseTool
from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools import MCPManager, TOOL_REGISTRY
from qwen_agent.tools.base import ToolServiceError
from qwen_agent.tools.simple_doc_parser import DocParserError

from .base import get_chat_model
from ..log import logger
from ..tools import get_qwen_cls
from ..utils.utils import merge_generate_cfgs


class BaseAssistant(ABC):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 name: Optional[str] = None,
                 system: Optional[str] = None,
                 description: Optional[str] = None, **kwargs):
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.name = name
        self.system = system
        self.description = description
        self.extra_generate_cfg: dict = {}
        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

    def _init_tool(self, tool: Union[str, Dict, BaseTool, qwen_agent.tools.BaseTool]):
        if isinstance(tool, qwen_agent.tools.BaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        elif isinstance(tool, BaseTool):
            ToolCls, tool_name = get_qwen_cls(tool)
            self.function_map[tool_name] = ToolCls()
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            # 使用qwen提供的初始化工具
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                if tool_name in self.function_map:
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                self.function_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def simulate_response_completion_with_chat(messages: List[BaseMessage]) -> List[BaseMessage]:
        if messages and (messages[-1].type == 'assistant'):
            assert (len(messages) > 1) and (messages[-2].type == 'user')
            assert messages[-1].function_call is None
            usr = messages[-2].content
            bot = messages[-1].content
            sep = '\n\n'
            if isinstance(usr, str) and isinstance(bot, str):
                usr = usr + sep + bot
            elif isinstance(usr, list) and isinstance(bot, list):
                usr = usr + [ContentItem(text=sep)] + bot
            else:
                raise NotImplementedError
            text_to_complete = copy.deepcopy(messages[-2])
            text_to_complete.content = usr
            messages = messages[:-2] + [text_to_complete]
        return messages

    def run_nonstream(self, messages: List[BaseMessage], **kwargs) -> List[BaseMessage]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: Union[str, List[BaseMessage]], **kwargs) -> Iterator[List[BaseMessage]]:
        if isinstance(messages, str):
            messages = [HumanMessage(messages)]
        messages = copy.deepcopy(messages)
        if self.system:
            if messages[0].type != 'system':
                messages.insert(0, SystemMessage(self.system))
            else:
                messages[0].content = self.system + '\n\n' + messages[0].content

        if 'tool_names' in kwargs:
            messages = self._preprocess_messages(messages=messages,
                                                 functions=[self.function_map.get(func_name).function for func_name in kwargs.get('tool_names') if func_name in self.function_map])
        else:
            messages = self._preprocess_messages(messages=messages,
                                             functions=[func.function for func in self.function_map.values()])
        for rsp in self._run(messages=messages, **kwargs):
            yield [x for x in rsp]

    def _fncall_prompt(self) -> str:
        return """# 工具相关
## 提供给你的工具

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

## 使用工具

You may call one or more functions to assist with the user query.
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
You are allowed to call functions multiple times across multiple turns if needed.
``` 调用工具模板
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
```
"""
    #     For code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:
    #     <tool_call>
    #     {{"name": <function-name>, "arguments": {{"code": ""}}}}
    #     <code>
    #     Here is the code.
    #     </code>
    #     </tool_call>

    def _preprocess_messages(self, messages: List[BaseMessage], lang: str = 'zh', generate_cfg: dict = None,
                             functions: Optional[List[Dict]] = None, ) -> List[BaseMessage]:
        new_messages = []
        fn_call_msg = ''
        for msg in copy.deepcopy(messages):
            role, content = msg.type, msg.content
            if role in ('system', 'human'):
                new_messages.append(msg)
            elif role == 'ai':
                fn_call = msg.tool_calls
                if fn_call:
                    if 'code_interpreter' not in fn_call.name:
                        args = fn_call.args
                        fc = {'name': fn_call.name, 'arguments': args}
                        fc = json.dumps(fc, ensure_ascii=False)
                        fc = f'<tool_call>\n{fc}\n</tool_call>'
                    else:
                        para = json5.loads(fn_call.arguments)
                        code = para['code']
                        para['code'] = ''
                        fc = {'name': fn_call.name, 'arguments': para}
                        fc = json.dumps(fc, ensure_ascii=False)
                        fc = f'<tool_call>\n{fc}\n<code>\n{code}\n</code>\n</tool_call>'
                    fn_call_msg = f"{fn_call_msg}\n\n{fc}"
            elif role == 'tool':
                fc = f'<tool_response>\n{content}\n</tool_response>'
                if messages[-1].role == 'human':
                    messages[-1].content = f'{messages[-1].content}\n{fc}'
                    messages[-1].content.extend(content)
                else:
                    new_messages.append(HumanMessage(content=content))

        if functions:
            tool_descs = [{'type': 'function', 'function': f} for f in functions]
            tool_descs = '\n'.join([json.dumps(f, ensure_ascii=False) for f in tool_descs])
            tools_prompt = self._fncall_prompt().format(tool_descs=tool_descs)
            if messages and messages[0].type == 'system':
                if "{tools}" in new_messages[0].content:
                    new_messages[0].content = new_messages[0].content.format(tools=tools_prompt)
                else:
                    new_messages[0].content = new_messages[0].content + ('\n\n' + tools_prompt)
            else:
                new_messages = [SystemMessage(content=tools_prompt)] + messages
        return new_messages

    @abstractmethod
    def _run(self, messages: List[BaseMessage], lang: str = 'zh', **kwargs) -> Iterator[List[BaseMessage]]:
        raise NotImplementedError

    def _call_llm(
            self,
            messages: List[BaseMessage],
            stream: bool = True,
            extra_generate_cfg: Optional[dict] = None,
            stream_usage: bool = True,
    ) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        if stream:
            return self.llm.stream(input=messages,
                                   stream_usage=stream_usage,
                                   config=merge_generate_cfgs(
                                       base_generate_cfg=self.extra_generate_cfg,
                                       new_generate_cfg=extra_generate_cfg
                                   ))
        else:
            return self.llm.invoke(input=messages,
                                   config=merge_generate_cfgs(
                                       base_generate_cfg=self.extra_generate_cfg,
                                       new_generate_cfg=extra_generate_cfg,
                                   ))

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.function_map[tool_name]
        try:
            if isinstance(tool_args, str):
                tool_args = json5.loads(tool_args) if tool_args else {}
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def extract_fn(self, text: str):
        fn_name, fn_args = '', ''
        fn_name_s = '"name": "'
        fn_name_e = '", "'
        fn_args_s = '"arguments": '
        i = text.find(fn_name_s)
        k = text.find(fn_args_s)
        if i > 0:
            _text = text[i + len(fn_name_s):]
            j = _text.find(fn_name_e)
            if j > -1:
                fn_name = _text[:j]
        if k > 0:
            fn_args = text[k + len(fn_args_s):]
        fn_args = fn_args.strip()
        if len(fn_args) > 2:
            fn_args = fn_args[:-1]
        else:
            fn_args = ''
        return fn_name, fn_args

    def _postprocess_messages(self, message: AIMessage) -> List[AIMessage]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        new_content = ''
        tool_call_list = message.content.split('<tool_call>')
        pre_thought = tool_call_list[0]
        if pre_thought.strip():
            new_content = new_content + pre_thought
        new_messages = []
        new_messages.append(AIMessage(
            content=new_content,
        ))  # split thought and function call
        toolCalls = []
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                fn_name, fn_args = self.extract_fn(text=txt)
                if fn_name:  # need to call function
                    # TODO: process incomplete tool-call messages
                    toolCalls.append(ToolCall(
                        id=fn_name,
                        name=fn_name,
                        # 此处json不完整，无法赋值
                        args={}
                    ))
                continue

            one_tool_call_txt = txt.split('</tool_call>')
            fn = None
            try:
                fn = json5.loads(one_tool_call_txt[0].strip())
            except Exception:
                logger.warning(f'Invalid json tool-calling arguments, txt:[{txt}]')
                fn_name, fn_args = self.extract_fn(text=one_tool_call_txt[0].strip())
                toolCalls.append(ToolCall(
                    id=fn_name,
                    name=fn_name,
                    args=json.loads(fn_args)
                ))
            if fn:
                toolCalls.append(ToolCall(
                    id=fn['name'],
                    name=fn['name'],
                    args=fn['arguments']
                ))
        if toolCalls:
            new_messages.append(AIMessage(content='', tool_calls=toolCalls))
        return new_messages
