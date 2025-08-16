import json
from typing import List, Iterator, Optional, Dict

from langchain_core.messages import BaseMessage, FunctionMessage, SystemMessage, AIMessage, ToolMessage

from .assistant import BaseAssistant

FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'

FN_CALL_TEMPLATE_INFO_ZH = """# 工具

## 你拥有如下工具：

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

%s: 工具1的名称，必须是[{tool_names}]之一
%s: 工具1的输入
%s: 工具2的名称
%s: 工具2的输入
...
%s: 工具N的名称
%s: 工具N的输入
%s: 工具1的结果
%s: 工具2的结果
...
%s: 工具N的结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)
FN_CALL_TEMPLATE = FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_ZH


def get_function_description(function: Dict) -> str:
    """
    Text description of function
    """
    tool_desc = '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}'
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)
    assert name_for_human and name_for_model

    if name_for_model == 'code_interpreter':
        args_format = '此工具的输入应为Markdown代码块。'
    else:
        args_format = '此工具的输入应为JSON对象。'
    args_format = function.get('args_format', args_format)

    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()


class QwenAssistant(BaseAssistant):

    def _run(self, messages: List[BaseMessage], lang: str = 'zh', **kwargs) -> Iterator[List[BaseMessage]]:
        extra_generate_cfg = {}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        while (True):
            output_stream = self._call_llm(messages=messages,
                                           extra_generate_cfg=extra_generate_cfg, stream=kwargs.get('stream', False), stream_usage=kwargs.get('usage', False))
            output: List[AIMessage] = []
            if isinstance(output_stream, AIMessage):
                output_stream = self._postprocess_messages(output_stream)
                output = output_stream
                yield output
            else:
                full_resp = ''
                for o in output_stream:
                    if o:
                        full_resp += o.content
                        output = self._postprocess_messages(AIMessage(content=full_resp))
                        yield [o]
            if output:
                messages.extend(output)
                used_any_tool = False
                for out in output:
                    if out.tool_calls:
                        for tool_call in out.tool_calls:
                            tool_result = self._call_tool(tool_call.get('name'), tool_call.get('args'), messages=messages, **kwargs)
                            fn_msg = ToolMessage(
                                tool_call_id=tool_call.get('name'),
                                content=tool_result,
                                artifact=tool_result,
                            )
                            messages.append(fn_msg)
                            yield [ToolMessage(
                                tool_call_id=tool_call.get('name'),
                                content=f'<tool_response>\n{tool_result}\n</tool_response>\n',
                                artifact=tool_result,
                            )]
                            used_any_tool = True
                if not used_any_tool:
                    break

