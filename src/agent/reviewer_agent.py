import re
from typing import Any

import json5
from langchain_core.messages import SystemMessage, AIMessage

from .agent_state import AgentState
from .base_agent import BaseAgent

systemMessage = SystemMessage(content="""
你是一个代码开发任务的审查专家，你要对当前任务的执行结果进行审核。你禁止回答问题或帮助执行工具，你仅被允许根据上下文做出审查。
整体的流程： 用户下达任务 ->> 规划器拆分任务 ->> 工具执行 ->> 审查专家
你将会有以下信息：
- 用户原始问题
- 规划器拆分的子任务
- 工具执行的结果
- 规划器的最终输出
你需要根据以上的信息判断最终是否能完成用户下达的任务且符合预期。
如果你认为不能，请给出理由及调整建议，规划器将会根据你的输出重新制定子任务。
如果你认为可以，则对以上所有信息进行总结，你的总结将直接回传给用户，因此你需要注意你的表达方式。
你需要让用户明白任务是否执行完整且符合预期，同时将整个执行经过总结后反馈给用户。

**注意**
- 如果规划器是因为某些报错导致的无法继续执行，且你认为无法修复的话，则finished设置为true。
- 如果你因为某些原因不知道如何处理，可以调用human_assistance工具

格式说明:
你需要输出输出一个json用于标识当前执行是否完成，若finished为true，则你的输出将返回给用户，反之返回给规划器。error用于标识当前是否因某些报错导致的无法继续完成任务。
你还需要输出一段自然语言用于总结并回答用户问题，content和rejectReason只能有一个有值

**输出格式**
```json
{
    "finished": bool #true则填写content，false填写rejectReason
    "error": bool,
    "content": string #总结并回答用户原始问题,
    "rejectReason": string #不通过的原因
}
```
""")

class ReviewerAgent(BaseAgent):

    def review(self, state: AgentState) -> Any:
        state.node = 'review'

        # 审查执行结果
        try:
            messages = [systemMessage]
            messages.extend(state.messages + [AIMessage(content=getattr(state.response, "response", "我已完成规划，但根据上下文可以推断出答案，因此无子任务"))])
            response = self.invoke_llm(messages)
            review_content = getattr(response, "content", "审查完成")
            print(f"[REVIEWER] response {response}")
            # 提取json代码块
            finished = False
            error = False
            summary = ''
            rejectReason = ''
            match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', review_content, re.DOTALL)
            if match:
                try:
                    json_block = json5.loads(match.group(1))
                    finished = json_block.get("finished", False)
                    error = json_block.get("error", False)
                    summary = json_block.get("content", '')
                    rejectReason = json_block.get("rejectReason", '')
                except Exception as e:
                    print(f"[REVIEWER] 解析json失败: {e}")
            else:
                print("[REVIEWER] 未检测到json代码块，使用默认逻辑")
                finished = True
                error = False
            state.review_result = {
                "finished": finished,
                "error": error,
                "content": summary,
                "rejectReason": rejectReason
            }
            print(f"[REVIEWER] 审查完成，review_result: {state.review_result}")
        except Exception as e:
            print(f"[REVIEWER] 审查过程出错: {e}")
            state.review_result = {
                "finished": False,
                "error": True,
                "content": f"审查过程出错: {str(e)}",
                "suggestions": ["请检查系统配置"]
            }
        return state