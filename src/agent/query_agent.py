from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.agent_state import AgentState
from src.agent.base_agent import BaseAgent
from src.llm.qwen import getLlm

system_prompt = """
你是多智能体系统中的“查询智能体”，负责为系统中的其他智能体（如执行、评审、规划等）提供外部信息支撑。
当前系统结构如下：
你（QueryAgent）：负责查询外部信息、补充专业名词解释、验证事实、获取实时数据等。
KnowledgeAgent：构建系统背景知识；
PlannerAgent：制定任务列表；
ExecutorAgent：执行任务；
ReviewerAgent：评估任务完成情况；
查询目标来自其他智能体或最终用户，但你不能直接照抄问题进行搜索。
必须：
理解其核心信息缺口；
提炼或重构查询关键词；
判断应访问的站点类型（如百科、实时资讯、垂直平台）；
再进行搜索；
最终给出提炼后的摘要供其他智能体使用。
"""

tools = ['query_url', 'search_web']
class QueryAgent(BaseAgent):
    queryLlm = None
    def __init__(self):
        queryLlm = getLlm({}, tools)

    def __call__(self, state: AgentState):
        messages = [SystemMessage(content=system_prompt)]
        messages.append(
            HumanMessage(content=state.user_task))
        while True:
            response = self.invoke_llm(messages)

