"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agent.agent_state import AgentState, ModelMode
from src.agent import (PlannerAgent, ReviewerAgent, ToolAgent, KnowledgeAgent)
from src.agent.execute_agent import GeneralExecuteAgent
from src.log import logger
from src.tools import all_tools

planner = PlannerAgent(mode=ModelMode.LOCAL_QWEN)
reviewer = ReviewerAgent(mode=ModelMode.LOCAL_QWEN)
tool_agent = ToolAgent(mode=ModelMode.LOCAL_QWEN,tools=all_tools)
general_execute_agent = GeneralExecuteAgent(mode=ModelMode.LOCAL_QWEN)
knowledge = KnowledgeAgent(mode=ModelMode.LOCAL_QWEN)


def knowledge_node(state: AgentState):
    state.node = 'KNOWLEDGE'
    state = knowledge.acquire_knowledge(state)
    logger.info(f"[KNOWLEDGE] => {state.knowledge}")
    return state


def plan_node(state: AgentState):
    state.node = 'PLAN'
    state = planner.plan(state)
    logger.info(f"[PLANNER] => {state.task_list}")
    return state


def general_execute_node(state: AgentState):
    state.node = 'EXECUTE'
    state = general_execute_agent.run(state)
    return state


def tools_node(state: AgentState):
    knowledge.node = 'tools'
    # print(f"\n[EXECUTOR节点] 输入状态: tool_calls={state.tool_calls is not None}")
    if state.tool_calls:
        # print(f"[EXECUTOR节点] 执行工具调用: {len(state.tool_calls)} 个工具")
        state = tool_agent(state)
        # print(f"[EXECUTOR节点] 工具执行完成，结果数量: {len(state.execution_results)}")
    else:
        print("[EXECUTOR节点] 没有工具调用，跳过")
    return state


def review_node(state: AgentState):
    state.node = 'review'
    # print(f"\n[REVIEW节点] 输入状态: execution_results={len(state.execution_results)}")
    state = reviewer.review(state)
    # print(f"[REVIEW节点] 审查完成: {state.review_result}")
    return state


def execute_route(state: AgentState):
    if not state.task_finish:
        return "plan"
    else:
        return END


def review_route(state: AgentState):
    if state.review_result.get("finished", True):
        return END
    else:
        return "plan"


# 平台不支持本地持久化
# memory = MemorySaver()
# Define the graph
graph = (
    StateGraph(AgentState)
    .add_node("knowledge", knowledge_node)
    .add_node("plan", plan_node)
    .add_node("execute", general_execute_node)
    # .add_node("executor", tools_node)
    # .add_node("review", review_node)
    .add_edge(START, "knowledge")
    .add_edge("knowledge", "plan")
    .add_edge("plan", "execute")
    .add_conditional_edges("execute", execute_route)
    # .add_conditional_edges("review", review_route)
    .compile(name="Planning Graph")
)


def run(task):
    try:
        state = AgentState(user_task=task)
        final_state: AgentState
        for step in graph.stream(state, config={"configurable": {"thread_id": "1"}}, stream_mode="values", ):
            print(f"\n--- {step.get('node', 'start')}本轮输出 ---")
            for node, output in step.items():
                if output not in (None, '', [], {}):
                    print(f"[{node}] => {output}")

                # 如果有中断（例如需要用户输入）
                if hasattr(output, "awaiting_input") and output.awaiting_input:
                    print(f"\n[{node}] 正在等待用户输入...")
                    user_input = input("请输入：")
                    step = graph.resume(user_input)
                    print("\n>>> 已恢复执行")

            # 每次保存最新状态
            final_state = step
            print("\n")

        print("\n" + "=" * 50)
        print("=== 最终执行结果 ===")

        # 处理返回结果 - 可能是AgentState或dict
        if hasattr(final_state, 'user_task'):
            get = lambda k: getattr(final_state, k)
        else:
            get = lambda k: final_state[k]

        print(f"用户任务: {get('user_task')}")
        print(f"计划步骤数: {len(get('task_list'))}")
        print(f"执行工具数: {len(get('execution_results'))}")
        print(f"审查结果: {get('review_result')}")

    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run("什么是世界模型")
