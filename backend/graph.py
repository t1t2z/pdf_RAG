from typing import TypedDict, Optional, List, Dict, AsyncGenerator
from backend.utils.tools import get_current_time
from backend.chain import build_rag_response_stream, build_rag_response_sync
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from backend.config import llm

memory = MemorySaver()


class AppState(TypedDict):
    user_input: str
    intent: Optional[str]
    response: Optional[str]
    chat_history: List[Dict[str, str]]
    files: List[str]
    sub_intent: Optional[str]


def update_history(state: AppState) -> None:
    if "chat_history" not in state:
        state["chat_history"] = []
    state["chat_history"].append({"role": "human", "content": state["user_input"]})
    state["chat_history"].append({"role": "ai", "content": state["response"]})


# ==================== 节点1：意图识别 (同步) ====================

def intent_node(state: AppState) -> AppState:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""
    对话历史：
    {history_context}

    最新用户输入：{user_input}

    请判断用户意图，只返回 time / calculate / chat 其中一个

    """
    try:
        res = llm.invoke(prompt)
        state["intent"] = res.content.strip()
    except Exception as e:
        print(f"意图识别出错: {e}")
        state["intent"] = "chat"

    if state["intent"] not in ["time", "calculate", "chat"]:
        state["intent"] = "chat"
    return state


# ==================== 节点2：查时间 ====================

def handle_time(state: AppState) -> AppState:
    now = get_current_time.invoke({})
    state["response"] = f"当前北京时间：{now}"
    update_history(state)
    return state


# ==================== 节点3：计算 ====================

def handle_calculate(state: AppState) -> AppState:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""
    对话历史：
    {history_context}

    请计算以下表达式的结果。用户输入：{user_input}
    """
    try:
        res = llm.invoke(prompt)
        state["response"] = res.content.strip()
    except Exception as e:
        print(f"计算出错: {e}")
        state["response"] = "计算出错，请重试"
    finally:
        update_history(state)

    return state


# ==================== 节点4：对话（带流式） ====================

def _decide_chat_sub_intent(state: AppState) -> str:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    selected_files = state.get("files", [])

    #如果有file，直接走rag
    if selected_files and len(selected_files) > 0:
        state["sub_intent"] = "rag_search_chat"
        return state["sub_intent"]

    prompt = f"""
    对话历史：
    {history_context}

    最新用户输入：{user_input}
    用户未选择任何检索文档。

    这条消息是知识性问题吗（需要检索资料/数据/事实才能回答）？
    只返回 rag_search_chat 或 normal_chat
    """
    try:
        res = llm.invoke(prompt)
        result = res.content.strip()
    except Exception:
        result = "normal_chat"

    state["sub_intent"] = result if result in ["rag_search_chat", "normal_chat"] else "normal_chat"
    return state["sub_intent"]


async def handle_chat(state: AppState, writer: StreamWriter) -> AppState:
    """
    异步节点：通过 writer 实现 token-level streaming。
    writer 是 LangGraph 的 StreamWriter，用来把 token 逐个传出去。
    """
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    selected_files = state.get("files")

    # 二次路由
    _decide_chat_sub_intent(state)

    if state["sub_intent"] == "rag_search_chat":
        writer({"type": "status", "content": "📚 正在检索知识库..."})
    else:
        writer({"type": "status", "content": "💬 正在生成回答..."})

    full_response = ""
    async for chunk in build_rag_response_stream(
        query=user_input,
        history=history_context,
        files=selected_files,
        search=(state["sub_intent"] == "rag_search_chat"),
    ):
        full_response += chunk
        writer({"type": "chunk", "content": chunk})

    state["response"] = full_response
    update_history(state)
    return state


# ==================== 路由边 ====================

def route_by_intent(state: AppState) -> str:
    if state["intent"] == "time":
        return "node_time"
    elif state["intent"] == "calculate":
        return "node_calculate"
    else:
        return "node_chat"


# ==================== 构建 Graph ====================

workflow = StateGraph(AppState)

workflow.add_node("intent_node", intent_node)
workflow.add_node("node_time", handle_time)
workflow.add_node("node_calculate", handle_calculate)
workflow.add_node("node_chat", handle_chat)

workflow.add_edge(START, "intent_node")

#path-map中第一个参数是源节点的输出，第二个参数是这个节点的输出结果满足什么条件时，走的节点名
workflow.add_conditional_edges(
    source="intent_node",
    path=route_by_intent,
    path_map={
        "node_time": "node_time",
        "node_calculate": "node_calculate",
        "node_chat": "node_chat"
    }
)

workflow.add_edge("node_time", END)
workflow.add_edge("node_calculate", END)
workflow.add_edge("node_chat", END)

graph = workflow.compile(checkpointer=memory)


# ==================== 流式 Graph 运行器 ====================

async def run_graph_with_streaming(
    input_state: dict,
    config: dict,
) -> AsyncGenerator[tuple[str, str], None]:
    """
    使用 graph.astream(stream_mode="custom") 运行 LangGraph，
    由 handle_chat 节点通过 writer 发出 token。

    Yield 格式: ("status" | "chunk" | "error", content)
    """
    # 确保 config 完整
    full_config = {
        "configurable": {
            "thread_id": config.get("configurable", {}).get("thread_id", "default"),
        }
    }
    checkpoint = memory.get_tuple(full_config)
    if checkpoint is None:
        # 新会话：传完整 state（graph.astream 需要 TypedDict 的初始值）
        full_config["configurable"]["checkpoint_ns"] = ""
    else:
        # 已有会话：图会自动合并
        full_config = checkpoint.config

    yield ("status", "🔍 正在分析意图...")

    try:
        async for chunk in graph.astream(
            input_state,
            config=full_config,
            stream_mode="custom",
        ):
            # chunk 就是 node 里 writer() 传出的 dict
            if isinstance(chunk, dict):
                event_type = chunk.get("type", "chunk")
                content = chunk.get("content", "")
                yield (event_type, content)

    except Exception as e:
        yield ("error", str(e))


if __name__ == "__main__":
    import asyncio

    async def main():
        config = {"configurable": {"thread_id": "test_session_1"}}

        async for status, content in run_graph_with_streaming(
            {"user_input": "你好，请简单介绍自己", "intent": None, "response": None, "chat_history": [], "files": []},
            config,
        ):
            if status == "chunk":
                print(content, end="", flush=True)
        print()

    asyncio.run(main())
