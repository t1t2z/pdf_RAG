from typing import TypedDict, Optional, List, Dict, AsyncGenerator
from backend.utils.tools import get_current_time
from backend.responseCore import response_stream
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from backend.config import llm

memory = MemorySaver()

class AppState(TypedDict):
    user_input: str 
    intent: Optional[str] 
    response: Optional[str] 
    chat_history: List[Dict[str, str]] #传进来时就为[]或[..]
    files: List[str] #传进来时就为[]或[..]
    sub_intent: Optional[str]


def update_history(state: AppState) -> None:

    state["chat_history"].append({"role": "human", "content": state["user_input"]})
    state["chat_history"].append({"role": "ai", "content": state["response"]})



async def intent_node(state: AppState) -> AppState:
    # breakpoint()
    user_input = state["user_input"]

    chat_history = state["chat_history"]#传进来就因该是[]或[..]
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    prompt = f"""
    对话历史：
    {history_context}

    最新用户输入：{user_input}

    请判断用户意图，只返回 time / calculate / chat 其中一个

    """
    try:
        res = llm.invoke(prompt) #TODO 这就会阻塞线程
        state["intent"] = res.content.strip()
    except Exception as e:
        print(f"意图识别出错: {e}")#TODO 这个因该写在日志里
        state["intent"] = "chat"

    finally:
        if state["intent"] not in ["time", "calculate", "chat"]:
            state["intent"] = "chat"

    return state


async def time_node(state: AppState) -> AppState:
    now = get_current_time.invoke({})
    state["response"] = f"当前北京时间：{now}"
    update_history(state) #TODO 调用工具因该写在log里，并且传给前端，让前端实时显示工具调用情况
    return state


async def calculate_node(state: AppState) -> AppState:
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
        print(f"计算出错: {e}")#TODO 这个因该写在日志里
        state["response"] = "计算出错，请重试"
    finally:
        update_history(state)

    return state


def intent_chat_sub(state: AppState) -> str:
    user_input = state["user_input"]
    chat_history = state["chat_history"]
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    selected_files = state["files"]


    if selected_files:
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
        res = llm.invoke(prompt) #TODO 会阻塞线程
        result = res.content.strip()
    except Exception:
        result = "normal_chat"

    state["sub_intent"] = result if result in ["rag_search_chat", "normal_chat"] else "normal_chat"
    return state["sub_intent"]


async def chat_node(state: AppState, writer: StreamWriter) -> AppState:

    user_input = state["user_input"]
    chat_history = state["chat_history"]
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    files = state["files"]

    intent_chat_sub(state)

    if state["sub_intent"] == "rag_search_chat":
        writer({"type": "status", "content": "📚 正在检索知识库..."})
    else:
        writer({"type": "status", "content": "💬 正在生成回答..."})

    full_response = ""
    async for chunk in response_stream( #response_stream返回一个异步迭代器
        query = user_input,
        history = history_context,
        files = files,
        search = (state["sub_intent"] == "rag_search_chat")
    ):
        full_response += chunk
        writer({"type": "chunk", "content": chunk})

    state["response"] = full_response
    update_history(state)
    return state


def route_by_intent(state: AppState) -> str:
    if state["intent"] == "time":
        return "time"
    elif state["intent"] == "calculate":
        return "calculate"
    else:
        return "chat"


#构建 Graph

workflow = StateGraph(AppState)

workflow.add_node("intent_node", intent_node)
workflow.add_node("time", time_node)
workflow.add_node("calculate", calculate_node)
workflow.add_node("chat", chat_node)

workflow.add_edge(START, "intent_node")

#path-map中第一个参数是源节点的输出，第二个参数是这个节点的输出结果满足什么条件时，走的节点名
workflow.add_conditional_edges(
    source="intent_node",
    path=route_by_intent,
    path_map={
        "time": "time",
        "calculate": "calculate",
        "chat": "chat"
    }
)

workflow.add_edge("time", END)
workflow.add_edge("calculate", END)
workflow.add_edge("chat", END)

graph = workflow.compile(checkpointer=memory)


# 流式Graph运行，graph才是工作链，chain只是工具，所以它只返回迭代器，而不真正输出
async def run_graph_with_streaming(
    input_state: dict,
    config: dict
) -> AsyncGenerator[tuple[str, str], None]:

    yield ("status", "🔍 正在分析意图...")

    async for chunk in graph.astream(
        input = input_state if not memory.get_tuple(config) else {"user_input" : input_state["user_input"]} ,
        config = config,
        stream_mode = "custom"
    ):
        if isinstance(chunk, dict):
            event_type = chunk["type"]
            content = chunk["content"]
            yield (event_type, content)


if __name__ == "__main__":
    import asyncio

    async def main():
        config = {"configurable": {"thread_id": "test_session_1"}}

        async for status, content in run_graph_with_streaming(
            {"user_input": "你好，我叫小明", "intent": None, "response": None, "chat_history": [], "files": []},
            config,
        ):
            if status == "chunk":
                print(content, end="", flush=True)
        print()
        
        async for status, content in run_graph_with_streaming(
            {"user_input": "我叫什么名字", "intent": None, "response": None, "chat_history": [], "files": []},
            config,
        ):
            if status == "chunk":
                print(content, end="", flush=True)
        print()

    asyncio.run(main())
