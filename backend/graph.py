from typing import TypedDict,Optional, List, Dict
from backend.utils.tools import get_current_time,rag_search
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from backend.config import llm

memory = MemorySaver()
class AppState(TypedDict):
    user_input: str
    intent: Optional[str]
    response: Optional[str]
    chat_history: List[Dict[str, str]]  # 对话历史 [{role: human/ai, content: 内容}]


def update_history(state: AppState) -> None:
    if "chat_history" not in state:
        state["chat_history"] = []
    state["chat_history"].append({"role": "human", "content": state["user_input"]})
    state["chat_history"].append({"role": "ai", "content": state["response"]})


#意图节点，各路由节点，条件路由函数（边）
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
        state["intent"] = "chat"  # 默认走闲聊
    
    finally:
        if state["intent"] not in ["time", "calculate", "chat"]:
            state["intent"] = "chat"  # 非法意图默认走闲聊
    return state

#  节点2：处理查时间
def handle_time(state: AppState) -> AppState:
    now = get_current_time.invoke({})
    
    state["response"] = f"当前北京时间：{now}"
    update_history(state)

    return state

# 节点3：处理计算（简易示例）
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

# 节点4：通用闲聊回答+rag知识库问答
def handle_chat(state: AppState) -> AppState:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    prompt = f"""
    对话历史：
    {history_context}
    
    最新用户输入：{user_input}
    
    请判断用户意图，只返回 rag_search_chat / normal_chat 其中一个
    """
   
    res = llm.invoke(prompt)
    print(f"是否需要检索知识库: {res.content.strip()}")

    if res.content.strip() == "rag_search_chat":
        # 被 @tool 装饰后的 StructuredTool 工具对象，不能直接像普通函数一样 get_current_time() 调用，必须用工具内置的 .invoke() 方法执行
        state["response"] = rag_search.invoke({"query": user_input, "history": history_context})
    else:
        state["response"] = rag_search.invoke({"query": user_input, "history": history_context, "search": False})

    update_history(state)
    return state

#边，输入为state，输出为节点名
def route_by_intent(state: AppState) -> str:
    if state["intent"] == "time":
        return "node_time"
    elif state["intent"] == "calculate":
        return "node_calculate"
    else:
        return "node_chat"
    
workflow = StateGraph(AppState)

#第一个参数为接节点标识名，第二个参数为action（具体执行函数）
workflow.add_node("intent_node", intent_node)
workflow.add_node("node_time", handle_time)
workflow.add_node("node_calculate", handle_calculate)
workflow.add_node("node_chat", handle_chat)

#入口
workflow.add_edge(START,"intent_node")

workflow.add_conditional_edges(
    source="intent_node",       # 源节点
    path=route_by_intent,       # 路由判断函数
    # path_map：字典，格式 {path返回值: 目标节点名}
    path_map={
        "node_time": "node_time",
        "node_calculate": "node_calculate",
        "node_chat": "node_chat"
    }
)

# 所有分支最终都走向 END（结束节点）
workflow.add_edge("node_time", END)
workflow.add_edge("node_calculate", END)
workflow.add_edge("node_chat", END)

#所有节点需要经过编译才能运行
graph = workflow.compile(
    # debug=True,
    checkpointer=memory # 接入MemorySaver
)


if __name__ == "__main__":
    
    #格式固定，依靠thread_id同步上一轮状态
    config = {"configurable": {"thread_id": "test_session_1"}}
 
    # 第一轮：自我介绍
    init_state1 = {
        "user_input": "你好，我叫小明",
        "intent": None,
        "response": None,
        "chat_history": []
    }
    result1 = graph.invoke(init_state1, config=config)
    print("第一轮回复：", result1["response"], "\n")
    
    # 第二轮：基于历史互动
    result2 = graph.invoke(
        {"user_input": "我叫什么名字？"},
        config=config
    )
    print("第二轮回复：", result2["response"], "\n")
