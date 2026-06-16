from langchain_core.tools import tool
from backend.chain import rag_chain
from datetime import datetime

@tool
def get_current_time() -> str:
    """
    获取当前系统北京时间
    当用户询问现在几点、当前时间、现在日期时使用该工具
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前北京时间：{now}"

@tool
def rag_search(query: str, history: str , search: bool = True, files: list[str] | None = None) -> str:
    """
    检索内部知识库文档，查询文档资料、相关知识点时使用此工具
    :param query: 需要检索的问题/关键词
    :param history: 对话历史
    :param search: 是否使用检索功能
    :param files: 用户指定检索的知识库文档列表,默认为None即不指定
    """
    # print(f"rag_search工具被调用，参数：query={query}, history={history}, search={search}, files={files}")
    return rag_chain.invoke({"query": query, "history": history,"search": search, "files": files})

@tool
def calculator(a: float, b: float, op: str) -> str:
    """
    数学四则运算，计算加减乘除
    :param a: 第一个数字
    :param b: 第二个数字
    :param op: 运算符，仅支持 +  -  *  /
    """
    if op == "+":
        res = a + b
    elif op == "-":
        res = a - b
    elif op == "*":
        res = a * b
    elif op == "/":
        if b == 0:
            return "计算失败:除数不能为0"
        res = a / b
    else:
        return f"不支持运算符 {op}，仅支持 + - * /"
    return f"计算结果：{a} {op} {b} = {res}"

tools = [get_current_time,rag_search,calculator]