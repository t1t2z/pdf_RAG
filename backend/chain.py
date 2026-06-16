from typing import Optional, List, AsyncGenerator
from backend.utils.datautil import get_custom_retriever
from backend.config import llm
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from operator import itemgetter


#返回指定文档搜索后的文本结果
def build_retriever(full_input: dict):
    query = full_input["query"]
    file_tags = full_input.get("files")
    retriever = get_custom_retriever(file_tags=file_tags)
    return retriever.invoke(query)


search_branch = RunnableLambda(build_retriever) #对于 RunnableLambda，|中导入不会执行，只会记录这个链路中的方法
no_search_branch = RunnableLambda(lambda x: None)

context_runnable = RunnableBranch(
    (lambda x: x.get("search", True), search_branch),
    no_search_branch
)

parser = StrOutputParser()



rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """这是历史对话：{history}。
    你是专业知识库问答助手，
     
    用户选择查找知识库文件{files}，所以你可以使用上下文中的信息来回答问题，
    请严格根据提供的上下文回答用户问题,输出简洁，
     
    如果上下文没有相关信息，则如实告知无法回答，不要编造内容。
    上下文：{context}"""),
    ("user", "{query}")
])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """这是历史对话：{history}。
     
    你是专业知识库问答助手，
    但是此次的问题没有上下文，回答用户问题,输出简洁，"""),

    ("user", "{query}")
])

prompt_branch = RunnableBranch(
    (lambda x: x.get("context") is not None and x.get("context") != "", rag_prompt),
    chat_prompt
)



#这样的chain无法流式，因为parser在最后一步，必须等到完整输出才能解析
#流式只能停在llm
rag_chain = (
    {
        "context": context_runnable,
        "query": itemgetter("query"),
        "history": itemgetter("history"),
        "files": itemgetter("files")
    }
    | prompt_branch
    | llm
    | parser
)





def _build_messages(
    query: str,
    history: str,
    files: Optional[List[str]],
    search: bool
) -> list:

    if search:
        retrieved_docs = build_retriever({
            "query": query,
            "files": files
        })
        context_text = ""
        if retrieved_docs:
            context_text = "\n\n".join([
                f"[来源: {doc.metadata.get('file_tag', '未知')}]\n{doc.page_content}"
                for doc in retrieved_docs
            ])

        system = f"""
        这是历史对话：{history}。

        你是专业知识库问答助手，
        用户选择查找知识库文件{files}，所以你可以使用上下文中的信息来回答问题，

        请严格根据提供的上下文回答用户问题,输出简洁，
        如果上下文没有相关信息，则如实告知无法回答，不要编造内容。
        上下文：{context_text}
        
        """
    else:
        system = f"""这是历史对话：{history}。
        你是专业知识库问答助手，
        但是此次的问题没有上下文，回答用户问题,输出简洁
        
        """

    return [
        SystemMessage(content=system),
        HumanMessage(content=query)
    ]


"""非流式生成回答，返回完整字符串"""
def build_rag_response_sync(
    query: str,
    history: str,
    files: Optional[List[str]] = None,
    search: bool = True,
) -> str:
   
    messages = _build_messages(query, history, files, search)
    res = llm.invoke(messages)
    return res.content if hasattr(res, 'content') else str(res)


""" 流式生成回答，逐个 yield token。直接作为 async generator 使用"""
async def build_rag_response_stream(
    query: str,
    history: str,
    files: Optional[List[str]] = None,
    search: bool = True,
) -> AsyncGenerator[str, None]:
    messages = _build_messages(query, history, files, search)
    #使用异步迭代，来真正实现不会卡死，可以处理其他用户请求等，不能同for迭代，这样即使是异步迭代器依然处理不了其他用户请求
    async for chunk in llm.astream(messages):#astream会返回一个异步迭代器，程序不会卡死，可以处理其他用户请求，迭代器每次迭代返回一个token级别的输出块
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            yield content


if __name__ == "__main__":
    import asyncio
    res = rag_chain.invoke({"query": "请介绍一下你自己", "search": False, "history": ""})
    print(res)
