from typing import Optional, List, AsyncGenerator
from backend.utils.datautil import get_custom_retriever
from backend.config import llm
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage



#返回指定文档搜索后的文本结果
# TODO: 把 build_retriever 改成异步的
def build_retriever(full_input: dict):
    query = full_input["query"]
    file_tags = full_input["files"]
    retriever = get_custom_retriever(file_tags=file_tags)
    return retriever.invoke(query)


#prompt动态构建
def _build_messages(
    query: str,
    history: str,
    files: List[str],
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




async def response_stream(
    query: str,
    history: str,
    files: List[str] ,
    search: bool = True,
) -> AsyncGenerator[str, None]:
    messages = _build_messages(query, history, files, search)

    async for chunk in llm.astream(messages):#astream会返回一个异步迭代器，程序不会卡死，可以处理其他用户请求，迭代器每次迭代返回一个token级别的输出块
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if content:
            print(f"Generated chunk: {repr(content)}") #TODO 写在日志里
            yield content


if __name__ == "__main__":
    
    async def main():
        async for chunk in response_stream(query = "请介绍一下你自己", history = "", files = None,search = False):
            print(chunk)

    asyncio.run(main())
