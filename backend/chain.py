from backend.utils.datautil import get_custom_retriever
from backend.config import llm
from langchain_core.runnables import RunnableLambda,RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


retriever = get_custom_retriever() 
search_branch = itemgetter("query") | retriever 
no_search_branch = RunnableLambda(lambda x: None) #目的是使context为空

#组装条件分支
context_runnable = RunnableBranch(
    (lambda x : x.get("search",True) , search_branch), #若无参数search则默认为True，后面不加括号的为默认 ，该结构按顺序匹配，可以多个条件类似（条件1）（条件2）命中第一个满足的条件就执行，后面不再判断
    no_search_branch #近似else
)

parser = StrOutputParser()

#rag知识库不使用promopt，而是用rag——prompt，后者会区分角色将检索到的上下文和用户问题一起发送给模型，模型根据上下文回答问题
#普通的prompt设计，直接将问题放在prompt中，发送给模型
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     这是历史对话：{history}。
     你是专业知识库问答助手，
     请严格根据提供的上下文回答用户问题,输出简洁，
     如果上下文没有相关信息，则如实告知无法回答，不要编造内容。
     上下文：{context}
     """),
    ("user", "{query}")
])
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     这是历史对话：{history}。
     你是专业知识库问答助手，
     但是此次的问题没有上下文，回答用户问题,输出简洁，
     """),
    ("user", "{query}")
])

prompt_branch = RunnableBranch(
    (lambda x: x["context"], rag_prompt),
    chat_prompt
)

rag_chain = (
    {
        "context": context_runnable, # 不加itemgetter相当于把上游的整个输入传递给context_runnable
        "query": itemgetter("query"),
        "history": itemgetter("history")
    }
    | prompt_branch
    | llm
    | parser)


if __name__ == "__main__":
    chat_res = rag_chain.invoke({"query": "请介绍一下你自己","search" : False, "history": "" })
    print(chat_res.content.strip())