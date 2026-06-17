from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import vectorstore,llm
from typing import Optional, List

#检索器，参数为相似度检查前三条记录，后续会根据这个检索器从数据库中找到最相似的记录
def get_custom_retriever(file_tags: List[str] ,k: int = 3):
    search_kwargs = {"k": k}
    if len(file_tags) > 0:
        # PGVector jsonb 多值过滤语法：$in
        search_kwargs["filter"] = {
            "file_tag": {"$in": file_tags}
        }

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

#文本分割器，单次分割长度为chunk_size，重叠为chunk_overlap，separators参数为分割文本的分隔符，以防止语义被破坏
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=7,
    separators=["\n\n", "\n", " ", ""]
)


