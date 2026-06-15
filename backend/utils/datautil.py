from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import vectorstore,llm

#检索器，参数为相似度检查前三条记录，后续会根据这个检索器从数据库中找到最相似的记录
def get_custom_retriever(k: int = 3, file_tag: str | None = None):
    search_kwargs = {"k": k}
    if file_tag is not None:
        search_kwargs["filter"] = {"file_tag": file_tag}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

#文本分割器，单次分割长度为chunk_size，重叠为chunk_overlap，separators参数为分割文本的分隔符，以防止语义被破坏
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=7,
    separators=["\n\n", "\n", " ", ""]
)


