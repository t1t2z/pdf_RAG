import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
env_path = os.path.join(os.path.dirname(__file__), "../data_env/.env")
load_dotenv(env_path)

#配置向量模型
#相较于手写embedding模型，调用openaiembeddings会自动进行批量处理和错误重试，且接口参数与openai官方兼容，后续如果需要更换向量模型或增加新的向量模型时，可以直接在这里进行修改和添加，而不需要修改后续的代码逻辑
#其中自动进行分页处理的默认chunk_size是1000
#但是国产大模型普遍有单次请求条数、请求体大小限制，所以改为10
embeddings = OpenAIEmbeddings(
    api_key=os.environ.get('TONGYI_API_KEY'),
    base_url=os.environ.get('TONGYI_API_BASE_URL'),
    model='text-embedding-v3',
    check_embedding_ctx_length=False,  # 因为langchain的aiembeddtings会自动进行数据长度超出处理，但是国产大模型的接口并不接受该类型的请求，如果强制开启会导致入库失败，所以这一步设为False
    chunk_size=10
)

#PGVecor向量数据库连接,connection_string为连接字符串，collection_name为表名
#相当于在数据库中建立一个向量数据知识库，后续可以通过这个知识库进行向量检索
CONNECTION_STRING = os.environ.get('db_url')
COLLECTION_NAME = "rag_docs"

vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    embeddings=embeddings ,#向量大模型的接口，负责将文本转换为向量
    use_jsonb=True#使用jsonb格式存储向量数据，可以更高效地进行检索和存储
)


llm = ChatOpenAI(
    api_key = os.environ.get('DEEPSEEK_API_KEY'),
    base_url = os.environ.get('DEEPSEEK_API_BASE_URL'),
    model = "deepseek-v4-pro",
    streaming=True,
    temperature=0
)

engine = create_engine(os.environ.get('db_url'))
session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
if __name__ == "__main__":
    chat_res = llm.invoke("请介绍一下你自己")
    print(chat_res.content.strip())