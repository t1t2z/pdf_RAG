import io
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from langchain_core.documents import Document
from backend.utils.datautil import text_splitter,get_custom_retriever
from backend.config import vectorstore,engine
from pydantic import BaseModel
from backend.graph import graph
from pathlib import Path
from sqlalchemy import text
from uuid import uuid4
import pdfplumber
from backend.graph import memory
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.base import CheckpointTuple
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, #代表请求可附带cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    thread_id: str = "default_session"  
    files: Optional[List[str]] = None    

#从本地接受一个pdf文件，提取文本，分割成chunk，生成embedding,保存这两个文件到目录中，并返回相关信息
@app.put('/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are allowed')
    
    content = await file.read()

    #接受pdf文件并提取文本text形式
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    doc = Document(page_content = text,metadata={"file_tag": file.filename})
    split_docs = text_splitter.split_documents([doc])
    vectorstore.add_documents(split_docs)#自动将文本批量转换成向量入库

    chunked_text = '\n\n'.join(
        f'--- chunk {index + 1} ---\n{chunk}' for index, chunk in enumerate(split_docs)
    )

    output_dir = Path("./data_env/uploaded_files").resolve()
    output_dir.mkdir(exist_ok=True)

    output_name = f"{Path(file.filename).stem}_{uuid4().hex}.txt"
    output_path = output_dir / output_name
    output_path.write_text(chunked_text, encoding='utf-8')

   
    return {
        "filename": file.filename,
        "total_chars": len(text),
        "chunk_size": 50,
        "overlap": 7,
        "chunk_count": len(split_docs),
        "embedding_model": "text-embedding-v3",
        "preview": text[:500]
        ,"saved_path": str(output_path)
    }


def _build_chat_input(text: str, thread_id: str, files: Optional[List[str]]) -> dict:
    """构建 graph.invoke 的输入，兼容新/旧会话"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint: CheckpointTuple | None = memory.get_tuple(config)

    if checkpoint is None:
        # 新会话
        return {
            "user_input": text,
            "intent": None,
            "response": None,
            "chat_history": [],
            "files": files or []
        }
    else:
        # 已有会话：只传新输入 + 可选的files更新 （热更新传参）
        return {"user_input": text, "files": files or []}



@app.get("/get_all_file_tags")
def get_all_file_tags():
    try:
        # 原生SQL提取去重后的file_tag
        sql = text("""
            SELECT DISTINCT cmetadata ->> 'file_tag' AS file_tag
            FROM langchain_pg_embedding
            WHERE cmetadata ? 'file_tag'
            ORDER BY file_tag;
        """)
        with engine.connect() as conn:
            result = conn.execute(sql)
            # 转为列表
            tag_list = [row.file_tag for row in result.fetchall()]
        
        return {
            "code": 200,
            "msg": "success",
            "all_file_tags": tag_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取标签失败: {str(e)}")
    
@app.get('/query_by_filetag')
def query_by_filetag(
    query_text: str = Query(..., description="你的问题"),
    selected_file_tag: str = Query(..., description="从/get_all_file_tags返回列表里选择的标签")
):  
    retriever = get_custom_retriever(file_tag=selected_file_tag)
    return graph_chat(text=query_text)
    


