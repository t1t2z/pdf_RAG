import io
import json
from typing import Optional, List, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.documents import Document
from backend.utils.datautil import text_splitter
from backend.config import vectorstore, engine
from backend.graph import run_graph_with_streaming
from pathlib import Path
from sqlalchemy import text
from uuid import uuid4
import pdfplumber


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str
    thread_id: str 
    files: Optional[List[str]]


@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are allowed')

    content = await file.read()

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    doc = Document(page_content=text, metadata={"file_tag": file.filename})
    split_docs = text_splitter.split_documents([doc])
    vectorstore.add_documents(split_docs)

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
        "chunk_count": len(split_docs),
        "preview": text[:500],
        "saved_path": str(output_path)
    }


def _build_chat_input(text: str,  files: Optional[List[str]]) -> dict:
    return {
        "user_input": text,
        "intent": None,
        "response": None,
        "chat_history": [],
        "files": files or [],
        "sub_intent": None
    }



def _sse_event(event_type: str, content: str) -> str:
    return f"data: {json.dumps({'type': event_type, 'content': content}, ensure_ascii=False)}\n\n"


async def _stream_graph_response(
    text: str,
    thread_id: str,
    files: Optional[List[str]] 
) -> AsyncGenerator[str, None]:
    """
    运行 LangGraph 流程，每步都发 status 事件。
    当到达 LLM 调用时，转向真正的 token-level streaming 并 yield chunk。
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_state = _build_chat_input(text, files)
    try:
        # yield _sse_event("status", "🔍 正在分析意图...")

        async for status, chunk in run_graph_with_streaming(
            input_state=input_state,
            config=config
        ):
            if status == "status":
                yield _sse_event("status", chunk)
            elif status == "chunk":
                yield  _sse_event("chunk", chunk)
            elif status == "error":
                yield _sse_event("error", chunk)
                return

        yield _sse_event("done", "")
    except Exception as e:
        yield _sse_event("error", str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):

    return StreamingResponse(
        content = _stream_graph_response(req.text, req.thread_id, req.files),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ==================== 文档标签接口 ====================

@app.get("/get_all_file_tags")
def get_all_file_tags():
    try:
        sql = text("""
            SELECT DISTINCT cmetadata ->> 'file_tag' AS file_tag
            FROM langchain_pg_embedding
            WHERE cmetadata ? 'file_tag'
            ORDER BY file_tag;
        """)
        with engine.connect() as conn:
            result = conn.execute(sql)
            tag_list = [row.file_tag for row in result.fetchall()]

        return {
            "code": 200,
            "msg": "success",
            "all_file_tags": tag_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取标签失败: {str(e)}")


# ==================== 清除会话 ====================

@app.delete("/session/{thread_id}")
def clear_session(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        memory.delete_thread(config)
        return {"code": 200, "msg": f"会话 {thread_id} 已清除"}
    except Exception:
        return {"code": 200, "msg": f"会话 {thread_id} 不存在或已清除"}


@app.get("/health")
def health():
    return {"status": "ok"}
