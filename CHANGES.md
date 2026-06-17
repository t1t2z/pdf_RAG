# 改动文档：真正的 Token-Level 流式输出 + 思考过程可视化

> 日期：2026-06-16  
> 目标：实现逐字流式输出 + 前端展示后端思考/状态过程

> 📖 如果你对 async/await、流式输出、StreamWriter 不理解，请先阅读：
> - **[LEARNING_GUIDE.md](./LEARNING_GUIDE.md)** — 从零开始的异步+流式学习指南
> - **[PRACTICE_TASKS.md](./PRACTICE_TASKS.md)** — 6个动手练习任务
> - 运行 `backend/test_sync_vs_async.py` 直观感受同步 vs 异步差异

---

## 核心问题分析

### 上一版的问题

1. **伪流式**：`main.py` 使用 `graph.astream(stream_mode="values")`，但 graph 各节点使用的是 `llm.invoke()`（同步调用）。`stream_mode="values"` 只在节点执行完成后才 yield 整个状态值，所以前端收到的是整段文本（分块发送），而不是逐 token。
2. **思考过程丢失**：`status` 类型的 SSE 事件后端有发（如 `"正在分析意图..."`），但前端只处理 `chunk` 类型，`status` 事件被忽略。

### 根本原因

`llm.invoke()` 是同步调用，内部虽然开启了 `streaming=True`，但 `invoke` 会等整个响应完成后再聚合返回。要真正逐 token 流式输出，必须使用 `llm.astream()` 并逐个 yield。

---

## 改动文件列表

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `backend/main.py` | 重写 | SSE 端点简化，流式逻辑下沉到 graph |
| `backend/graph.py` | 重写 | 使用 LangGraph 的 `StreamWriter` + `stream_mode="custom"` 实现 token streaming |
| `backend/chain.py` | 重写 | 拆分为同步/异步两个函数，新增真正的 async generator |
| `frontend/src/App.jsx` | 重写 | 显示思考过程标签 + 逐字渲染 |
| `frontend/src/index.css` | 更新 | 新增思考过程气泡样式 + 状态指示器 |

---

## 一、后端改动（逐文件）

### 1. `backend/chain.py` — 核心流式引擎

**改动前：**
```python
rag_chain = (... | prompt_branch | llm | parser)
# 只有一个 invoke() 的 chain，无法流式
```

**改动后：** 拆分为两个独立函数：

```python
# 新增：非流式函数
def build_rag_response_sync(query, history, files, search) -> str:
    messages = _build_messages(query, history, files, search)
    res = llm.invoke(messages)
    return res.content

# 新增：真正的流式 async generator
async def build_rag_response_stream(query, history, files, search) -> AsyncGenerator[str, None]:
    messages = _build_messages(query, history, files, search)
    async for chunk in llm.astream(messages):
        content = chunk.content
        if content:
            yield content
```

**关键变化：**
- `llm.invoke()` → `llm.astream()`：从"等全部完成"变为"来一个 token 就 yield 一个"
- `_build_messages()` 提取：把 prompt 构建 + 检索逻辑抽成独立函数，对流式/非流式共用
- 返回值从 `str` 变为 `AsyncGenerator[str, None]`，调用方必须 `async for`

---

### 2. `backend/graph.py` — 使用 LangGraph StreamWriter

**改动前：**
```python
def handle_chat(state: AppState) -> AppState:
    # 同步节点，llm.invoke()
    state["response"] = rag_search.invoke(...)
    return state
```

**改动后：**
```python
# 新增 StreamWriter import
from langgraph.types import StreamWriter

# 节点变成异步函数，接收 writer 参数
async def handle_chat(state: AppState, writer: StreamWriter) -> AppState:
    writer({"type": "status", "content": "📚 正在检索知识库..."})

    full_response = ""
    async for chunk in build_rag_response_stream(...):
        full_response += chunk
        writer({"type": "chunk", "content": chunk})

    state["response"] = full_response
    return state
```

**同时实现 `run_graph_with_streaming()` 函数：**
```python
async def run_graph_with_streaming(input_state, config):
    # 使用 graph.astream(stream_mode="custom")
    async for chunk in graph.astream(input_state, config=config, stream_mode="custom"):
        if isinstance(chunk, dict):
            yield (chunk["type"], chunk["content"])
```

**关键变化：**
- `handle_chat` 从同步函数 → 异步函数，签名加 `writer: StreamWriter`
- 节点内不再用 `rag_search.invoke()`（chain 同步调用），改用 `build_rag_response_stream()`（原生 async generator）
- `stream_mode="custom"` 使得 writer 发出的每个 dict 都作为独立事件传给 graph 外部
- `graph.astream()` 自动处理 state 的 checkpoint/merge，不再需要手动调用 `memory.put()`

**AppState 新增字段：**
```python
class AppState(TypedDict):
    ...
    sub_intent: Optional[str]  # 🆕 handle_chat 内部路由结果（rag_search_chat / normal_chat）
```

---

### 3. `backend/main.py` — SSE 端点简化

**改动前：**
```python
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        yield sse("status", "正在分析意图...")
        async for event in graph.astream(input_state, ...):
            yield sse("chunk", event["response"])
        yield sse("done", "")
    return StreamingResponse(event_generator(), ...)
```

**改动后：**
```python
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _stream_graph_response(req.text, req.thread_id, req.files),
        ...
    )

async def _stream_graph_response(text, thread_id, files):
    async for status, chunk in run_graph_with_streaming(input_state, config):
        if status == "status":
            yield sse("status", chunk)
        elif status == "chunk":
            yield sse("chunk", chunk)
```

**关键变化：**
- 移除了手动的 `event_generator` 内嵌逻辑
- 统一通过 `run_graph_with_streaming()` (graph.py) 获取流事件
- `await _sse_event(type, content)` 辅助函数：统一 SSE 格式（`ensure_ascii=False` 保证中文正常）
- `done` 事件在流结束时自动发送

---

### 4. SSE 事件协议

| type | 含义 | 前端行为 |
|------|------|---------|
| `status` | 思考/状态信息 | 显示为思考过程标签（带动画） |
| `chunk` | 逐 token 文本 | 追加到消息内容，逐字显示 |
| `error` | 错误信息 | 显示为错误消息 |
| `done` | 流结束 | 标记消息为非流式状态 |

### 状态事件列表

后端可能发送的 status 事件：
- `🔍 正在分析意图...` — intent_node 执行
- `🤔 判断是否需要检索知识库...` — 无选中文件时的二次路由
- `📚 正在检索知识库...` — RAG 检索后回答
- `💬 正在生成回答...` — 普通聊天
- `🕐 查询时间中...` — 时间查询
- `🧮 计算中...` — 计算

---

## 二、前端改动

### 1. `frontend/src/App.jsx`

**新增功能：**

1. **思考过程展示** — 每个助手消息现在有 `thinking` 数组属性
   ```jsx
   {msg.thinking && msg.thinking.length > 0 && (
     <div className="thinking-process">
       {msg.thinking.map((step, i) => (
         <div key={i} className="thinking-step">{step}</div>
       ))}
     </div>
   )}
   ```

2. **输入区实时状态指示器**
   ```jsx
   {thinkingText && (
     <div className="thinking-indicator">
       <span className="thinking-dot" />
       {thinkingText}
     </div>
   )}
   ```

3. **SSE 事件分发** — `status` 事件不再被忽略：
   ```javascript
   if (parsed.type === 'status') {
     thinkingSteps.push(parsed.content)        // 累积思考步骤
     setThinkingText(parsed.content)            // 实时状态
     setMessages(prev => {
       // 更新最后一条消息的 thinking 数组
       updated[last].thinking = [...thinkingSteps]
     })
   }
   ```

4. **消息结构调整** — 从 `message-content` 改为 `message-body` 包裹 `thinking-process` + `message-content`：
   ```jsx
   <div className="message-body">
     <div className="thinking-process">...</div>
     <div className="message-content">...</div>
   </div>
   ```

5. **`formatContent` 修复** — 使用正确的数组切片避免重复计算。

### 2. `frontend/src/index.css`

**新增样式：**

```css
/* 思考过程标签 */
.thinking-process { display: flex; flex-wrap: wrap; gap: 4px; }
.thinking-step {
  font-size: 12px; padding: 4px 10px;
  border-radius: 12px;
  background: #1a1a2e; border: 1px solid #2a2a4a;
  color: #a5b4fc;
  animation: fadeIn 0.3s ease;
}
.thinking-step:last-child { animation: pulse 1.5s infinite; }

/* 输入区状态指示器 */
.thinking-indicator {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; color: #a5b4fc;
}
.thinking-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #6366f1;
  animation: dotPulse 1s infinite;
}
```

---

## 三、数据流全景

```
用户输入 "什么是机器学习"
    ↓
POST /chat/stream  {text, thread_id, files}
    ↓
run_graph_with_streaming()
    ↓
graph.astream(stream_mode="custom")
    ↓
intent_node: llm.invoke → intent="chat"
    ↓ yield status
SSE: {"type":"status","content":"🔍 正在分析意图..."}
    ↓
handle_chat(state, writer):
    ↓ writer({"type":"status","content":"📚 正在检索知识库..."})
SSE: {"type":"status","content":"📚 正在检索知识库..."}
    ↓
retriever.invoke(query) → 检索到 3 个相关 chunk
    ↓
build_rag_response_stream():
    ↓ llm.astream(messages):
        ↓ yield "机器" → writer({"type":"chunk","content":"机器"})
SSE:     {"type":"chunk","content":"机器"}
        ↓ yield "学习" → writer({"type":"chunk","content":"学习"})
SSE:     {"type":"chunk","content":"学习"}
        ↓ yield "是"   → ...
        ...
    ↓
SSE: {"type":"done","content":""}
```

---

## 四、启动方式

```bash
# 终端1：启动后端
cd E:\pdf_RAG
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 终端2：启动前端
cd E:\pdf_RAG\frontend
npm run dev
```

前端 → `http://localhost:5173`，自动连接后端 `http://localhost:8000`。

---

## 五、技术要点

1. **LangGraph Custom Streaming**：通过 `stream_mode="custom"` + `StreamWriter`，在节点内部将 token 逐个推送到图外部，这是 LangGraph 官方推荐的流式输出方式。

2. **Async Generator 链**：`llm.astream()` → `build_rag_response_stream()` → `handle_chat(writer)` → `graph.astream()` → `run_graph_with_streaming()` → `_stream_graph_response()` → SSE → 前端。每一层都是纯 async generator，无缓冲。

3. **同步/异步分离**：`build_rag_response_sync()` 用于非流式场景（如 Node 的 `invoke` 测试），`build_rag_response_stream()` 用于流式场景。两个函数共享 `_build_messages()`。

4. **State 管理**：`graph.astream()` 自动通过 `checkpointer` (MemorySaver) 管理 state，每次调用自动合并 checkpoint 中的历史数据，无需手动 `memory.put()`。

5. **中文编码**：`json.dumps(ensure_ascii=False)` 确保中文在 SSE 中正常传输。
