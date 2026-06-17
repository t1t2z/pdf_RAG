import { useState, useEffect, useRef, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

// ====== SVG Icons ======
const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
)

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

// ====== 主组件 ======
export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [docTags, setDocTags] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [uploadStatus, setUploadStatus] = useState(null)
  const [threadId, setThreadId] = useState('default_session')
  const [thinkingText, setThinkingText] = useState('')  // 当前思考/状态文本
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const uploadRef = useRef(null)
  const abortRef = useRef(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, thinkingText, scrollToBottom])

  // 加载文档标签列表
  const fetchDocTags = async () => {
    try {
      const res = await fetch(`${API_BASE}/get_all_file_tags`)
      const data = await res.json()
      if (data.code === 200) {
        setDocTags(data.all_file_tags || [])
      }
    } catch (err) {
      console.warn('获取文档列表失败:', err)
    }
  }

  useEffect(() => { fetchDocTags() }, [])

  // 切换文档选中
  const toggleFile = (tag) => {
    setSelectedFiles(prev =>
      prev.includes(tag) ? prev.filter(f => f !== tag) : [...prev, tag]
    )
  }

  // 清除选中
  const clearSelection = () => setSelectedFiles([])

  // 文件上传
  const handleUpload = async (file) => {
    setUploadStatus({ type: 'loading', text: `上传中: ${file.name}` })
    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || '上传失败')
      }

      const data = await res.json()
      setUploadStatus({ type: 'success', text: `✅ ${data.filename} (${data.chunk_count} chunks)` })
      fetchDocTags()
      setTimeout(() => setUploadStatus(null), 4000)
    } catch (err) {
      setUploadStatus({ type: 'error', text: `❌ ${err.message}` })
      setTimeout(() => setUploadStatus(null), 5000)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      handleUpload(file)
    }
  }

  const onDragOver = (e) => {
    e.preventDefault()
  }

  // 发送消息
  const sendMessage = async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setThinkingText('')

    // 添加占位助手消息
    const assistantMsg = {
      role: 'assistant',
      content: '',
      isStreaming: true,
      thinking: [],  // 思考过程的步骤列表
    }
    setMessages(prev => [...prev, assistantMsg])
    setIsStreaming(true)

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          thread_id: threadId,
          files: selectedFiles.length > 0 ? selectedFiles : null,
        }),
        signal: controller.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullResponse = ''
      const thinkingSteps = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const dataStr = line.slice(6).trim()
          if (!dataStr) continue

          try {
            const parsed = JSON.parse(dataStr)

            if (parsed.type === 'status') {
              // 思考/状态消息
              const step = parsed.content
              thinkingSteps.push(step)
              setThinkingText(step)

              // 更新消息中附带的 thinking 列表
              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    thinking: [...thinkingSteps],
                    isStreaming: true,
                  }
                }
                return updated
              })

            } else if (parsed.type === 'chunk') {
              // 真正的 token 增量
              fullResponse += parsed.content
              // setThinkingText('')  // 开始输出内容后隐藏状态

              setMessages(prev => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    content: fullResponse,
                    thinking: [...thinkingSteps],
                    isStreaming: true,
                  }
                }
                return updated
              })

            } else if (parsed.type === 'done') {
              setThinkingText('')
            } else if (parsed.type === 'error') {
              fullResponse = `⚠️ 错误: ${parsed.content}`
              setThinkingText('')
            }
          } catch {
            // 非 JSON 行忽略
          }
        }
      }

      // 标记流结束
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            isStreaming: false,
            thinking: thinkingSteps,
          }
        }
        return updated
      })
      setThinkingText('')
    } catch (err) {
      if (err.name === 'AbortError') return
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: last.content || '请求失败，请重试',
            isStreaming: false,
          }
        }
        return updated
      })
      setThinkingText('')
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleNewSession = async () => {
    try {
      await fetch(`${API_BASE}/session/${threadId}`, { method: 'DELETE' })
    } catch { /* ignore */ }

    const newId = 'session_' + Date.now()
    setThreadId(newId)
    setMessages([])
    setSelectedFiles([])
    setUploadStatus(null)
    setThinkingText('')
  }

  const formatContent = (content) => {
    return content.split('\n').map((line, i, arr) => (
      <span key={i}>
        {line}
        {i < arr.length - 1 && <br />}
      </span>
    ))
  }

  return (
    <div className="app-container">
      {/* ====== 侧边栏 ====== */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>📚 PDF RAG</h2>
          <p>知识库问答系统</p>
        </div>

        {/* 上传区域 */}
        <div className="sidebar-section">
          <h3>📤 上传PDF</h3>
          <div
            className="upload-area"
            onClick={() => uploadRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
          >
            <div className="upload-icon">📄</div>
            <div className="upload-text">点击或拖拽上传 PDF</div>
            <div className="upload-hint">仅支持 .pdf 文件</div>
            <input
              ref={uploadRef}
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files[0]
                if (file) handleUpload(file)
                e.target.value = ''
              }}
            />
          </div>
          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.type}`}>
              {uploadStatus.text}
            </div>
          )}
        </div>

        {/* 文档列表 */}
        <div className="sidebar-section">
          <h3>📂 知识库文档</h3>
          {docTags.length === 0 ? (
            <div className="no-docs">暂无文档，请上传 PDF</div>
          ) : (
            <ul className="doc-list">
              {docTags.map((tag) => (
                <li
                  key={tag}
                  className={`doc-item ${selectedFiles.includes(tag) ? 'selected' : ''}`}
                  onClick={() => toggleFile(tag)}
                >
                  <div className="doc-checkbox">
                    {selectedFiles.includes(tag) && <CheckIcon />}
                  </div>
                  <span className="doc-icon">📄</span>
                  <span className="doc-name" title={tag}>{tag}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* ====== 主聊天区 ====== */}
      <main className="main-area">
        <div className="chat-header">
          <h1>
            {selectedFiles.length > 0
              ? `与 ${selectedFiles.length} 个文档对话中`
              : 'AI 对话助手'}
          </h1>
          <div className="chat-header-actions">
            <button className="btn btn-danger" onClick={handleNewSession}>
              <span style={{ marginRight: 4 }}>🗑</span>
              新对话
            </button>
          </div>
        </div>

        {/* 当前选中文件标签 */}
        {selectedFiles.length > 0 && (
          <div className="selected-files-bar">
            <span>检索文档:</span>
            {selectedFiles.map(tag => (
              <span key={tag} className="file-tag">
                📄 {tag}
                <span className="remove-tag" onClick={() => toggleFile(tag)}>×</span>
              </span>
            ))}
            <button
              className="btn btn-secondary"
              style={{ fontSize: 11, padding: '3px 10px', marginLeft: 8 }}
              onClick={clearSelection}
            >
              清除全部
            </button>
          </div>
        )}

        {/* 消息列表 */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <div className="welcome-icon">🤖</div>
              <div className="welcome-title">PDF RAG 知识库问答</div>
              <div className="welcome-subtitle">
                在左侧上传 PDF 文档，选中文档后可进行知识库检索问答<br />
                不选文档也可以直接对话
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-body">
                  {/* 思考过程 */}
                  {msg.thinking && msg.thinking.length > 0 && (
                    <div className="thinking-process">
                      {msg.thinking.map((step, i) => (
                        <div key={i} className="thinking-step">
                          {step}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* 正文内容 */}
                  <div className="message-content">
                    {msg.isStreaming && !msg.content ? (
                      <div className="typing-indicator">
                        <span /><span /><span />
                      </div>
                    ) : (
                      formatContent(msg.content)
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* 输入区 */}
        <div className="input-area">
          {/* 当前状态指示 */}
          {thinkingText && (
            <div className="thinking-indicator">
              <span className="thinking-dot" />
              {thinkingText}
            </div>
          )}
          <div className="input-container">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedFiles.length > 0
                  ? `询问关于 "${selectedFiles[0]}${selectedFiles.length > 1 ? `" 等 ${selectedFiles.length} 个文档` : '"'} 的问题...`
                  : '输入你的问题...'
              }
              disabled={isStreaming}
              autoFocus
            />
            <button
              className="send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || isStreaming}
            >
              <SendIcon />
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}
