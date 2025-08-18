import React, { useEffect, useState } from 'react'

type ApiResult<T> = { ok: boolean; data?: T; error?: string }

async function call<T>(method: string, path: string, body?: any): Promise<ApiResult<T>> {
  try {
    const res = await fetch(`/api${path}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    })
    if (!res.ok) {
      const msg = await res.text()
      return { ok: false, error: msg }
    }
    const data = await res.json()
    return { ok: true, data }
  } catch (e: any) {
    return { ok: false, error: String(e) }
  }
}

export default function App() {
  const [initialized, setInitialized] = useState(false)
  const [busy, setBusy] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const [papers, setPapers] = useState<any[]>([])
  const [paperUrl, setPaperUrl] = useState('')
  const [msg, setMsg] = useState<string>('')
  const [drafts, setDrafts] = useState<any[]>([])
  const [q, setQ] = useState('Transformer')
  const [mem, setMem] = useState<any[]>([])
  const [preview, setPreview] = useState<{name: string, html?: string} | null>(null)
  const [sseMsg, setSseMsg] = useState<string>('')

  const init = async () => {
    setBusy(true)
    const r = await call('POST', '/initialize', { config_path: 'config.yaml' })
    setBusy(false)
    if (r.ok) {
      setInitialized(true)
      setMsg('初始化成功')
      loadStats()
      listDrafts()
    } else setMsg(r.error || '初始化失败')
  }

  const loadStats = async () => {
    const r = await call<any>('GET', '/stats')
    if (r.ok) setStats(r.data)
  }

  const discover = async () => {
    setBusy(true)
    const r = await call<any>('POST', '/discover')
    setBusy(false)
    if (r.ok) setPapers(r.data.papers || [])
  }

  const process = async () => {
    if (!paperUrl) return
    setBusy(true)
    const r = await call<any>('POST', '/process', { paper_url: paperUrl })
    setBusy(false)
    if (r.ok) {
      setMsg(`已生成: ${r.data.article_path}`)
      listDrafts()
    } else setMsg(r.error || '处理失败')
  }

  const listDrafts = async () => {
    const r = await call<any>('GET', '/drafts')
    if (r.ok) setDrafts(r.data.items || [])
  }

  const searchMemory = async () => {
    const r = await fetch(`/api/memory/search?q=${encodeURIComponent(q)}&limit=10`)
    if (r.ok) {
      const data = await r.json()
      setMem(data.results || [])
    }
  }

  useEffect(() => {
    // noop on mount
  }, [])

  const previewHtml = async (name: string) => {
    const res = await fetch(`/api/drafts/${encodeURIComponent(name)}/export/html`)
    if (res.ok) {
      const html = await res.text()
      setPreview({ name, html })
    }
  }

  const sseDiscover = () => {
    if (!initialized) return
    setPapers([])
    setSseMsg('')
    const es = new EventSource('/api/sse/discover')
    es.addEventListener('start', () => setSseMsg('正在发现论文...'))
    es.addEventListener('paper', (ev) => {
      try { const d = JSON.parse((ev as MessageEvent).data); setPapers(p => [d, ...p]) } catch {}
    })
    es.addEventListener('complete', () => { setSseMsg('完成'); es.close() })
    es.addEventListener('error', () => { setSseMsg('错误'); es.close() })
  }

  const sseProcess = () => {
    if (!initialized || !paperUrl) return
    setSseMsg('')
    const es = new EventSource(`/api/sse/process?paper_url=${encodeURIComponent(paperUrl)}`)
    es.addEventListener('start', () => setSseMsg('开始处理'))
    es.addEventListener('progress', (ev) => {
      try { const d = JSON.parse((ev as MessageEvent).data); setSseMsg(`${d.stage || ''} ${(d.progress ?? '').toString()}`) } catch {}
    })
    es.addEventListener('stage', (ev) => { try { const d = JSON.parse((ev as MessageEvent).data); setSseMsg(`阶段: ${d.name}`) } catch {} })
    es.addEventListener('complete', async (ev) => {
      try { const d = JSON.parse((ev as MessageEvent).data); setMsg(`已生成: ${d.article_path}`); await listDrafts() } catch {}
      es.close()
    })
    es.addEventListener('error', () => { setSseMsg('错误'); es.close() })
  }

  return (
    <div className="min-h-full">
      <header className="border-b bg-white">
        <div className="max-w-6xl mx-auto p-4 flex items-center gap-4">
          <h1 className="text-xl font-semibold">Deep Scholar AI 控制台</h1>
          <button className="px-3 py-1.5 rounded bg-black text-white" onClick={init} disabled={busy || initialized}>
            {initialized ? '已初始化' : '初始化'}
          </button>
          <button className="px-3 py-1.5 rounded bg-gray-900 text-white" onClick={loadStats} disabled={busy || !initialized}>统计</button>
          <button className="px-3 py-1.5 rounded bg-indigo-600 text-white" onClick={discover} disabled={busy || !initialized}>发现论文</button>
          <span className="text-sm text-gray-600">{busy ? '处理中…' : msg}</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-4 grid grid-cols-1 md:grid-cols-3 gap-6">
        <section className="md:col-span-2 bg-white rounded shadow p-4">
          <h2 className="font-medium mb-3">处理单篇论文</h2>
          <div className="flex gap-2">
            <input className="flex-1 border rounded px-3 py-2" placeholder="https://arxiv.org/abs/1706.03762 或 file:///.../paper.pdf"
              value={paperUrl} onChange={(e) => setPaperUrl(e.target.value)} />
            <button className="px-3 py-2 rounded bg-emerald-600 text-white" onClick={process} disabled={!initialized || busy}>处理</button>
          </div>

          <div className="mt-6 mb-2 flex items-center gap-2">
            <h3 className="font-medium">发现列表</h3>
            <button className="px-2 py-1 rounded bg-indigo-700 text-white text-xs" onClick={sseDiscover} disabled={!initialized}>SSE发现</button>
          </div>
          {!!sseMsg && <div className="mb-2 text-xs text-gray-600">{sseMsg}</div>}
          <div className="space-y-2 max-h-72 overflow-auto">
            {papers.map((p) => (
              <div key={p.id || p.title} className="border rounded p-3">
                <div className="font-semibold text-sm">{p.title}</div>
                <div className="text-xs text-gray-600">{(p.authors || []).join(', ')}</div>
                <div className="text-xs">score: {p.relevance_score ?? '-'}</div>
              </div>
            ))}
            {!papers.length && <div className="text-sm text-gray-500">点击“发现论文”加载</div>}
          </div>
        </section>

        <section className="bg-white rounded shadow p-4">
          <h2 className="font-medium mb-3">草稿</h2>
          <div className="space-y-2 max-h-96 overflow-auto">
            {drafts.map((d) => (
              <div key={d.path} className="flex justify-between items-center border rounded p-2">
                <div className="text-sm truncate mr-2" title={d.name}>{d.name}</div>
                <div className="flex items-center gap-3">
                  <button className="text-sm text-sky-700" onClick={() => previewHtml(d.name)}>预览</button>
                  <a className="text-indigo-600 text-sm" href={`vscode://file/${encodeURIComponent(d.path)}`}>
                    在本地编辑器打开
                  </a>
                  <a className="text-sm text-emerald-700" href={`/api/drafts/${encodeURIComponent(d.name)}/export/html`} target="_blank">导出HTML</a>
                  <a className="text-sm text-emerald-700" href={`/api/drafts/${encodeURIComponent(d.name)}/export/pdf`} target="_blank">导出PDF</a>
                </div>
              </div>
            ))}
            {!drafts.length && <div className="text-sm text-gray-500">暂无草稿</div>}
          </div>
          {!!preview?.html && (
            <div className="mt-4 border rounded overflow-auto max-h-96">
              <iframe title={preview.name} srcDoc={preview.html} className="w-full h-96" />
            </div>
          )}
        </section>

        <section className="md:col-span-3 bg-white rounded shadow p-4">
          <h2 className="font-medium mb-3">混合记忆检索</h2>
          <div className="flex gap-2 mb-3">
            <input className="flex-1 border rounded px-3 py-2" value={q} onChange={(e) => setQ(e.target.value)} />
            <button className="px-3 py-2 rounded bg-sky-600 text-white" onClick={searchMemory} disabled={!initialized}>搜索</button>
          </div>
          <div className="grid md:grid-cols-2 gap-3 max-h-96 overflow-auto">
            {mem.map((r, i) => (
              <div key={i} className="border rounded p-3">
                <div className="text-xs text-gray-500">{r.memory_type} · {r.similarity_score.toFixed(3)}</div>
                {r.memory_type === 'episodic' ? (
                  <>
                    <div className="font-medium text-sm mb-1">{r.context || r.source || r.id}</div>
                    <div className="text-sm whitespace-pre-line">{r.content}</div>
                  </>
                ) : (
                  <>
                    <div className="font-medium text-sm mb-1">{r.name} ({r.node_type})</div>
                    <div className="text-xs">{JSON.stringify(r.attributes)}</div>
                  </>
                )}
              </div>
            ))}
            {!mem.length && <div className="text-sm text-gray-500">输入关键词后搜索</div>}
          </div>
        </section>
      </main>
    </div>
  )
}


