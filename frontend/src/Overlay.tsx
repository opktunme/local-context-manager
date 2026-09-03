import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ArrowUpRight, BrainCircuit, ShieldCheck, Sparkles } from 'lucide-react'
import './Overlay.css'

type Segment = { id: string; category: string; tokens: number; sequence: number }
type OverlayState = {
  active: boolean
  message?: string
  session_id?: string
  name?: string
  model?: string
  updated_at?: string
  request_count?: number
  context_limit?: number
  context_budget?: number
  auto_mode?: boolean
  auto_cleanup_mode?: 'conservative' | 'balanced' | 'aggressive'
  metrics?: { active_tokens: number; removed_tokens: number; context_percent: number; categories: Record<string, number> }
  segments?: Segment[]
}

const TILE_COUNT = 400
const categoryColors: Record<string, string> = {
  System: '#a996f8', Conversation: '#6ca6fa', 'Code / files': '#4bc1a0',
  'Tool outputs': '#f2b76b', 'Memory / RAG': '#d47bba', Images: '#58c6d6',
  Summaries: '#b5d56a', Free: '#242a31',
}
const legendOrder = ['System', 'Conversation', 'Code / files', 'Tool outputs', 'Memory / RAG', 'Images', 'Summaries']
const categoryAbbreviations: Record<string, string> = { System: 'S', Conversation: 'C', 'Code / files': 'F', 'Tool outputs': 'T', 'Memory / RAG': 'M', Images: 'I', Summaries: 'Σ' }

function compact(value: number) { return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)}K` : Math.round(value).toLocaleString() }

function Overlay() {
  const [data, setData] = useState<OverlayState | null>(null)
  const [connected, setConnected] = useState(false)
  const [pulse, setPulse] = useState(false)
  const previousTokens = useRef(0)
  const sessionId = new URLSearchParams(window.location.search).get('session')

  useEffect(() => {
    document.title = 'LCM Context Window'
    document.documentElement.classList.add('overlay-document')
    let stopped = false
    let timer = 0
    const load = async () => {
      try {
        const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''
        const response = await fetch(`/api/overlay-state${query}`, { cache: 'no-store' })
        if (!response.ok) throw new Error('Overlay feed unavailable')
        const next = await response.json() as OverlayState
        if (stopped) return
        const nextTokens = next.metrics?.active_tokens || 0
        if (previousTokens.current && nextTokens !== previousTokens.current) {
          setPulse(false)
          window.requestAnimationFrame(() => setPulse(true))
        }
        previousTokens.current = nextTokens
        setData(next)
        setConnected(true)
      } catch {
        if (!stopped) setConnected(false)
      }
    }
    void load()
    timer = window.setInterval(() => void load(), 650)
    return () => { stopped = true; window.clearInterval(timer); document.documentElement.classList.remove('overlay-document') }
  }, [sessionId])

  const cells = useMemo(() => {
    const result = Array.from({ length: TILE_COUNT }, () => ({ category: 'Free', segment: '', start: false }))
    if (!data?.active || !data.context_limit) return result
    let tokens = 0
    let cursor = 0
    for (const segment of data.segments || []) {
      tokens += segment.tokens
      const idealEnd = Math.round(tokens / data.context_limit * TILE_COUNT)
      const end = Math.min(TILE_COUNT, Math.max(cursor + (segment.tokens > 0 ? 1 : 0), idealEnd))
      for (let index = cursor; index < end; index += 1) result[index] = { category: segment.category, segment: segment.id, start: index === cursor }
      cursor = end
      if (cursor >= TILE_COUNT) break
    }
    return result
  }, [data])

  if (!data?.active) return <main className="overlay-shell waiting"><div className="overlay-brand"><div><BrainCircuit size={18} /></div><span><strong>LCM</strong><small>CONTEXT WINDOW</small></span></div><section><Activity size={26} /><h1>Waiting for context</h1><p>{connected ? data?.message || 'Send the first model request to begin.' : 'Connecting to the local context feed…'}</p></section></main>

  const limit = data.context_limit || 1
  const budget = data.context_budget || limit
  const used = data.metrics?.active_tokens || 0
  const percent = Math.min(100, data.metrics?.context_percent || 0)
  const visibleCategories = legendOrder.filter((category) => (data.metrics?.categories[category] || 0) > 0)

  return <main className="overlay-shell">
    <header className="overlay-header"><div className="overlay-brand"><div><BrainCircuit size={14} /></div><span><strong>LCM</strong><small>CONTEXT</small></span></div><span className={`overlay-live ${connected ? 'connected' : ''}`} title={connected ? 'Live local feed' : 'Reconnecting'}><i />{connected ? 'LIVE' : '…'}</span><button title="Open full dashboard" onClick={() => window.open('/', '_blank')}><ArrowUpRight size={13} /></button></header>
    <section className={`literal-window ${pulse ? 'just-changed' : ''}`} title={`${data.name} · ${data.model}`} onAnimationEnd={() => setPulse(false)}><div className="window-titlebar"><span><i /><i /><i /></span><strong>{compact(used)} / {compact(limit)}</strong><small>{percent.toFixed(1)}%</small></div><div className="pixel-chamber"><div className="pixel-grid">{cells.map((cell, index) => <i key={index} className={`${cell.category === 'Free' ? 'empty' : 'filled'} ${cell.start ? 'segment-start' : ''}`} style={{ '--cell-color': categoryColors[cell.category] } as React.CSSProperties} title={cell.category} />)}</div><div className="budget-line" style={{ bottom: `${Math.min(100, budget / limit * 100)}%` }}><span>{compact(budget)}</span></div></div><footer><span className="mini-legend">{visibleCategories.map((category) => <i key={category} title={category} style={{ '--legend-color': categoryColors[category] } as React.CSSProperties}>{categoryAbbreviations[category]}</i>)}</span><span>≈{Math.max(1, Math.round(limit / TILE_COUNT)).toLocaleString()} tok / px</span></footer></section>
    <footer className={`overlay-policy ${data.auto_mode ? 'active' : ''}`}>{data.auto_mode ? <Sparkles size={11} /> : <ShieldCheck size={11} />}<span><strong>{data.auto_mode ? `${data.auto_cleanup_mode} auto-clean` : 'Cleanup paused'}</strong></span><i>{data.auto_mode ? 'AUTO' : 'SAFE'}</i></footer>
  </main>
}

export default Overlay
