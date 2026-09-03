import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity, ArchiveRestore, ArrowDownUp, BrainCircuit, Braces, Check, CheckCircle2, ChevronDown,
  CircleAlert, Clipboard, Clock3, Database, FileCode2, Gauge, HardDrive, History,
  Info, LayoutDashboard, Link2, LockKeyhole, Menu, MessageSquareText, PanelRightClose,
  Pin, PinOff, Play, Power, RefreshCw, RotateCcw, Search, Server, Settings2, ShieldCheck,
  Sparkles, SquareStack, Trash2, Undo2, Unplug, X, Zap,
} from 'lucide-react'
import './App.css'

type AdapterCatalogEntry = { kind: string; name: string; default_url: string; status: string }
type Backend = { id: string; name: string; kind: string; base_url: string; active: boolean; context_limit: number; architecture?: Record<string, unknown> | null }
type SessionSummary = { id: string; name: string; model: string; updated_at: string; request_count: number; token_count: number; context_limit: number; context_budget: number; item_count: number }
type Recommendation = 'safe_to_remove' | 'summarize' | 'probably_keep' | 'pinned_critical' | 'review_manually'
type CleanupMode = 'conservative' | 'balanced' | 'aggressive'
type ContextItem = {
  item_id: string; fingerprint: string; sequence: number; kind: string; role: string; source: string
  status: 'active' | 'removed' | 'pinned' | 'summarized'; token_count: number; token_exact: boolean
  token_source: string; preview: string; text: string; raw_message: Record<string, unknown>
  relationships: Array<{ type: string; item_id: string }>; recommendation: Recommendation
  recommendation_reason: string; reason: string
}
type RequestRecord = {
  id: string; incoming: Record<string, unknown>; forwarded: Record<string, unknown>
  incoming_tokens: number; forwarded_tokens: number; status_code: number | null
  ttft_ms: number | null; total_ms: number | null; prompt_tokens: number | null
  completion_tokens: number | null; created_at: string
}
type AuditRecord = { id: number; action: string; details: { item_ids?: string[]; token_delta?: number; original_action?: string; actor?: string }; created_at: string }
type SessionDetail = SessionSummary & {
  backend_id?: string | null; backend?: Backend | null; tokenizer_family: string; auto_mode: boolean; auto_cleanup_mode: CleanupMode; items: ContextItem[]
  summaries: Array<{ id: string; text: string; source_ids: string[]; token_count: number; sequence: number }>
  requests: RequestRecord[]; audit: AuditRecord[]
  metrics: { active_tokens: number; incoming_tokens: number; removed_tokens: number; context_percent: number; budget_percent: number; categories: Record<string, number>; counts_are_estimates: boolean; kv_cache: { gib: number; formula: string; assumptions: Record<string, unknown> } | null; kv_cache_incoming: { gib: number } | null }
}
type StatusData = { configured: boolean; backend: Backend | null; backends: Backend[]; sessions: SessionSummary[]; adapter_catalog: AdapterCatalogEntry[]; proxy_url: string; dashboard_url: string; privacy: Record<string, unknown> }
type ViewName = 'context' | 'largest' | 'requests' | 'audit' | 'settings'
type Mutate = (path: string, options: RequestInit, message: string) => Promise<void>

const categoryColors: Record<string, string> = { System: '#9b87f5', Conversation: '#5b9cf5', 'Code / files': '#4bc1a0', 'Tool outputs': '#f2aa5b', 'Memory / RAG': '#d47bba', Images: '#58c6d6', Summaries: '#b5d56a', Free: '#24272d' }
const recommendationLabels: Record<Recommendation, string> = { safe_to_remove: 'Safe to remove', summarize: 'Summarize', probably_keep: 'Probably keep', pinned_critical: 'Pinned / critical', review_manually: 'Review manually' }
const cleanupModes: Array<{ value: CleanupMode; label: string; short: string; description: string }> = [
  { value: 'conservative', label: 'Conservative', short: 'Duplicates only', description: 'Removes later exact copies. Unique content stays untouched.' },
  { value: 'balanced', label: 'Balanced', short: 'High-confidence waste', description: 'Also removes oversized stale logs and tool output.' },
  { value: 'aggressive', label: 'Aggressive', short: 'Compress + prune', description: 'Also summarizes old turns and drops stale retrieved context.' },
]

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) } })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || body.error?.message || 'Request failed')
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}
function compactNumber(value: number, estimated = false) { const marker = estimated ? '~' : ''; return value >= 1000 ? `${marker}${(value / 1000).toFixed(value >= 10000 ? 1 : 2)}K` : `${marker}${Math.round(value).toLocaleString()}` }
function timeAgo(value: string) { const minutes = Math.floor((Date.now() - new Date(value).getTime()) / 60000); if (minutes < 1) return 'just now'; if (minutes < 60) return `${minutes}m ago`; const hours = Math.floor(minutes / 60); return hours < 24 ? `${hours}h ago` : `${Math.floor(hours / 24)}d ago` }
function kindLabel(kind: string) { return kind.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) }

function App() {
  const [status, setStatus] = useState<StatusData | null>(null)
  const [session, setSession] = useState<SessionDetail | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [view, setView] = useState<ViewName>('context')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const loadStatus = useCallback(async (preferSession?: string) => {
    try {
      const next = await api<StatusData>('/api/status')
      setStatus(next)
      const id = preferSession || selectedSession || next.sessions[0]?.id
      if (id) { setSelectedSession(id); setSession(await api<SessionDetail>(`/api/sessions/${id}`)) } else setSession(null)
      setError(null)
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Unable to load LCM') }
    finally { setLoading(false) }
  }, [selectedSession])

  // oxlint-disable-next-line react/set-state-in-effect
  useEffect(() => { void loadStatus() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(null), 3200); return () => window.clearTimeout(timer) }, [toast])

  const openSession = async (id: string) => {
    setSelectedSession(id); setSidebarOpen(false); setLoading(true)
    try { setSession(await api<SessionDetail>(`/api/sessions/${id}`)); setView('context') }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Session could not be opened') }
    finally { setLoading(false) }
  }
  const createDemo = async () => {
    setLoading(true)
    try { const demo = await api<SessionDetail>('/api/demo', { method: 'POST' }); await loadStatus(demo.id); setToast('Demo session loaded — no model connection required') }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Demo could not be created') }
    finally { setLoading(false) }
  }

  if (loading && !status) return <LoadingScreen />
  if (error && !status) return <ErrorScreen message={error} onRetry={() => void loadStatus()} />
  if (!status) return null
  if (!status.configured && !session) return <SetupFlow status={status} onConfigured={(id) => void loadStatus(id)} onDemo={createDemo} />

  return <div className="app-shell">
    <Sidebar status={status} selectedSession={selectedSession} open={sidebarOpen} onClose={() => setSidebarOpen(false)} onOpenSession={(id) => void openSession(id)} onDemo={createDemo} />
    <main className="workspace">
      <Topbar session={session} proxyUrl={status.proxy_url} view={view} onView={setView} onMenu={() => setSidebarOpen(true)} onRefresh={() => void loadStatus(selectedSession || undefined)} />
      {error && <div className="inline-alert"><CircleAlert size={16} />{error}<button onClick={() => setError(null)}><X size={14} /></button></div>}
      {session ? <Dashboard session={session} view={view} status={status} onSession={setSession} onStatus={() => void loadStatus(selectedSession || undefined)} onToast={setToast} onError={setError} /> : <EmptySessions onDemo={createDemo} />}
    </main>
    {toast && <div className="toast"><CheckCircle2 size={17} />{toast}</div>}
  </div>
}

function LoadingScreen() { return <div className="loading-screen"><div className="brand-mark"><Braces size={25} /></div><span className="eyebrow">LOCAL CONTEXT MANAGER</span><div className="loading-line" /></div> }
function ErrorScreen({ message, onRetry }: { message: string; onRetry: () => void }) { return <div className="center-state"><Unplug size={34} /><h1>LCM is not responding</h1><p>{message}</p><button className="primary-button" onClick={onRetry}><RefreshCw size={16} />Retry</button></div> }
function EmptySessions({ onDemo }: { onDemo: () => void }) { return <div className="center-state"><SquareStack size={36} /><h1>No context captured yet</h1><p>Point a frontend at the LCM proxy, or load the sample session to explore the dashboard.</p><button className="primary-button" onClick={onDemo}><Play size={16} />Load demo session</button></div> }

function Sidebar({ status, selectedSession, open, onClose, onOpenSession, onDemo }: { status: StatusData; selectedSession: string | null; open: boolean; onClose: () => void; onOpenSession: (id: string) => void; onDemo: () => void }) {
  return <>{open && <button aria-label="Close navigation" className="sidebar-scrim" onClick={onClose} />}<aside className={`sidebar ${open ? 'is-open' : ''}`}>
    <div className="brand"><div className="brand-mark"><Braces size={21} /></div><div><strong>LCM</strong><span>Context runtime</span></div><button className="mobile-close" onClick={onClose}><X size={18} /></button></div>
    <div className="proxy-state"><span className={`status-dot ${status.configured ? 'online' : 'idle'}`} /><div><strong>{status.configured ? 'Proxy ready' : 'Setup needed'}</strong><code>{status.proxy_url}</code></div></div>
    <div className="section-label"><span>SESSIONS</span><button title="Load demo" onClick={onDemo}><Play size={14} /></button></div>
    <div className="session-list">{status.sessions.map((entry) => <button key={entry.id} className={`session-link ${selectedSession === entry.id ? 'active' : ''}`} onClick={() => onOpenSession(entry.id)}><MessageSquareText size={16} /><span><strong>{entry.name}</strong><small>{compactNumber(entry.token_count, true)} tokens · {timeAgo(entry.updated_at)}</small></span></button>)}{!status.sessions.length && <p className="sidebar-empty">Waiting for the first request.</p>}</div>
    <div className="privacy-note"><ShieldCheck size={16} /><div><strong>Local-only</strong><span>No accounts · no telemetry · no cloud</span></div></div>
    <a className="docs-link" href="/api/docs" target="_blank" rel="noreferrer"><Braces size={15} />API reference</a>
  </aside></>
}

function Topbar({ session, proxyUrl, view, onView, onMenu, onRefresh }: { session: SessionDetail | null; proxyUrl: string; view: ViewName; onView: (view: ViewName) => void; onMenu: () => void; onRefresh: () => void }) {
  const views: Array<[ViewName, string]> = [['context', 'Context'], ['largest', 'Largest'], ['requests', 'Requests'], ['audit', 'Audit'], ['settings', 'Settings']]
  return <header className="topbar"><button className="menu-button" onClick={onMenu}><Menu size={19} /></button><nav>{views.map(([value, label]) => <button key={value} className={view === value ? 'active' : ''} onClick={() => onView(value)}>{label}</button>)}</nav><div className="topbar-tools"><code>{proxyUrl}</code><button title="Refresh" onClick={onRefresh}><RefreshCw size={16} /></button><span className="model-chip"><span className="status-dot online" />{session?.model || 'No model'}</span></div></header>
}

function Dashboard({ session, view, status, onSession, onStatus, onToast, onError }: { session: SessionDetail; view: ViewName; status: StatusData; onSession: (session: SessionDetail) => void; onStatus: () => void; onToast: (message: string) => void; onError: (message: string | null) => void }) {
  const mutate = async (path: string, options: RequestInit, message: string) => {
    try { const result = await api<{ session?: SessionDetail } | SessionDetail>(path, options); const next = 'session' in result && result.session ? result.session : result as SessionDetail; if (next?.id) onSession(next); onStatus(); onToast(message) }
    catch (caught) { onError(caught instanceof Error ? caught.message : 'Action failed') }
  }
  if (view === 'largest') return <LargestView session={session} mutate={mutate} />
  if (view === 'requests') return <RequestsView session={session} />
  if (view === 'audit') return <AuditView session={session} mutate={mutate} />
  if (view === 'settings') return <SettingsView session={session} status={status} mutate={mutate} onToast={onToast} onError={onError} />
  return <ContextView key={session.id} session={session} mutate={mutate} onSession={onSession} onToast={onToast} onError={onError} />
}

function ContextView({ session, mutate, onToast, onError }: { session: SessionDetail; mutate: Mutate; onSession: (session: SessionDetail) => void; onToast: (message: string) => void; onError: (message: string | null) => void }) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [inspected, setInspected] = useState<ContextItem | null>(null)
  const [sort, setSort] = useState<'chronological' | 'largest' | 'type' | 'recommendation'>('chronological')
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [confirmRemove, setConfirmRemove] = useState(false)
  const [summaryDialog, setSummaryDialog] = useState<{ loading: boolean; text: string; source: string[]; original: number; after: number } | null>(null)
  const items = useMemo(() => {
    let result = session.items.filter((item) => filter === 'all' || (filter === 'active' ? !['removed', 'summarized'].includes(item.status) : item.recommendation === filter))
    if (query) result = result.filter((item) => `${item.preview} ${item.text} ${item.kind}`.toLowerCase().includes(query.toLowerCase()))
    return [...result].sort((a, b) => sort === 'largest' ? b.token_count - a.token_count : sort === 'type' ? a.kind.localeCompare(b.kind) : sort === 'recommendation' ? a.recommendation.localeCompare(b.recommendation) : a.sequence - b.sequence)
  }, [session.items, filter, query, sort])
  const selectedItems = session.items.filter((item) => selected.has(item.item_id))
  const selectedTokens = selectedItems.reduce((sum, item) => sum + item.token_count, 0)
  const afterTokens = Math.max(0, session.metrics.active_tokens - selectedItems.filter((item) => !['removed', 'summarized'].includes(item.status)).reduce((sum, item) => sum + item.token_count, 0))
  const toggle = (id: string) => setSelected((current) => { const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next })
  const action = async (kind: string, ids: string[], summary?: string) => { await mutate(`/api/sessions/${session.id}/actions`, { method: 'POST', body: JSON.stringify({ action: kind, item_ids: ids, summary }) }, `${kind[0].toUpperCase()}${kind.slice(1)} applied to context`); setSelected(new Set()); setInspected(null) }
  const draftSummary = async (mode = 'zero_compute', source = [...selected]) => {
    const original = session.items.filter((item) => source.includes(item.item_id)).reduce((sum, item) => sum + item.token_count, 0)
    setSummaryDialog({ loading: true, text: '', source, original, after: 0 })
    try { const draft = await api<{ summary: string; item_ids: string[]; original_tokens: number; summary_tokens: number }>(`/api/sessions/${session.id}/summarize-preview`, { method: 'POST', body: JSON.stringify({ item_ids: source, mode }) }); setSummaryDialog({ loading: false, text: draft.summary, source: draft.item_ids, original: draft.original_tokens, after: draft.summary_tokens }) }
    catch (caught) { setSummaryDialog(null); onError(caught instanceof Error ? caught.message : 'Summary draft failed') }
  }

  return <div className="page-content">
    <section className="context-hero"><div className="hero-heading"><div><span className="eyebrow">MODEL / BACKEND</span><h1>{session.model}</h1><p><Server size={15} />{session.backend?.name || 'Demo data'}<span>·</span>{session.backend?.base_url || 'No live connection'}</p></div>{session.name.startsWith('Demo') && <span className="demo-badge">DEMO DATA</span>}</div><ContextMeter session={session} /></section>
    <div className="context-cockpit"><ContextWorld session={session} onToast={onToast} onError={onError} /><CleanupCommand session={session} mutate={mutate} /></div>
    <div className="overview-grid"><Breakdown session={session} /><Recommendations session={session} onSelect={(id) => { setSelected(new Set([id])); setInspected(session.items.find((item) => item.item_id === id) || null) }} /><HardwareCard session={session} /></div>
    <Timeline session={session} onSelect={(item) => setInspected(item)} />
    <section className="panel context-list-panel"><div className="panel-header table-header"><div><span className="eyebrow">CONTEXT OBJECTS</span><h2>What reaches the model</h2></div><div className="table-tools"><label className="search-field"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search context" /></label><label className="select-field"><ArrowDownUp size={14} /><select value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="chronological">Chronological</option><option value="largest">Largest first</option><option value="type">Type</option><option value="recommendation">Recommendation</option></select><ChevronDown size={13} /></label></div></div>
      <div className="filter-row">{[['all', 'All'], ['active', 'Active'], ['safe_to_remove', 'Safe to remove'], ['summarize', 'Summarize'], ['pinned_critical', 'Critical']].map(([value, label]) => <button key={value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}</div>
      <div className="context-table" role="table" aria-label="Context objects"><div className="context-row context-row-head" role="row"><span /><span>Context item</span><span>Type</span><span>Tokens</span><span>Recommendation</span><span>Status</span></div>{items.map((item) => <ContextRow key={item.item_id} item={item} selected={selected.has(item.item_id)} onToggle={() => toggle(item.item_id)} onInspect={() => setInspected(item)} />)}{!items.length && <div className="no-results">No context objects match this filter.</div>}</div>
    </section>
    {selected.size > 0 && <div className="selection-bar"><span><strong>{selected.size}</strong> selected · {compactNumber(selectedTokens, true)} tokens</span><div><button onClick={() => setSelected(new Set())}>Clear</button><button onClick={() => void draftSummary()}><Sparkles size={15} />Summarize</button><button className="danger-button" onClick={() => setConfirmRemove(true)}><Trash2 size={15} />Remove selected</button></div></div>}
    {inspected && <Inspector item={inspected} session={session} onClose={() => setInspected(null)} onAction={(kind) => void action(kind, [inspected.item_id])} />}
    {confirmRemove && <ReviewDialog title="Remove selected context?" before={session.metrics.active_tokens} after={afterTokens} onCancel={() => setConfirmRemove(false)} onApply={() => { setConfirmRemove(false); void action('remove', [...selected]) }} />}
    {summaryDialog && <SummaryDialog data={summaryDialog} setData={setSummaryDialog} backendLabel={session.backend ? `${session.model} via ${session.backend.name}` : null} onLocalModel={() => void draftSummary('llm', summaryDialog.source)} onCancel={() => setSummaryDialog(null)} onApply={() => { const dialog = summaryDialog; setSummaryDialog(null); void action('summarize', dialog.source, dialog.text); onToast('Summary will replace its source context on the next request') }} />}
  </div>
}

function ContextMeter({ session }: { session: SessionDetail }) {
  const percent = Math.min(100, session.metrics.context_percent); const budgetPosition = Math.min(100, session.context_budget / session.context_limit * 100)
  return <div className="context-meter"><div className="meter-numbers"><span>Context</span><strong>{compactNumber(session.metrics.active_tokens, session.metrics.counts_are_estimates)} <em>/</em> {compactNumber(session.context_limit)}</strong><span>{percent}% of model limit</span></div><div className="meter-track"><div className={`meter-fill ${percent > 85 ? 'danger' : percent > 70 ? 'warn' : ''}`} style={{ width: `${percent}%` }} /><i style={{ left: `${budgetPosition}%` }} title={`Budget: ${session.context_budget.toLocaleString()} tokens`} /></div><div className="meter-legend"><span>{session.metrics.counts_are_estimates ? 'Estimated token attribution' : 'Backend tokenizer counts'}</span><span>Budget marker {compactNumber(session.context_budget)}</span></div></div>
}
function ContextWorld({ session, onToast, onError }: { session: SessionDetail; onToast: (message: string) => void; onError: (message: string | null) => void }) {
  const [launching, setLaunching] = useState(false)
  const entries = Object.entries(session.metrics.categories).filter(([label, value]) => label !== 'Free' && value > 0)
  const total = Math.max(1, session.metrics.active_tokens)
  const launch = async () => { setLaunching(true); try { const result = await api<{ launched: boolean; url: string }>('/api/overlay/launch', { method: 'POST', body: JSON.stringify({ session_id: session.id }) }); if (!result.launched) window.open(result.url, 'lcm-context-overlay', 'popup,width=448,height=700'); onToast(result.launched ? 'Always-on-top context window launched' : 'Context window opened in your browser') } catch (caught) { onError(caught instanceof Error ? caught.message : 'Overlay could not be launched') } finally { setLaunching(false) } }
  return <section className="panel context-world-panel"><div className="panel-header"><div><span className="eyebrow">CONTEXT WORLD</span><h2>What occupies the model's mind</h2></div><button className="overlay-launch" disabled={launching} onClick={() => void launch()}>{launching ? <RefreshCw className="spin" size={15} /> : <SquareStack size={15} />}Live overlay</button></div><div className="world-layout"><div className="context-world" role="img" aria-label={`Context world showing ${entries.map(([label, value]) => `${label} ${value} tokens`).join(', ')}`}><svg className="world-orbits" viewBox="0 0 320 320" aria-hidden="true"><circle cx="160" cy="160" r="118" /><circle cx="160" cy="160" r="82" /><circle cx="160" cy="160" r="48" /></svg><div className="world-core"><BrainCircuit size={25} /><strong>{compactNumber(session.metrics.active_tokens, session.metrics.counts_are_estimates)}</strong><span>live tokens</span><small>{session.metrics.context_percent}% full</small></div>{entries.map(([label, value], index) => { const angle = -Math.PI / 2 + index * (Math.PI * 2 / Math.max(1, entries.length)); const distance = entries.length > 5 ? 38 : 36; const size = Math.max(42, Math.min(72, 38 + Math.sqrt(value / total) * 62)); return <div key={label} className="world-node" title={`${label}: ${value.toLocaleString()} tokens`} style={{ left: `${50 + Math.cos(angle) * distance}%`, top: `${50 + Math.sin(angle) * distance}%`, width: `${size}px`, height: `${size}px`, borderColor: categoryColors[label], background: `color-mix(in srgb, ${categoryColors[label]} 18%, #11151a)` }}><i style={{ background: categoryColors[label] }} /><strong>{Math.round(value / total * 100)}%</strong><span>{label.split(' ')[0]}</span></div> })}</div><div className="world-legend">{Object.entries(session.metrics.categories).filter(([, value]) => value > 0).map(([label, value]) => <div key={label}><span><i style={{ background: categoryColors[label] }} />{label}</span><strong>{compactNumber(value, label !== 'Free' && session.metrics.counts_are_estimates)}</strong><small>{Math.round(value / Math.max(1, session.context_limit) * 100)}%</small></div>)}</div></div></section>
}
function CleanupCommand({ session, mutate }: { session: SessionDetail; mutate: Mutate }) {
  const [busy, setBusy] = useState(false)
  const mode = session.auto_cleanup_mode || 'conservative'
  const active = session.items.filter((item) => item.status === 'active')
  const fingerprints = new Set<string>(); const duplicates = new Set<string>()
  active.forEach((item) => { if (fingerprints.has(item.fingerprint)) duplicates.add(item.item_id); else fingerprints.add(item.fingerprint) })
  const candidates = active.filter((item) => duplicates.has(item.item_id) || (mode !== 'conservative' && item.recommendation === 'safe_to_remove') || (mode === 'aggressive' && ['summarize', 'review_manually'].includes(item.recommendation)))
  const potential = candidates.reduce((sum, item) => sum + item.token_count, 0)
  const update = async (data: { auto_mode?: boolean; auto_cleanup_mode?: CleanupMode }, message: string) => { setBusy(true); try { await mutate(`/api/sessions/${session.id}/policy`, { method: 'PATCH', body: JSON.stringify(data) }, message) } finally { setBusy(false) } }
  const cleanNow = async () => { setBusy(true); try { await mutate(`/api/sessions/${session.id}/auto-cleanup`, { method: 'POST', body: JSON.stringify({ mode }) }, `${cleanupModes.find((entry) => entry.value === mode)?.label} cleanup complete`) } finally { setBusy(false) } }
  return <section className={`panel cleanup-command ${session.auto_mode ? 'is-live' : ''}`}><div className="cleanup-glow" /><div className="panel-header"><div><span className="eyebrow">CLEANUP CONTROL</span><h2>Keep the window clear</h2></div><span className={`live-badge ${session.auto_mode ? 'on' : ''}`}><i />{session.auto_mode ? 'LIVE' : 'OFF'}</span></div><label className="cleanup-master"><span><strong>Real-time cleanup</strong><small>{session.auto_mode ? 'Runs locally before every model request.' : 'Turn on to clean every request automatically.'}</small></span><input type="checkbox" checked={session.auto_mode} disabled={busy} onChange={(event) => void update({ auto_mode: event.target.checked }, event.target.checked ? 'Real-time cleanup enabled' : 'Real-time cleanup paused')} /><i><Power size={13} /></i></label><div className="cleanup-modes" role="group" aria-label="Cleanup mode">{cleanupModes.map((entry) => <button key={entry.value} className={mode === entry.value ? 'active' : ''} disabled={busy} onClick={() => void update({ auto_cleanup_mode: entry.value }, `${entry.label} cleanup selected`)}><span>{entry.label}</span><strong>{entry.short}</strong><small>{entry.description}</small></button>)}</div><div className="cleanup-impact"><div><span>Ready to clean</span><strong>{candidates.length} object{candidates.length === 1 ? '' : 's'}</strong></div><div><span>Potential space</span><strong>{compactNumber(potential, true)} tokens</strong></div></div><button className="cleanup-now" disabled={busy || candidates.length === 0} onClick={() => void cleanNow()}>{busy ? <RefreshCw className="spin" size={17} /> : <Sparkles size={17} />}<span><strong>Clean now</strong><small>Apply {mode} policy immediately</small></span><ChevronDown className="cleanup-arrow" size={17} /></button><p className="cleanup-safety"><ShieldCheck size={14} />Pinned, instruction, and current-user content is always protected.</p></section>
}
function Breakdown({ session }: { session: SessionDetail }) {
  const entries = Object.entries(session.metrics.categories); const denominator = session.context_limit || 1
  return <section className="panel breakdown-panel"><div className="panel-header"><div><span className="eyebrow">CONTEXT BREAKDOWN</span><h2>Composition</h2></div><Gauge size={19} /></div><div className="stacked-bar">{entries.map(([label, value]) => <i key={label} title={`${label}: ${value.toLocaleString()}`} style={{ width: `${Math.max(0.4, value / denominator * 100)}%`, background: categoryColors[label] }} />)}</div><div className="breakdown-list">{entries.map(([label, value]) => <div key={label}><span><i style={{ background: categoryColors[label] }} />{label}</span><strong>{compactNumber(value, label !== 'Free' && session.metrics.counts_are_estimates)}</strong></div>)}</div></section>
}
function Recommendations({ session, onSelect }: { session: SessionDetail; onSelect: (id: string) => void }) {
  const candidates = session.items.filter((item) => ['safe_to_remove', 'summarize', 'review_manually'].includes(item.recommendation) && !['removed', 'summarized'].includes(item.status)).sort((a, b) => b.token_count - a.token_count).slice(0, 3); const potential = candidates.reduce((sum, item) => sum + item.token_count, 0)
  return <section className="panel recommendation-panel"><div className="panel-header"><div><span className="eyebrow">ZERO-COMPUTE REVIEW</span><h2>Potential cleanup</h2></div><span className="saving-pill">{compactNumber(potential, true)} tokens</span></div>{candidates.map((item) => <button className="recommendation-item" key={item.item_id} onClick={() => onSelect(item.item_id)}><span className={`recommendation-icon ${item.recommendation}`}><Zap size={15} /></span><span><strong>{item.preview}</strong><small>{item.recommendation_reason}</small></span><b>{compactNumber(item.token_count, !item.token_exact)}</b></button>)}{!candidates.length && <div className="healthy-state"><CheckCircle2 size={19} /><span><strong>No obvious waste detected</strong><small>LCM found no high-confidence cleanup candidates.</small></span></div>}</section>
}
function HardwareCard({ session }: { session: SessionDetail }) {
  const kv = session.metrics.kv_cache; const saving = kv && session.metrics.kv_cache_incoming ? Math.max(0, session.metrics.kv_cache_incoming.gib - kv.gib) : null
  return <section className="panel hardware-panel"><div className="panel-header"><div><span className="eyebrow">LOCAL COST</span><h2>Hardware impact</h2></div><HardDrive size={19} /></div>{kv ? <><div className="kv-number"><strong>~{kv.gib.toFixed(2)} GB</strong><span>estimated KV cache</span></div><div className="cost-row"><span>Context removed</span><strong>{compactNumber(session.metrics.removed_tokens, true)}</strong></div><div className="cost-row"><span>Estimated KV saved</span><strong>~{saving?.toFixed(2)} GB</strong></div><details><summary>Calculation assumptions</summary><p>{kv.formula}</p><code>{JSON.stringify(kv.assumptions, null, 2)}</code></details></> : <div className="unavailable-state"><Info size={18} /><span><strong>KV estimate unavailable</strong><small>Add layers, KV heads, head dimension, and precision in Settings. LCM will not invent the number.</small></span></div>}<div className="cost-footnote"><Activity size={14} />Potential prompt work reduction: {session.metrics.incoming_tokens ? Math.round(session.metrics.removed_tokens / session.metrics.incoming_tokens * 100) : 0}%</div></section>
}
function Timeline({ session, onSelect }: { session: SessionDetail; onSelect: (item: ContextItem) => void }) {
  const active = session.items.filter((item) => !['removed', 'summarized'].includes(item.status)); const total = Math.max(1, active.reduce((sum, item) => sum + item.token_count, 0))
  return <section className="panel timeline-panel"><div className="panel-header"><div><span className="eyebrow">CONTEXT TIMELINE</span><h2>How the window accumulated</h2></div><Clock3 size={18} /></div><div className="timeline-scale"><span>0</span><span>{compactNumber(total / 4, true)}</span><span>{compactNumber(total / 2, true)}</span><span>{compactNumber(total * .75, true)}</span><span>{compactNumber(total, true)}</span></div><div className="timeline-track">{active.map((item) => <button key={item.item_id} style={{ width: `${Math.max(1.2, item.token_count / total * 100)}%`, background: categoryColors[item.kind === 'code_or_file' ? 'Code / files' : item.kind.includes('tool') ? 'Tool outputs' : item.kind.includes('instruction') ? 'System' : 'Conversation'] }} title={`${item.preview} · ${item.token_count} tokens`} onClick={() => onSelect(item)} />)}</div><div className="timeline-spikes">{[...active].sort((a, b) => b.token_count - a.token_count).slice(0, 3).map((item) => <button key={item.item_id} onClick={() => onSelect(item)}><span>{kindLabel(item.kind)}</span><strong>{item.preview.slice(0, 34)}</strong><small>+{compactNumber(item.token_count, !item.token_exact)}</small></button>)}</div></section>
}
function ContextRow({ item, selected, onToggle, onInspect }: { item: ContextItem; selected: boolean; onToggle: () => void; onInspect: () => void }) {
  return <div className={`context-row ${item.status === 'removed' || item.status === 'summarized' ? 'muted' : ''}`} role="row"><label className="checkbox"><input type="checkbox" checked={selected} onChange={onToggle} /><span><Check size={12} /></span></label><button className="item-title" onClick={onInspect}><strong>{item.preview || kindLabel(item.kind)}</strong><small>#{item.sequence + 1} · {item.role} · {item.source}</small></button><span className="type-cell"><KindIcon kind={item.kind} />{kindLabel(item.kind)}</span><strong className="token-cell">{compactNumber(item.token_count, !item.token_exact)}</strong><span className={`recommendation-tag ${item.recommendation}`}>{recommendationLabels[item.recommendation]}</span><span className={`status-tag ${item.status}`}>{item.status === 'active' ? 'Keep' : item.status}</span></div>
}
function KindIcon({ kind }: { kind: string }) { if (kind === 'code_or_file') return <FileCode2 size={15} />; if (kind.includes('tool')) return <Braces size={15} />; if (kind.includes('instruction')) return <LockKeyhole size={15} />; if (kind.includes('rag') || kind.includes('memory')) return <Database size={15} />; return <MessageSquareText size={15} /> }

function Inspector({ item, session, onClose, onAction }: { item: ContextItem; session: SessionDetail; onClose: () => void; onAction: (kind: string) => void }) {
  const percent = item.token_count / Math.max(1, session.metrics.active_tokens) * 100
  return <div className="drawer-scrim" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose() }}><aside className="inspector"><div className="inspector-header"><div><span className="eyebrow">CONTEXT INSPECTOR</span><h2>{kindLabel(item.kind)}</h2></div><button onClick={onClose}><PanelRightClose size={19} /></button></div><div className="inspector-metrics"><div><span>Tokens</span><strong>{compactNumber(item.token_count, !item.token_exact)}</strong><small>{item.token_exact ? 'Exact' : 'Estimated'} via {item.token_source.replaceAll('_', ' ')}</small></div><div><span>Context share</span><strong>{percent.toFixed(1)}%</strong><small>of forwarded context</small></div><div><span>Potential saving</span><strong>{compactNumber(item.token_count, true)}</strong><small>if removed</small></div></div><section><span className="eyebrow">WHY IS THIS IN CONTEXT?</span><div className="reason-box"><Link2 size={16} /><span><strong>{item.reason}</strong><small>Fingerprint {item.fingerprint}</small></span></div></section><section><span className="eyebrow">RECOMMENDATION</span><div className={`recommendation-explanation ${item.recommendation}`}><strong>{recommendationLabels[item.recommendation]}</strong><p>{item.recommendation_reason}</p></div></section>{item.relationships.length > 0 && <section><span className="eyebrow">DEPENDENCIES</span>{item.relationships.map((relation) => <div className="dependency" key={`${relation.type}-${relation.item_id}`}><Link2 size={14} />{relation.type.replaceAll('_', ' ')} → {relation.item_id}</div>)}<p className="helper-text">Linked tool-call objects are changed atomically.</p></section>}<section className="content-section"><span className="eyebrow">FULL CONTENT</span><pre>{item.text || JSON.stringify(item.raw_message, null, 2)}</pre></section><details className="raw-details"><summary>Raw message JSON</summary><pre>{JSON.stringify(item.raw_message, null, 2)}</pre></details><div className="inspector-actions">{item.status === 'removed' || item.status === 'summarized' ? <button className="primary-button" onClick={() => onAction('restore')}><ArchiveRestore size={16} />Restore</button> : <button className="danger-outline" onClick={() => onAction('remove')}><Trash2 size={16} />Remove</button>}{item.status === 'pinned' ? <button onClick={() => onAction('unpin')}><PinOff size={16} />Unpin</button> : <button onClick={() => onAction('pin')}><Pin size={16} />Pin</button>}<button onClick={() => onAction('keep')}><Check size={16} />Keep</button></div></aside></div>
}

function ReviewDialog({ title, before, after, onCancel, onApply }: { title: string; before: number; after: number; onCancel: () => void; onApply: () => void }) {
  const saving = Math.max(0, before - after)
  return <div className="modal-scrim"><div className="modal review-modal" role="dialog" aria-modal="true"><div className="modal-icon danger"><Trash2 size={21} /></div><h2>{title}</h2><p>This policy changes the next payload sent to the backend. It does not modify the original frontend history.</p><div className="before-after"><div><span>BEFORE</span><strong>{compactNumber(before, true)}</strong><small>tokens</small></div><div className="arrow">→</div><div><span>AFTER</span><strong>{compactNumber(after, true)}</strong><small>tokens</small></div></div><div className="saving-summary"><strong>{compactNumber(saving, true)} tokens saved</strong><span>{before ? (saving / before * 100).toFixed(1) : 0}% reduction · Undo remains available</span></div><div className="modal-actions"><button onClick={onCancel}>Cancel</button><button className="danger-button" onClick={onApply}>Apply removal</button></div></div></div>
}
function SummaryDialog({ data, setData, backendLabel, onLocalModel, onCancel, onApply }: { data: { loading: boolean; text: string; source: string[]; original: number; after: number }; setData: (data: { loading: boolean; text: string; source: string[]; original: number; after: number }) => void; backendLabel: string | null; onLocalModel: () => void; onCancel: () => void; onApply: () => void }) {
  return <div className="modal-scrim"><div className="modal summary-modal" role="dialog" aria-modal="true"><div className="modal-icon"><Sparkles size={21} /></div><h2>Edit context summary</h2><p>Review the draft before replacing {data.source.length} source object{data.source.length === 1 ? '' : 's'}. Local-model mode uses {backendLabel || 'a configured local backend'} and never calls cloud inference.</p>{data.loading ? <div className="summary-loading"><RefreshCw className="spin" size={18} />Building the summary locally…</div> : <><textarea value={data.text} onChange={(event) => setData({ ...data, text: event.target.value })} rows={12} /><div className="summary-stats"><span>Original <strong>{compactNumber(data.original, true)}</strong></span><span>Draft <strong>{compactNumber(data.after, true)}</strong></span><span>Saving <strong>{compactNumber(Math.max(0, data.original - data.after), true)}</strong></span></div></>}<div className="modal-actions"><button onClick={onCancel}>Cancel</button><button disabled={data.loading || !backendLabel} onClick={onLocalModel}><Sparkles size={14} />Use local model</button><button className="primary-button" disabled={data.loading || !data.text.trim()} onClick={onApply}>Apply summary</button></div></div></div>
}

function LargestView({ session, mutate }: { session: SessionDetail; mutate: Mutate }) {
  const [inspected, setInspected] = useState<ContextItem | null>(null); const active = session.items.filter((item) => !['removed', 'summarized'].includes(item.status)).sort((a, b) => b.token_count - a.token_count); const duplicates = active.filter((item, index, all) => all.findIndex((candidate) => candidate.fingerprint === item.fingerprint) !== index)
  const groups = [{ title: 'Exact duplicate blocks', items: duplicates, icon: <SquareStack size={17} /> }, { title: 'Large tool outputs', items: active.filter((item) => item.kind.includes('tool') && item.token_count > 500), icon: <Braces size={17} /> }, { title: 'Old conversation turns', items: active.filter((item) => item.recommendation === 'summarize'), icon: <History size={17} /> }]
  return <div className="page-content"><PageHeading eyebrow="CONTEXT CONSUMERS" title="What is taking up my context?" copy="Largest objects and structural waste, ranked without running a model." /><section className="panel ranking-panel"><div className="panel-header"><div><span className="eyebrow">LARGEST INDIVIDUAL ITEMS</span><h2>Top consumers</h2></div><span className="saving-pill">{active.length} active</span></div><div className="ranking-list">{active.slice(0, 10).map((item, index) => <button key={item.item_id} onClick={() => setInspected(item)}><span className="rank">{String(index + 1).padStart(2, '0')}</span><span><strong>{item.preview}</strong><small>{kindLabel(item.kind)} · {recommendationLabels[item.recommendation]}</small></span><b>{compactNumber(item.token_count, !item.token_exact)}</b><i style={{ width: `${item.token_count / Math.max(1, active[0]?.token_count) * 100}%` }} /></button>)}</div></section><div className="consumer-grid">{groups.map((group) => <section className="panel" key={group.title}><div className="panel-header compact"><h2>{group.icon}{group.title}</h2><span>{group.items.length}</span></div>{group.items.slice(0, 4).map((item) => <button className="compact-item" key={item.item_id} onClick={() => setInspected(item)}><span>{item.preview}</span><strong>{compactNumber(item.token_count, !item.token_exact)}</strong></button>)}{!group.items.length && <p className="empty-copy">Nothing detected in this category.</p>}</section>)}</div>{inspected && <Inspector item={inspected} session={session} onClose={() => setInspected(null)} onAction={(kind) => void mutate(`/api/sessions/${session.id}/actions`, { method: 'POST', body: JSON.stringify({ action: kind, item_ids: [inspected.item_id] }) }, `${kind} applied`)} />}</div>
}

function RequestsView({ session }: { session: SessionDetail }) {
  const [active, setActive] = useState(session.requests[0]?.id || ''); const [mode, setMode] = useState<'diff' | 'incoming' | 'forwarded'>('diff'); const selected = session.requests.find((request) => request.id === active) || session.requests[0]
  return <div className="page-content"><PageHeading eyebrow="REQUEST INSPECTOR" title="Trust, byte for byte" copy="Compare what the frontend sent with the payload LCM forwarded to your local model." />{!selected ? <section className="panel empty-panel"><Activity size={25} /><h2>No proxied requests yet</h2><p>The demo contains context, but request timing appears after a real backend call.</p></section> : <div className="request-layout"><aside className="panel request-list">{session.requests.map((request) => <button className={request.id === selected.id ? 'active' : ''} key={request.id} onClick={() => setActive(request.id)}><span><strong>{new Date(request.created_at).toLocaleTimeString()}</strong><small>{request.status_code || 'pending'} · {request.total_ms ? `${Math.round(request.total_ms)} ms` : 'in progress'}</small></span><b>-{compactNumber(request.incoming_tokens - request.forwarded_tokens, true)}</b></button>)}</aside><section className="panel request-inspector"><div className="request-metrics"><Metric label="Incoming" value={compactNumber(selected.incoming_tokens, true)} /><Metric label="Forwarded" value={compactNumber(selected.forwarded_tokens, true)} /><Metric label="Difference" value={`-${compactNumber(selected.incoming_tokens - selected.forwarded_tokens, true)}`} accent /><Metric label="TTFT" value={selected.ttft_ms ? `${Math.round(selected.ttft_ms)} ms` : 'Unavailable'} /></div><div className="segmented-control"><button className={mode === 'diff' ? 'active' : ''} onClick={() => setMode('diff')}>Readable diff</button><button className={mode === 'incoming' ? 'active' : ''} onClick={() => setMode('incoming')}>Incoming JSON</button><button className={mode === 'forwarded' ? 'active' : ''} onClick={() => setMode('forwarded')}>Forwarded JSON</button></div>{mode === 'diff' ? <ReadableDiff request={selected} /> : <pre className="json-view">{JSON.stringify(mode === 'incoming' ? selected.incoming : selected.forwarded, null, 2)}</pre>}</section></div>}</div>
}
function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) { return <div className={accent ? 'accent' : ''}><span>{label}</span><strong>{value}</strong></div> }
function ReadableDiff({ request }: { request: RequestRecord }) {
  const incoming = (request.incoming.messages || []) as Array<Record<string, unknown>>; const forwarded = (request.forwarded.messages || []) as Array<Record<string, unknown>>; const serialized = new Set(forwarded.map((message) => JSON.stringify(message)))
  return <div className="readable-diff"><div className="diff-summary"><span className="minus">−{request.incoming_tokens - request.forwarded_tokens}</span> estimated tokens removed before inference</div>{incoming.map((message, index) => { const kept = serialized.has(JSON.stringify(message)); return <div className={`diff-row ${kept ? 'kept' : 'removed'}`} key={index}><span>{kept ? <Check size={14} /> : <Trash2 size={14} />}</span><strong>{String(message.role || 'unknown')}</strong><p>{typeof message.content === 'string' ? message.content.slice(0, 240) : JSON.stringify(message.content).slice(0, 240)}</p><b>{kept ? 'FORWARDED' : 'REMOVED'}</b></div>})}</div>
}

function AuditView({ session, mutate }: { session: SessionDetail; mutate: Mutate }) {
  return <div className="page-content"><PageHeading eyebrow="LOCAL AUDIT LOG" title="Every context decision" copy="Manual and automatic changes are recorded locally and remain inspectable." actions={<button className="primary-button" onClick={() => void mutate(`/api/sessions/${session.id}/undo`, { method: 'POST' }, 'Last action undone')}><Undo2 size={16} />Undo latest</button>} /><section className="panel audit-panel">{session.audit.map((entry) => <div className="audit-entry" key={entry.id}><span className={`audit-icon ${entry.action}`}><History size={16} /></span><div><strong>{entry.action.replaceAll('_', ' ')}</strong><p>{entry.details.actor ? `${entry.details.actor} · ` : ''}{entry.details.item_ids?.length ? `${entry.details.item_ids.length} context object${entry.details.item_ids.length === 1 ? '' : 's'}` : entry.details.original_action ? `Reversed ${entry.details.original_action}` : 'Context policy updated'}</p></div><div><strong>{entry.details.token_delta ? `${entry.details.token_delta.toLocaleString()} tokens` : '—'}</strong><small>{new Date(entry.created_at).toLocaleString()}</small></div></div>)}{!session.audit.length && <div className="no-results">No context actions have been recorded.</div>}</section></div>
}

function SettingsView({ session, status, mutate, onToast, onError }: { session: SessionDetail; status: StatusData; mutate: Mutate; onToast: (message: string) => void; onError: (message: string | null) => void }) {
  const [budget, setBudget] = useState(session.context_budget); const [limit, setLimit] = useState(session.context_limit); const [auto, setAuto] = useState(session.auto_mode); const [cleanupMode, setCleanupMode] = useState<CleanupMode>(session.auto_cleanup_mode || 'conservative'); const [copied, setCopied] = useState(false)
  const savePolicy = () => void mutate(`/api/sessions/${session.id}/policy`, { method: 'PATCH', body: JSON.stringify({ context_budget: budget, context_limit: limit, auto_mode: auto, auto_cleanup_mode: cleanupMode }) }, 'Context policy saved')
  const copy = async () => { await navigator.clipboard.writeText(status.proxy_url); setCopied(true); onToast('Proxy endpoint copied') }
  return <div className="page-content"><PageHeading eyebrow="LOCAL CONFIGURATION" title="Context policy & diagnostics" copy="Budgets limit practical context use without changing the model's advertised maximum." /><div className="settings-grid"><section className="panel settings-card"><div className="panel-header"><div><span className="eyebrow">ENDPOINT</span><h2>Proxy connection</h2></div><Server size={19} /></div><label>LCM endpoint<div className="copy-field"><code>{status.proxy_url}</code><button onClick={() => void copy()}>{copied ? <Check size={16} /> : <Clipboard size={16} />}</button></div></label><div className="backend-summary"><span className="status-dot online" /><div><strong>{session.backend?.name || 'Demo mode'}</strong><small>{session.backend?.base_url || 'No live backend attached'}</small></div></div><p className="helper-text">Point your existing OpenAI-compatible frontend here. LCM forwards unknown sampling and model fields unchanged.</p></section><section className="panel settings-card"><div className="panel-header"><div><span className="eyebrow">CONTEXT LIMITS</span><h2>Budget & cleanup</h2></div><Gauge size={19} /></div><label>Model context limit<input type="number" min="512" value={limit} onChange={(event) => setLimit(Number(event.target.value))} /></label><label>Working context budget<input type="number" min="256" max={limit} value={budget} onChange={(event) => setBudget(Number(event.target.value))} /></label><label className="toggle-row"><span><strong>Real-time cleanup</strong><small>Apply the selected local policy before every model request.</small></span><input type="checkbox" checked={auto} onChange={(event) => setAuto(event.target.checked)} /><i /></label><div className="settings-mode-grid">{cleanupModes.map((entry) => <button key={entry.value} className={cleanupMode === entry.value ? 'active' : ''} onClick={() => setCleanupMode(entry.value)}><strong>{entry.label}</strong><small>{entry.short}</small></button>)}</div><button className="primary-button" onClick={savePolicy}>Save policy</button></section><section className="panel settings-card"><div className="panel-header"><div><span className="eyebrow">PRIVACY</span><h2>Local guarantees</h2></div><ShieldCheck size={19} /></div><ul className="check-list"><li><Check size={15} />Binds to 127.0.0.1 by default</li><li><Check size={15} />Only private/local backend URLs accepted</li><li><Check size={15} />No analytics or telemetry code</li><li><Check size={15} />No cloud-model fallback</li><li><Check size={15} />Original frontend history is never mutated</li></ul></section><section className="panel settings-card danger-zone"><div className="panel-header"><div><span className="eyebrow">SESSION POLICY</span><h2>Reset overlay</h2></div><RotateCcw size={19} /></div><p>Restore removed and summarized objects, clear pins, and deactivate generated summaries for this session.</p><button className="danger-outline" onClick={() => void mutate(`/api/sessions/${session.id}/actions`, { method: 'POST', body: JSON.stringify({ action: 'reset', item_ids: [] }) }, 'Context overlay reset')}><RotateCcw size={16} />Reset context policy</button></section></div><SessionDataControls session={session} mutate={mutate} onError={onError} /><BackendEditor backend={status.backend} catalog={status.adapter_catalog} onSaved={() => { onToast('Backend configuration saved'); window.location.reload() }} onError={onError} /></div>
}

function SessionDataControls({ session, mutate, onError }: { session: SessionDetail; mutate: Mutate; onError: (message: string | null) => void }) {
  const [name, setName] = useState(session.name)
  const [tokenizer, setTokenizer] = useState(session.tokenizer_family || 'auto')
  const clearSession = async () => {
    if (!window.confirm(`Clear the local session “${session.name}”? This cannot be undone.`)) return
    try { await api(`/api/sessions/${session.id}`, { method: 'DELETE' }); window.location.reload() }
    catch (caught) { onError(caught instanceof Error ? caught.message : 'Session deletion failed') }
  }
  const clearAll = async () => {
    if (!window.confirm('Delete all LCM sessions, policies, requests, audit data, and backend settings? This cannot be undone.')) return
    try { await api('/api/data', { method: 'DELETE' }); window.location.reload() }
    catch (caught) { onError(caught instanceof Error ? caught.message : 'Local data deletion failed') }
  }
  return <section className="panel backend-editor"><div className="panel-header"><div><span className="eyebrow">LOCAL SESSION DATA</span><h2>Name, tokenizer, or clear</h2></div><Database size={19} /></div><div className="backend-form"><label>Session name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Tokenizer / model family<select value={tokenizer} onChange={(event) => setTokenizer(event.target.value)}><option value="auto">Auto from model name</option><option value="qwen">Qwen</option><option value="llama">Llama</option><option value="mistral">Mistral</option><option value="gemma">Gemma</option><option value="phi">Phi</option><option value="deepseek">DeepSeek</option><option value="coder">Code-oriented</option></select></label><div className="backend-buttons"><button onClick={() => void mutate(`/api/sessions/${session.id}`, { method: 'PATCH', body: JSON.stringify({ name }) }, 'Session renamed')}>Rename</button><button onClick={() => void mutate(`/api/sessions/${session.id}/policy`, { method: 'PATCH', body: JSON.stringify({ tokenizer_family: tokenizer }) }, 'Tokenizer family saved')}>Save tokenizer</button><button className="danger-outline" onClick={() => void clearSession()}><Trash2 size={14} />Clear session</button><button className="danger-outline" onClick={() => void clearAll()}><Trash2 size={14} />Delete all local data</button></div></div></section>
}

function BackendEditor({ backend, catalog, onSaved, onError }: { backend: Backend | null; catalog: AdapterCatalogEntry[]; onSaved: () => void; onError: (message: string | null) => void }) {
  const current = catalog.find((entry) => entry.kind === backend?.kind) || catalog[0]; const [kind, setKind] = useState(current.kind); const [url, setUrl] = useState(backend?.base_url || current.default_url); const [name, setName] = useState(backend?.name || current.name); const [testing, setTesting] = useState(false); const [testResult, setTestResult] = useState<string | null>(null)
  const test = async () => { setTesting(true); try { const result = await api<{ ok: boolean; message: string; models: unknown[] }>('/api/backends/test', { method: 'POST', body: JSON.stringify({ name, kind, base_url: url, context_limit: backend?.context_limit || 32768 }) }); setTestResult(result.ok ? `${result.message} · ${result.models.length} model${result.models.length === 1 ? '' : 's'}` : result.message) } catch (caught) { setTestResult(caught instanceof Error ? caught.message : 'Connection failed') } finally { setTesting(false) } }
  const save = async () => { try { await api('/api/backends', { method: 'POST', body: JSON.stringify({ id: backend?.id, name, kind, base_url: url, context_limit: backend?.context_limit || 32768 }) }); onSaved() } catch (caught) { onError(caught instanceof Error ? caught.message : 'Save failed') } }
  return <section className="panel backend-editor"><div className="panel-header"><div><span className="eyebrow">BACKEND ADAPTER</span><h2>Inference server</h2></div><Settings2 size={19} /></div><div className="backend-form"><label>Runtime<select value={kind} onChange={(event) => { const next = catalog.find((entry) => entry.kind === event.target.value)!; setKind(next.kind); setName(next.name); setUrl(next.default_url) }}>{catalog.map((entry) => <option key={entry.kind} value={entry.kind}>{entry.name}</option>)}</select></label><label>Display name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label className="wide">Backend URL<input value={url} onChange={(event) => setUrl(event.target.value)} /></label><div className="backend-buttons"><button onClick={() => void test()} disabled={testing}>{testing ? <RefreshCw className="spin" size={15} /> : <Activity size={15} />}Test connection</button><button className="primary-button" onClick={() => void save()}>Save backend</button></div></div>{testResult && <p className="test-result"><Info size={15} />{testResult}</p>}</section>
}
function PageHeading({ eyebrow, title, copy, actions }: { eyebrow: string; title: string; copy: string; actions?: React.ReactNode }) { return <div className="page-heading"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{copy}</p></div>{actions}</div> }

function SetupFlow({ status, onConfigured, onDemo }: { status: StatusData; onConfigured: (id?: string) => void; onDemo: () => void }) {
  const [step, setStep] = useState(1); const [kind, setKind] = useState('lmstudio'); const current = status.adapter_catalog.find((entry) => entry.kind === kind) || status.adapter_catalog[0]; const [url, setUrl] = useState(current.default_url); const [limit, setLimit] = useState(32768); const [testing, setTesting] = useState(false); const [result, setResult] = useState<{ ok: boolean; message: string; models: Array<{ id?: string }> } | null>(null); const [saved, setSaved] = useState(false)
  const select = (entry: AdapterCatalogEntry) => { setKind(entry.kind); setUrl(entry.default_url); setResult(null); setStep(2) }
  const test = async () => { setTesting(true); try { setResult(await api('/api/backends/test', { method: 'POST', body: JSON.stringify({ name: current.name, kind, base_url: url, context_limit: limit }) })) } catch (caught) { setResult({ ok: false, message: caught instanceof Error ? caught.message : 'Connection failed', models: [] }) } finally { setTesting(false) } }
  const save = async () => { await api('/api/backends', { method: 'POST', body: JSON.stringify({ name: current.name, kind, base_url: url, context_limit: limit }) }); setSaved(true); setStep(3) }
  if (saved || step === 3) return <div className="setup-shell"><SetupBrand /><div className="setup-card setup-success"><div className="success-orbit"><Check size={28} /></div><span className="eyebrow">SETUP COMPLETE</span><h1>Your context runtime is ready</h1><p>Use this base URL in the frontend that currently connects directly to your model server.</p><div className="endpoint-box"><code>{status.proxy_url}</code><button onClick={() => void navigator.clipboard.writeText(status.proxy_url)}><Clipboard size={17} /></button></div><div className="setup-flow-line"><span>Any frontend</span><i>→</i><strong>LCM :7331</strong><i>→</i><span>{current.name}</span></div><button className="primary-button large" onClick={() => onConfigured()}>Open dashboard <LayoutDashboard size={17} /></button></div><PrivacyFooter /></div>
  return <div className="setup-shell"><SetupBrand /><div className="setup-progress"><span className={step >= 1 ? 'active' : ''}>1 <em>Choose backend</em></span><i /><span className={step >= 2 ? 'active' : ''}>2 <em>Connect</em></span><i /><span>3 <em>Ready</em></span></div><div className="setup-card"><span className="eyebrow">FIRST-RUN SETUP</span><h1>{step === 1 ? 'Where does your model run?' : `Connect to ${current.name}`}</h1><p>{step === 1 ? 'LCM sits between your existing frontend and this local inference runtime.' : 'This address must resolve to localhost or a private network host.'}</p>{step === 1 ? <div className="adapter-grid">{status.adapter_catalog.map((entry) => <button key={entry.kind} onClick={() => select(entry)}><Server size={19} /><span><strong>{entry.name}</strong><small>{entry.status}</small></span></button>)}</div> : <div className="connect-form"><label>Backend URL<input value={url} onChange={(event) => setUrl(event.target.value)} autoFocus /></label><label>Model context limit<input type="number" value={limit} min="512" onChange={(event) => setLimit(Number(event.target.value))} /></label>{result && <div className={`connection-result ${result.ok ? 'ok' : 'failed'}`}>{result.ok ? <CheckCircle2 size={17} /> : <CircleAlert size={17} />}<span><strong>{result.ok ? 'Backend found' : 'Could not connect'}</strong><small>{result.message}{result.ok && result.models.length ? ` · ${result.models.map((model) => model.id).join(', ')}` : ''}</small></span></div>}<div className="setup-actions"><button onClick={() => setStep(1)}>Back</button><button onClick={() => void test()} disabled={testing}>{testing ? <RefreshCw className="spin" size={16} /> : <Activity size={16} />}Test connection</button><button className="primary-button" onClick={() => void save()} disabled={!result?.ok}>Save & continue</button></div></div>}<div className="demo-invite"><span>Don’t have a model running?</span><button onClick={onDemo}><Play size={14} />Explore with demo data</button></div></div><PrivacyFooter /></div>
}
function SetupBrand() { return <div className="setup-brand"><div className="brand-mark"><Braces size={23} /></div><div><strong>Local Context Manager</strong><span>Context runtime & debugger</span></div></div> }
function PrivacyFooter() { return <div className="privacy-footer"><ShieldCheck size={15} />Runs entirely on your machine<span>·</span>No account<span>·</span>No telemetry<span>·</span>No cloud</div> }

export default App
