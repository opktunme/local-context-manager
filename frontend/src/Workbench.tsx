import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, Bot, Box, BrainCircuit, Check, ChevronDown, CircleAlert,
  Cpu, Gauge, HardDrive, LayoutDashboard, Menu, MessageSquare, MonitorUp,
  PanelRightClose, PanelRightOpen, Paperclip, Plus, RotateCcw, Send, Settings2,
  ShieldCheck, Sparkles, Square, WandSparkles, X, Zap,
} from 'lucide-react'
import './Workbench.css'

type Profile = {
  id: 'exact' | 'full'; name: string; description: string; precision: string
  max_ram_gib: number; recommended_ram_gib: number; measured_ram_gib: number
  max_vram_gib: number; recommended_vram_gib: number; measured_vram_gib: number
  max_device_slots: number; context_limit: number; executor_context_limit: number
  max_output_tokens: number; installed: boolean
}
type WorkbenchConfig = {
  available: boolean; xtllm_root: string; xtllm_binary_root: string; profiles: Profile[]
  host: { ram_gib: number | null; gpu: string; runtime: string }
  limits: {
    total_ram_gib: number; max_vram_gib: number; context_tokens: number
    executor_context_tokens: number; context_bytes_per_token: number
    device_context_bytes_per_token: number
    device_expert_slot_bytes: number; minimum_device_slots: number; vram_safety_gib: number
    output_tokens: number; note: string
  }
}
type ChatMessage = {
  id: string; role: 'user' | 'assistant'; content: string; status: string
  token_count: number; metrics?: RunMetrics | null; created_at: string
}
type Chat = {
  id: string; title: string; profile: 'exact' | 'full'; ram_budget_gib: number
  context_ram_budget_gib: number; vram_budget_gib: number
  context_limit: number; output_token_limit: number
  auto_mode: boolean; cleanup_mode: CleanupMode; lcm_session_id?: string | null
  archived: boolean; message_count?: number; updated_at: string; messages?: ChatMessage[]
}
type RunMetrics = {
  profile_name?: string; throughput_tps?: number; ram_actual_gib?: number
  vram_actual_gib?: number; prompt_tokens?: number; completion_tokens?: number
  elapsed_ms?: number; precision?: string
}
type CleanupMode = 'conservative' | 'balanced' | 'aggressive'
type RunStatus = { stage: string; message: string; timestamp: number; detail?: string }
type OverlayState = {
  active: boolean; session_id?: string; name?: string; auto_mode?: boolean
  auto_cleanup_mode?: CleanupMode; context_limit?: number
  metrics?: { active_tokens: number; context_percent: number; removed_tokens: number; categories: Record<string, number> }
  segments?: Array<{ id: string; category: string; tokens: number }>
}

const categoryColors: Record<string, string> = {
  System: '#a78bfa', Conversation: '#60a5fa', 'Code / files': '#34d399',
  'Tool outputs': '#f59e62', 'Memory / RAG': '#e879c4', Images: '#22d3ee',
  Summaries: '#b4d455', Free: '#252930',
}
const cleanupModes: Array<{ id: CleanupMode; label: string; description: string }> = [
  { id: 'conservative', label: 'Conservative', description: 'Duplicates only' },
  { id: 'balanced', label: 'Balanced', description: 'High-confidence waste' },
  { id: 'aggressive', label: 'Aggressive', description: 'Compress + prune' },
]

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || 'Request failed')
  }
  return response.json() as Promise<T>
}

function relativeTime(value: string) {
  const minutes = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 60000))
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`
}
function formatTokens(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 100_000 ? 0 : 1)}K` : `${value}`
}

export default function Workbench() {
  const [config, setConfig] = useState<WorkbenchConfig | null>(null)
  const [chats, setChats] = useState<Chat[]>([])
  const [chat, setChat] = useState<Chat | null>(null)
  const [composer, setComposer] = useState('')
  const [running, setRunning] = useState(false)
  const [statuses, setStatuses] = useState<RunStatus[]>([])
  const [overlay, setOverlay] = useState<OverlayState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(true)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsDraft, setSettingsDraft] = useState<Chat | null>(null)
  const booted = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const fileInput = useRef<HTMLInputElement | null>(null)
  const transcriptEnd = useRef<HTMLDivElement | null>(null)

  const loadChat = useCallback(async (id: string) => {
    const detail = await api<Chat>(`/api/workbench/chats/${id}`)
    setChat(detail)
    setStatuses([])
    setSidebarOpen(false)
    return detail
  }, [])
  const refreshChats = useCallback(async () => {
    const result = await api<{ chats: Chat[] }>('/api/workbench/chats')
    setChats(result.chats)
    return result.chats
  }, [])
  const createChat = useCallback(async () => {
    const created = await api<Chat>('/api/workbench/chats', { method: 'POST', body: '{}' })
    setChat(created)
    setChats((current) => [created, ...current])
    setComposer('')
    setStatuses([])
    setOverlay(null)
    setSidebarOpen(false)
    return created
  }, [])

  useEffect(() => {
    if (booted.current) return
    booted.current = true
    void (async () => {
      try {
        const [nextConfig, list] = await Promise.all([
          api<WorkbenchConfig>('/api/workbench/config'),
          api<{ chats: Chat[] }>('/api/workbench/chats'),
        ])
        setConfig(nextConfig)
        setChats(list.chats)
        if (list.chats[0]) await loadChat(list.chats[0].id)
        else await createChat()
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Workbench could not start')
      }
    })()
  }, [createChat, loadChat])
  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: running ? 'smooth' : 'auto' })
  }, [chat?.messages, statuses, running])
  useEffect(() => {
    if (!notice) return
    const timer = window.setTimeout(() => setNotice(null), 3000)
    return () => window.clearTimeout(timer)
  }, [notice])

  const loadOverlay = useCallback(async (sessionId?: string | null) => {
    if (!sessionId) { setOverlay(null); return }
    try { setOverlay(await api<OverlayState>(`/api/overlay-state?session_id=${encodeURIComponent(sessionId)}`)) }
    catch { setOverlay(null) }
  }, [])
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void loadOverlay(chat?.lcm_session_id)
    if (!chat?.lcm_session_id) return
    const timer = window.setInterval(() => void loadOverlay(chat.lcm_session_id), 2500)
    return () => window.clearInterval(timer)
  }, [chat?.lcm_session_id, loadOverlay])

  const patchChat = async (changes: Partial<Chat>, toast?: string) => {
    if (!chat) return
    try {
      const updated = await api<Chat>(`/api/workbench/chats/${chat.id}`, {
        method: 'PATCH', body: JSON.stringify(changes),
      })
      setChat(updated)
      setChats((current) => current.map((entry) => entry.id === updated.id ? { ...entry, ...updated } : entry))
      if (toast) setNotice(toast)
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Settings could not be saved') }
  }

  const sendMessage = async () => {
    const content = composer.trim()
    if (!chat || !content || running) return
    const localMessage: ChatMessage = {
      id: `local-${Date.now()}`, role: 'user', content, status: 'complete', token_count: 0,
      created_at: new Date().toISOString(),
    }
    setComposer('')
    setRunning(true)
    setError(null)
    setStatuses([])
    setChat((current) => current ? { ...current, messages: [...(current.messages || []), localMessage] } : current)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const response = await fetch(`/api/workbench/chats/${chat.id}/messages`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }), signal: controller.signal,
      })
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(body.detail || 'Generation could not start')
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let complete = false
      while (!complete) {
        const { done, value } = await reader.read()
        complete = done
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() || ''
        for (const frame of frames) {
          const data = frame.split('\n').filter((line) => line.startsWith('data: ')).map((line) => line.slice(6)).join('')
          if (!data) continue
          const event = JSON.parse(data) as Record<string, unknown>
          const kind = String(event.event)
          if (kind === 'status' || kind === 'context') {
            setStatuses((current) => [...current, {
              stage: String(event.stage || kind), message: String(event.message || ''), timestamp: Date.now(),
              detail: kind === 'context' ? `${event.forwarded_tokens} forwarded · ${event.removed_tokens} cleaned` : undefined,
            }])
            if (kind === 'context' && event.session_id) {
              setChat((current) => current ? { ...current, lcm_session_id: String(event.session_id) } : current)
            }
          } else if (kind === 'done') {
            setChat(event.chat as Chat)
            setStatuses((current) => [...current, { stage: 'done', message: 'Response complete', timestamp: Date.now() }])
            void refreshChats()
            const doneChat = event.chat as Chat
            void loadOverlay(doneChat.lcm_session_id)
          } else if (kind === 'error') {
            throw new Error(String(event.message || 'LongCat generation failed'))
          }
        }
      }
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') setError(caught instanceof Error ? caught.message : 'Generation failed')
      await loadChat(chat.id).catch(() => undefined)
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }
  const stopGeneration = async () => {
    if (!chat) return
    await api<{ stopped: boolean }>(`/api/workbench/chats/${chat.id}/stop`, { method: 'POST', body: '{}' }).catch(() => ({ stopped: false }))
    abortRef.current?.abort()
    setRunning(false)
    setStatuses((current) => [...current, { stage: 'stopped', message: 'Generation stopped', timestamp: Date.now() }])
  }
  const attachFile = async (file?: File) => {
    if (!file) return
    try {
      const text = (await file.text()).slice(0, 12000)
      setComposer((current) => `${current}${current ? '\n\n' : ''}[File: ${file.name}]\n${text}`)
      setNotice(`${file.name} added to the prompt locally`)
    } catch { setError('That file could not be read as text') }
  }
  const runCleanup = async () => {
    if (!chat?.lcm_session_id) return
    try {
      const result = await api<{ result: { estimated_token_saving: number } }>(`/api/sessions/${chat.lcm_session_id}/auto-cleanup`, {
        method: 'POST', body: JSON.stringify({ mode: chat.cleanup_mode }),
      })
      setNotice(`Cleanup saved about ${result.result.estimated_token_saving} tokens`)
      await loadOverlay(chat.lcm_session_id)
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Cleanup failed') }
  }
  const undoCleanup = async () => {
    if (!chat?.lcm_session_id) return
    try {
      await api(`/api/sessions/${chat.lcm_session_id}/undo`, { method: 'POST', body: '{}' })
      setNotice('Last context action restored')
      await loadOverlay(chat.lcm_session_id)
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Nothing to undo') }
  }
  const launchOverlay = async () => {
    try {
      await api('/api/overlay/launch', { method: 'POST', body: JSON.stringify({ session_id: chat?.lcm_session_id || null }) })
      setNotice('Tiny always-on-top overlay launched')
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Overlay could not launch') }
  }

  const currentProfile = config?.profiles.find((entry) => entry.id === chat?.profile)
  const openSettings = () => { if (chat) { setSettingsDraft({ ...chat }); setSettingsOpen(true) } }
  const applySettings = async () => {
    if (!settingsDraft) return
    await patchChat({
      profile: settingsDraft.profile, ram_budget_gib: settingsDraft.ram_budget_gib,
      context_ram_budget_gib: settingsDraft.context_ram_budget_gib,
      vram_budget_gib: settingsDraft.vram_budget_gib, context_limit: settingsDraft.context_limit,
      output_token_limit: settingsDraft.output_token_limit,
    }, 'Runtime limits saved')
    setSettingsOpen(false)
  }

  if (!config || !chat) return <div className="wb-loading"><div className="cat-mark"><Sparkles size={23} /></div><span>Starting the local workbench</span><div className="wb-loading-line" /></div>

  return <div className="wb-shell">
    {sidebarOpen && <button className="wb-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside className={`wb-sidebar ${sidebarOpen ? 'open' : ''}`}>
      <div className="wb-brand"><div className="cat-mark"><Sparkles size={19} /></div><strong>LongCat</strong><ChevronDown size={14} /></div>
      <button className="new-chat-button" onClick={() => void createChat()}><Plus size={17} />New chat</button>
      <nav className="wb-nav">
        <a href="/context"><BrainCircuit size={16} />Context inspector</a>
        <button onClick={openSettings}><Settings2 size={16} />Runtime settings</button>
      </nav>
      <div className="wb-section-label">RECENTS</div>
      <div className="wb-recents">{chats.map((entry) => <button key={entry.id} className={entry.id === chat.id ? 'active' : ''} onClick={() => void loadChat(entry.id)}><MessageSquare size={15} /><span><strong>{entry.title}</strong><small>{entry.profile === 'exact' ? 'Exact' : 'Full'} · {relativeTime(entry.updated_at)}</small></span></button>)}</div>
      <div className="wb-local-badge"><ShieldCheck size={15} /><span><strong>Local on A:</strong><small>No cloud · no telemetry</small></span></div>
    </aside>

    <main className="wb-main">
      <header className="wb-topbar">
        <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)}><Menu size={18} /></button>
        <div className="chat-heading"><strong>{chat.title}</strong><span><i className={config.available ? 'online' : ''} />{currentProfile?.name} · XTLLM</span></div>
        <div className="topbar-actions">
          <span className="private-chip"><ShieldCheck size={13} />Private</span>
          <button className="icon-button" title="Runtime settings" onClick={openSettings}><Settings2 size={17} /></button>
          <button className="icon-button" title={inspectorOpen ? 'Hide context inspector' : 'Show context inspector'} onClick={() => setInspectorOpen((value) => !value)}>{inspectorOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}</button>
        </div>
      </header>
      {error && <div className="wb-alert"><CircleAlert size={16} /><span>{error}</span><button onClick={() => setError(null)}><X size={15} /></button></div>}
      <section className="transcript">
        {!chat.messages?.length && <Welcome profile={currentProfile} onStarter={setComposer} />}
        {chat.messages?.map((message) => <ChatBubble key={message.id} message={message} />)}
        {!!statuses.length && <ActivityTrail statuses={statuses} running={running} />}
        <div ref={transcriptEnd} />
      </section>
      <div className="composer-wrap">
        <div className={`composer-card ${running ? 'running' : ''}`}>
          <textarea value={composer} onChange={(event) => setComposer(event.target.value)} placeholder="Message LongCat…" rows={2} disabled={running} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendMessage() } }} />
          <div className="composer-tools">
            <div className="composer-left">
              <input ref={fileInput} type="file" hidden onChange={(event) => { void attachFile(event.target.files?.[0]); event.target.value = '' }} />
              <button className="icon-button" title="Attach a local text file" onClick={() => fileInput.current?.click()}><Paperclip size={17} /></button>
              <label className="model-select"><Bot size={15} /><select value={chat.profile} disabled={running} onChange={(event) => void patchChat({ profile: event.target.value as Chat['profile'] }, 'Model profile changed')}><option value="exact">LongCat Exact</option><option value="full">LongCat Full</option></select><ChevronDown size={12} /></label>
              <button className="token-chip" onClick={openSettings}>{chat.output_token_limit} max out</button>
            </div>
            {running ? <button className="send-button stop" title="Stop generation" onClick={() => void stopGeneration()}><Square size={14} fill="currentColor" /></button> : <button className="send-button" title="Send" disabled={!composer.trim()} onClick={() => void sendMessage()}><Send size={16} /></button>}
          </div>
        </div>
        <p className="composer-note">Local model output can be inaccurate. Context status messages are UI-only and use no model tokens.</p>
      </div>
    </main>

    {inspectorOpen && <ContextInspector chat={chat} overlay={overlay} onPatch={patchChat} onClean={runCleanup} onUndo={undoCleanup} onLaunch={launchOverlay} />}
    {settingsOpen && settingsDraft && <SettingsDialog config={config} draft={settingsDraft} onDraft={setSettingsDraft} onClose={() => setSettingsOpen(false)} onApply={applySettings} />}
    {notice && <div className="wb-toast"><Check size={16} />{notice}</div>}
  </div>
}

function Welcome({ profile, onStarter }: { profile?: Profile; onStarter: (value: string) => void }) {
  const starters = ['Explain this codebase in three bullets', 'Help me debug a local build', 'Write a compact Python utility']
  return <div className="welcome">
    <div className="welcome-icon"><WandSparkles size={27} /></div>
    <h1>What should we build?</h1>
    <p>{profile?.name || 'LongCat'} is ready through the native XTLLM Vulkan runtime. Your conversation and context policy stay on this machine.</p>
    <div className="starter-grid">{starters.map((value) => <button key={value} onClick={() => onStarter(value)}><Sparkles size={14} />{value}</button>)}</div>
  </div>
}

function ChatBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') return <article className="chat-message user-message"><div>{message.content}</div></article>
  const parts = message.content.split('```')
  return <article className="chat-message assistant-message">
    <div className="assistant-avatar"><Sparkles size={15} /></div>
    <div className="assistant-content">{parts.map((part, index) => index % 2 ? <pre key={index}><code>{part.replace(/^\w+\n/, '')}</code></pre> : <p key={index}>{part}</p>)}
      {message.metrics && <div className="message-metrics"><span>{message.metrics.profile_name}</span>{message.metrics.throughput_tps !== undefined && <span>{message.metrics.throughput_tps.toFixed(2)} tok/s</span>}{message.metrics.elapsed_ms !== undefined && <span>{(message.metrics.elapsed_ms / 1000).toFixed(1)}s</span>}<span>{message.token_count} tokens</span></div>}
    </div>
  </article>
}

function ActivityTrail({ statuses, running }: { statuses: RunStatus[]; running: boolean }) {
  const visible = statuses.slice(-4)
  return <div className="activity-trail">
    <div className="activity-heading"><span className={running ? 'pulse' : 'complete'}>{running ? <Activity size={14} /> : <Check size={14} />}</span><strong>{running ? 'Working locally' : visible.at(-1)?.message}</strong></div>
    <div className="activity-steps">{visible.map((status, index) => <div key={`${status.timestamp}-${index}`} className={index === visible.length - 1 && running ? 'current' : ''}><i /> <span>{status.message}</span>{status.detail && <small>{status.detail}</small>}</div>)}</div>
  </div>
}

function ContextInspector({ chat, overlay, onPatch, onClean, onUndo, onLaunch }: {
  chat: Chat; overlay: OverlayState | null; onPatch: (changes: Partial<Chat>, toast?: string) => Promise<void>
  onClean: () => Promise<void>; onUndo: () => Promise<void>; onLaunch: () => Promise<void>
}) {
  const pixels = useMemo(() => buildPixels(overlay), [overlay])
  const metrics = overlay?.metrics
  return <aside className="context-inspector">
    <div className="inspector-heading"><div><span className="eyebrow">LIVE CONTEXT</span><h2>Window overlay</h2></div><span className="live-dot"><i />LIVE</span></div>
    <section className="mini-window">
      <div className="mini-window-bar"><span /><span /><span /><strong>{metrics ? `${metrics.context_percent.toFixed(1)}%` : 'WAITING'}</strong></div>
      <div className="pixel-grid">{pixels.map((color, index) => <i key={index} style={{ background: color }} />)}</div>
      <div className="context-readout"><strong>{formatTokens(metrics?.active_tokens || 0)}</strong><span>/ {chat.context_limit} tokens</span></div>
    </section>
    <div className="context-legend">{Object.entries(categoryColors).filter(([name]) => name !== 'Free').map(([name, color]) => <span key={name}><i style={{ background: color }} />{name}</span>)}</div>
    <section className="cleanup-card">
      <div className="cleanup-title"><div><Zap size={16} /><span><strong>Real-time cleanup</strong><small>Runs before each model request</small></span></div><button className={`switch ${chat.auto_mode ? 'on' : ''}`} aria-label="Toggle automatic cleanup" onClick={() => void onPatch({ auto_mode: !chat.auto_mode }, `Automatic cleanup ${chat.auto_mode ? 'off' : 'on'}`)}><i /></button></div>
      <div className="cleanup-modes">{cleanupModes.map((mode) => <button key={mode.id} className={chat.cleanup_mode === mode.id ? 'active' : ''} onClick={() => void onPatch({ cleanup_mode: mode.id })}><strong>{mode.label}</strong><span>{mode.description}</span></button>)}</div>
      <button className="clean-now" disabled={!chat.lcm_session_id} onClick={() => void onClean()}><WandSparkles size={16} />Clean context now</button>
      <button className="undo-context" disabled={!chat.lcm_session_id} onClick={() => void onUndo()}><RotateCcw size={14} />Undo last cleanup</button>
    </section>
    <div className="inspector-stats"><div><span>Free</span><strong>{Math.max(0, chat.context_limit - (metrics?.active_tokens || 0))}</strong></div><div><span>Cleaned</span><strong>{metrics?.removed_tokens || 0}</strong></div><div><span>Mode</span><strong>{chat.auto_mode ? chat.cleanup_mode : 'manual'}</strong></div></div>
    <div className="inspector-actions"><button onClick={() => void onLaunch()}><MonitorUp size={15} />Floating overlay</button><a href="/context"><LayoutDashboard size={15} />Full inspector</a></div>
  </aside>
}

function buildPixels(overlay: OverlayState | null) {
  const count = 256
  const colors = Array<string>(count).fill(categoryColors.Free)
  if (!overlay?.metrics || !overlay.segments || !overlay.context_limit) return colors
  let cursor = 0
  for (const segment of overlay.segments) {
    const pixels = Math.max(1, Math.round(segment.tokens / overlay.context_limit * count))
    for (let index = 0; index < pixels && cursor < count; index += 1) colors[cursor++] = categoryColors[segment.category] || categoryColors.Conversation
  }
  return colors
}

function gibForTokens(config: WorkbenchConfig, tokens: number) {
  return tokens * config.limits.context_bytes_per_token / 1024 ** 3
}

function tokensForGib(config: WorkbenchConfig, gib: number) {
  return Math.floor(gib * 1024 ** 3 / config.limits.context_bytes_per_token)
}

function roundRamUp(value: number) {
  return Math.ceil(value * 10) / 10
}

function runtimeContextCapacity(config: WorkbenchConfig, profile: Profile, vramGib: number) {
  const deviceGibPerToken = config.limits.device_context_bytes_per_token / 1024 ** 3
  const slotGib = config.limits.device_expert_slot_bytes / 1024 ** 3
  const fixedVram = profile.measured_vram_gib - 256 * deviceGibPerToken - profile.max_device_slots * slotGib
  const contextGib = vramGib - config.limits.vram_safety_gib - fixedVram - config.limits.minimum_device_slots * slotGib
  return Math.max(64, Math.min(profile.context_limit, Math.floor(contextGib / deviceGibPerToken)))
}

function SettingsDialog({ config, draft, onDraft, onClose, onApply }: {
  config: WorkbenchConfig; draft: Chat; onDraft: (chat: Chat) => void; onClose: () => void; onApply: () => Promise<void>
}) {
  const profile = config.profiles.find((entry) => entry.id === draft.profile) || config.profiles[0]
  const update = (changes: Partial<Chat>) => onDraft({ ...draft, ...changes })
  const totalRam = config.limits.total_ram_gib
  const outputRam = gibForTokens(config, draft.output_token_limit)
  const contextBodyRam = Math.max(0, draft.context_ram_budget_gib - outputRam)
  const allocatedRam = draft.ram_budget_gib + draft.context_ram_budget_gib
  const freeRam = Math.max(0, totalRam - allocatedRam)
  const maxContextRam = Math.max(0.1, totalRam - draft.ram_budget_gib)
  const maxInferenceRam = Math.max(4, totalRam - draft.context_ram_budget_gib)
  const contextTokenMax = Math.max(64, Math.min(profile.context_limit, tokensForGib(config, maxContextRam)))
  const activeRuntimeContext = runtimeContextCapacity(config, profile, draft.vram_budget_gib)
  const updateInferenceRam = (value: number) => update({ ram_budget_gib: Math.min(value, maxInferenceRam) })
  const updateContextRam = (value: number) => {
    const nextRam = Math.min(value, maxContextRam)
    const nextTokenMax = Math.max(64, Math.min(profile.context_limit, tokensForGib(config, nextRam)))
    update({
      context_ram_budget_gib: nextRam,
      context_limit: Math.min(draft.context_limit, nextTokenMax),
      output_token_limit: Math.min(draft.output_token_limit, Math.max(1, nextTokenMax - 16)),
    })
  }
  const updateContextTokens = (value: number) => {
    const nextTokens = Math.min(value, contextTokenMax)
    const requiredRam = roundRamUp(gibForTokens(config, nextTokens))
    update({
      context_limit: nextTokens,
      context_ram_budget_gib: Math.max(draft.context_ram_budget_gib, requiredRam),
      output_token_limit: Math.min(draft.output_token_limit, Math.max(1, nextTokens - 16)),
    })
  }
  return <div className="settings-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><section className="settings-dialog">
    <header><div><span className="eyebrow">XTLLM RUNTIME</span><h2>Model & resource limits</h2><p>Mix inference/cache RAM and context RAM inside one hard 70 GiB host-memory envelope. VRAM has its own 11.5 GiB ceiling.</p></div><button className="icon-button" onClick={onClose}><X size={18} /></button></header>
    <div className="profile-options">{config.profiles.map((entry) => <button key={entry.id} className={draft.profile === entry.id ? 'active' : ''} onClick={() => update({ profile: entry.id, vram_budget_gib: entry.recommended_vram_gib })}><span className="profile-icon">{entry.id === 'exact' ? <Gauge size={19} /> : <Zap size={19} />}</span><span><strong>{entry.name}</strong><small>{entry.description}</small><em>{entry.precision}</em></span>{draft.profile === entry.id && <Check size={17} />}</button>)}</div>
    <section className="ram-envelope">
      <div className="ram-envelope-heading"><span><HardDrive size={16} /><strong>Shared RAM envelope</strong></span><output>{allocatedRam.toFixed(1)} <small>/ {totalRam.toFixed(0)} GiB</small></output></div>
      <div className="ram-meter" aria-label={`${allocatedRam.toFixed(1)} of ${totalRam.toFixed(0)} GiB allocated`}>
        <i className="ram-inference" style={{ width: `${draft.ram_budget_gib / totalRam * 100}%` }} />
        <i className="ram-context" style={{ width: `${contextBodyRam / totalRam * 100}%` }} />
        <i className="ram-output" style={{ width: `${outputRam / totalRam * 100}%` }} />
      </div>
      <div className="ram-legend"><span><i className="ram-inference" />Inference + cache <strong>{draft.ram_budget_gib.toFixed(1)}</strong></span><span><i className="ram-context" />Context <strong>{contextBodyRam.toFixed(2)}</strong></span><span><i className="ram-output" />Output reserve <strong>{outputRam.toFixed(3)}</strong></span><span><i className="ram-free" />Free <strong>{freeRam.toFixed(1)}</strong></span></div>
    </section>
    <div className="resource-grid">
      <ResourceControl icon={<HardDrive size={17} />} label="Inference + cache RAM" value={draft.ram_budget_gib} unit="GiB" min={4} max={maxInferenceRam} step={0.5} note={`${profile.measured_ram_gib.toFixed(2)} GiB measured · 70 GiB shared maximum`} onChange={updateInferenceRam} />
      <ResourceControl icon={<Box size={17} />} label="Context-window RAM" value={draft.context_ram_budget_gib} unit="GiB" min={0.1} max={maxContextRam} step={0.1} note={`Host planning reserve · current LongCat KV is device-local`} onChange={updateContextRam} />
      <ResourceControl icon={<Cpu size={17} />} label="VRAM budget" value={draft.vram_budget_gib} unit="GiB" min={4} max={config.limits.max_vram_gib} step={0.1} note={`${profile.measured_vram_gib.toFixed(2)} GiB measured · ${profile.max_device_slots} slots`} onChange={(value) => update({ vram_budget_gib: value })} />
      <ContextTokenControl value={draft.context_limit} min={64} max={contextTokenMax} modelMax={profile.context_limit} runtimeMax={activeRuntimeContext} onChange={updateContextTokens} />
      <ResourceControl icon={<Send size={17} />} label="Output limit" value={draft.output_token_limit} unit="tokens" min={1} max={Math.min(profile.max_output_tokens, draft.context_limit - 16)} step={1} note="Unused capacity remains available to the prompt" onChange={(value) => update({ output_token_limit: value })} />
    </div>
    {draft.context_limit > activeRuntimeContext && <div className="runtime-caveat"><CircleAlert size={16} /><span><strong>Model plan vs current VRAM capacity</strong><small>XTLLM no longer has a compiled 256-token cap. At {draft.vram_budget_gib.toFixed(1)} GiB VRAM, this profile safely allocates about {activeRuntimeContext.toLocaleString()} tokens with its current device-local KV cache. Requests are fitted to that capacity.</small></span></div>}
    <div className="hardware-strip"><Cpu size={17} /><span><strong>{config.host.gpu}</strong><small>{config.host.runtime} · {config.host.ram_gib?.toFixed(1)} GiB system RAM</small></span><i className={config.available ? 'online' : ''}>{config.available ? 'READY' : 'MISSING'}</i></div>
    <footer><span><ShieldCheck size={14} />Runtime files and chat history stay on A:</span><div><button className="secondary" onClick={onClose}>Cancel</button><button className="primary" onClick={() => void onApply()}>Apply limits</button></div></footer>
  </section></div>
}

function ResourceControl({ icon, label, value, unit, min, max, step, note, onChange }: { icon: React.ReactNode; label: string; value: number; unit: string; min: number; max: number; step: number; note: string; onChange: (value: number) => void }) {
  return <label className="resource-control"><div><span>{icon}<strong>{label}</strong></span><output>{Number.isInteger(value) ? value : value.toFixed(1)} <small>{unit}</small></output></div><input type="range" min={min} max={max} step={step} value={Math.min(value, max)} onChange={(event) => onChange(Number(event.target.value))} /><p>{note}</p></label>
}

function ContextTokenControl({ value, min, max, modelMax, runtimeMax, onChange }: { value: number; min: number; max: number; modelMax: number; runtimeMax: number; onChange: (value: number) => void }) {
  const safeMax = Math.max(min, max)
  const rangeMax = Math.log2(safeMax / min)
  const position = Math.min(rangeMax, Math.max(0, Math.log2(Math.max(min, value) / min)))
  return <label className="resource-control"><div><span><Box size={17} /><strong>Context window</strong></span><output>{formatTokens(value)} <small>tokens</small></output></div><input aria-label="Context window tokens" aria-valuetext={`${value.toLocaleString()} tokens`} type="range" min={0} max={rangeMax || 1} step={0.01} value={position} onChange={(event) => {
    const raw = min * 2 ** Number(event.target.value)
    const rounded = raw >= 16_384 ? Math.round(raw / 1024) * 1024 : Math.round(raw / 64) * 64
    onChange(Math.min(safeMax, Math.max(min, rounded)))
  }} /><p>Model maximum: {modelMax.toLocaleString()} · current VRAM capacity: ~{runtimeMax.toLocaleString()}</p></label>
}
