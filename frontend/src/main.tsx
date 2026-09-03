import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Overlay from './Overlay.tsx'
import Workbench from './Workbench.tsx'

const RootView = window.location.pathname === '/overlay'
  ? Overlay
  : window.location.pathname === '/context'
    ? App
    : Workbench

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RootView />
  </StrictMode>,
)
