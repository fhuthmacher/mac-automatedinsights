import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Add favicon link in the head
const favicon = document.createElement('link')
favicon.rel = 'icon'
favicon.href = '/icon.ico'
document.head.appendChild(favicon)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
