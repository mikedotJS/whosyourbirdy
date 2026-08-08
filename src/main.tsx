import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { dismissSplash } from './splash'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// After `render`, so the splash is only taken away once there is an app behind
// it. The inline splash in index.html covers everything before this point.
dismissSplash()
