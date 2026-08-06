import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { ATTRIBUTION, ATTRIBUTION_URL, MODEL_LICENSE } from './lib/birdnet/constants'

/**
 * P0 ships the pipeline, not the interface. This placeholder exists so the static
 * build has an entry point and so the model attribution is already where it has
 * to be — visible on the page, not buried in an "about" dialog, as CC BY-NC-SA
 * requires. The real interface lands in P1.
 */
function Placeholder() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-2xl flex-col justify-center gap-6 px-6 py-16">
      <div>
        <h1 className="text-2xl font-medium tracking-tight">whosyourbirdy</h1>
        <p className="mt-2 text-neutral-500">
          Identification d'oiseaux au chant, entièrement dans le navigateur. Aucun fichier n'est
          envoyé nulle part.
        </p>
      </div>

      <p className="text-sm text-neutral-500">
        Phase P0 : le pipeline d'analyse et la vérification de parité avec BirdNET sont en place.
        L'interface arrive en P1.
      </p>

      <footer className="mt-4 border-t border-neutral-200 pt-4 text-xs text-neutral-500 dark:border-neutral-800">
        <a href={ATTRIBUTION_URL} className="underline underline-offset-2" rel="noreferrer">
          {ATTRIBUTION}
        </a>
        <span className="mt-1 block">
          Modèle sous {MODEL_LICENSE} — usage non commercial. Ce projet est non commercial.
        </span>
      </footer>
    </main>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Placeholder />
  </StrictMode>,
)
