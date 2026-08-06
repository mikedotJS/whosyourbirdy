import type { AnalysisState } from '../hooks/useBirdNet'

interface Props {
  state: AnalysisState
}

/**
 * Reports what is actually slow.
 *
 * The first run is dominated by a ~52 MB model download, which deserves a real
 * byte counter rather than an indeterminate spinner; later runs skip it entirely
 * via the Cache API and go straight to the per-window progress.
 */
export function ProgressPanel({ state }: Props) {
  const { phase, modelProgress, progress } = state

  if (phase === 'loading-model' && modelProgress) {
    const { loaded, total, fromCache } = modelProgress
    const pct = total > 0 ? (loaded / total) * 100 : 0
    return (
      <Panel
        label={
          fromCache
            ? 'Modèle chargé depuis le cache'
            : modelProgress.phase === 'compiling'
              ? 'Préparation du modèle…'
              : 'Téléchargement du modèle BirdNET'
        }
        detail={
          total > 0 && !fromCache
            ? `${formatMB(loaded)} / ${formatMB(total)} Mo — une seule fois, ensuite mis en cache`
            : undefined
        }
        percent={total > 0 ? pct : null}
      />
    )
  }

  if (phase === 'decoding') {
    return <Panel label="Décodage de l'audio…" percent={null} />
  }

  if (phase === 'analyzing' && progress) {
    const pct = (progress.completed / progress.total) * 100
    return (
      <Panel
        label="Analyse en cours"
        detail={`fenêtre ${progress.completed} / ${progress.total} — ${formatTime(progress.seconds)}`}
        percent={pct}
      />
    )
  }

  return <Panel label="Préparation…" percent={null} />
}

function Panel({
  label,
  detail,
  percent,
}: {
  label: string
  detail?: string
  percent: number | null
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-4 text-sm">
        <span>{label}</span>
        {percent !== null && (
          <span className="tabular-nums text-neutral-500">{percent.toFixed(0)} %</span>
        )}
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
        <div
          className={[
            'h-full bg-neutral-900 dark:bg-neutral-100',
            percent === null ? 'w-1/3 animate-pulse' : 'transition-[width] duration-150 ease-out',
          ].join(' ')}
          style={percent === null ? undefined : { width: `${percent}%` }}
        />
      </div>
      {detail && <p className="text-xs text-neutral-500">{detail}</p>}
    </div>
  )
}

function formatMB(bytes: number): string {
  return (bytes / 1e6).toFixed(1)
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
