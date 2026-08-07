import { useCallback, useState } from 'react'

interface Props {
  onFile: (file: File) => void
  /** Opens the picker the action bar also opens — one input for the whole app. */
  onBrowse: () => void
}

export function DropZone({ onFile, onBrowse }: Props) {
  const [over, setOver] = useState(false)

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      setOver(false)
      const file = event.dataTransfer.files[0]
      if (file) onFile(file)
    },
    [onFile],
  )

  return (
    <button
      type="button"
      onClick={onBrowse}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={(e) => {
        // Fires when the pointer crosses onto a child too, which made the
        // highlight flicker; only clear when the pointer truly leaves.
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false)
      }}
      onDrop={handleDrop}
      className={[
        // The whole zone is the target, not a small button inside it. On a phone
        // there is nothing to drag, so tapping anywhere has to open the picker.
        // `grow` so it fills the idle screen: it is the only thing to do here,
        // and a small dashed box floating above a lot of nothing reads as a
        // form field rather than as the app's one action.
        'flex w-full grow flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed',
        'px-6 py-14 text-center transition-colors duration-150',
        over ? 'border-focus bg-raised' : 'border-line hover:border-line-strong hover:bg-hover',
      ].join(' ')}
    >
      <UploadIcon />
      <span className="mt-1 text-base font-medium">Choisir un enregistrement</span>
      <span className="text-sm text-ink-3">ou le déposer ici</span>
      <span className="text-xs text-ink-3">wav, mp3, flac, m4a, ogg — idéalement en 48 kHz</span>
    </button>
  )
}

function UploadIcon() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className="h-7 w-7 text-ink-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 16V4m0 0L8 8m4-4 4 4" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  )
}
