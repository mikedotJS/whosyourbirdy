import { useCallback, useRef, useState } from 'react'

interface Props {
  onFile: (file: File) => void
}

export function DropZone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
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
    <div
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
        'flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors',
        over
          ? 'border-focus bg-raised'
          : 'border-line',
      ].join(' ')}
    >
      <p className="text-base">Déposez un enregistrement ici</p>
      <p className="text-sm text-ink-3">
        wav, mp3, flac, m4a, ogg — idéalement en 48 kHz
      </p>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-2 rounded-md bg-ink px-4 py-2 text-sm font-medium text-surface transition-colors hover:opacity-90"
      >
        Choisir un fichier
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          // Allow re-selecting the same file after a reset.
          e.target.value = ''
        }}
      />
    </div>
  )
}
