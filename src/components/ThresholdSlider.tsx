interface Props {
  value: number
  min: number
  onChange: (value: number) => void
  total: number
  visible: number
}

/**
 * Confidence threshold, defaulting to BirdNET's own 0.25.
 *
 * Filters results that are already in memory rather than re-running the model,
 * so dragging it is instant. Its lower bound is the floor the analysis actually
 * used — going below it would promise results that were never computed.
 */
export function ThresholdSlider({ value, min, onChange, total, visible }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-4">
        <label htmlFor="threshold" className="text-sm font-medium">
          Seuil de confiance
        </label>
        <span className="text-sm tabular-nums text-neutral-500">
          {value.toFixed(2)} · {visible}/{total}
        </span>
      </div>
      <input
        id="threshold"
        type="range"
        min={min}
        max={0.99}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-neutral-900 dark:accent-neutral-100"
      />
      <p className="text-xs text-neutral-500">
        BirdNET utilise 0,25 par défaut. Plus bas = plus de détections, dont davantage de faux
        positifs.
      </p>
    </div>
  )
}
