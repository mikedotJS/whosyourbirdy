import { Sheet } from './Sheet'
import {
  WEEKS_PER_YEAR,
  WEEK_WHOLE_YEAR,
  describeWeek,
  isValidLatitude,
  isValidLongitude,
} from '../lib/birdnet/geo'
import type { GeoSettings, GeoState } from '../hooks/useGeoFilter'

interface Props {
  open: boolean
  onClose: () => void
  geo: GeoState & {
    update: (patch: Partial<GeoSettings>) => void
    locate: () => Promise<void>
  }
  /** How many species the filter is currently removing from the results. */
  masked: number
}

/**
 * Where and when the recording was made.
 *
 * BirdNET's own species filter, with its own semantics kept intact: the week
 * runs 1–48 (four per month, not ISO weeks), and a species under the threshold
 * is *removed* rather than down-weighted. Both are upstream's behaviour, and
 * both are the kind of thing that quietly produces a different report if you
 * assume the obvious instead of reading the source.
 */
export function GeoSheet({ open, onClose, geo, masked }: Props) {
  const { settings, update, locate, loading, progress, error } = geo
  const coordsValid =
    isValidLatitude(settings.latitude) && isValidLongitude(settings.longitude)

  return (
    <Sheet
      open={open}
      onClose={onClose}
      title="Lieu et saison"
      aside={
        <label className="flex shrink-0 cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={(e) => update({ enabled: e.target.checked })}
            className="h-5 w-5 accent-[var(--color-focus)]"
          />
          Activer
        </label>
      }
    >
      <p className="mb-4 text-sm text-ink-3">
        Le filtre géographique de BirdNET retire les espèces qui n'ont pas lieu d'être à cet
        endroit à cette période. Il est facultatif : sans lui, l'analyse reste celle de BirdNET
        sans position.
      </p>

      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-3">
          <Field
            label="Latitude"
            value={settings.latitude}
            onChange={(v) => update({ latitude: v })}
            invalid={!isValidLatitude(settings.latitude)}
            hint="−90 à 90"
          />
          <Field
            label="Longitude"
            value={settings.longitude}
            onChange={(v) => update({ longitude: v })}
            invalid={!isValidLongitude(settings.longitude)}
            hint="−180 à 180"
          />
        </div>

        <button
          type="button"
          onClick={() => void locate()}
          className="flex h-11 items-center justify-center gap-2 rounded-xl border border-line text-sm transition-colors duration-150 hover:border-line-strong hover:bg-hover"
        >
          <PinIcon />
          Utiliser ma position
        </button>
        {/* The Geolocation prompt only ever appears after that button. Nothing
            asks for a position on load. */}
        <p className="-mt-2 text-xs text-ink-3">
          La demande de position n'est faite qu'à ce clic, jamais au chargement.
        </p>

        <div>
          <div className="flex items-baseline justify-between gap-3">
            <label htmlFor="geo-week" className="text-sm font-medium">
              Période
            </label>
            <span className="text-sm tabular-nums text-ink-3">
              {describeWeek(settings.week)}
            </span>
          </div>
          <input
            id="geo-week"
            type="range"
            min={1}
            max={WEEKS_PER_YEAR}
            step={1}
            value={settings.week === WEEK_WHOLE_YEAR ? 1 : settings.week}
            onChange={(e) => update({ week: Number(e.target.value) })}
            style={
              {
                '--fill': `${(((settings.week === WEEK_WHOLE_YEAR ? 1 : settings.week) - 1) / (WEEKS_PER_YEAR - 1)) * 100}%`,
              } as React.CSSProperties
            }
            className="range-aube h-11 w-full"
          />
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={settings.week === WEEK_WHOLE_YEAR}
              onChange={(e) => update({ week: e.target.checked ? WEEK_WHOLE_YEAR : 20 })}
              className="h-5 w-5 accent-[var(--color-focus)]"
            />
            Toute l'année
          </label>
          {/* Not ISO weeks — upstream splits each month into four. Saying so
              here is cheaper than letting someone convert from a calendar. */}
          <p className="mt-1 text-xs text-ink-3">
            BirdNET découpe l'année en 48 semaines, soit quatre par mois — ce ne sont pas les
            semaines ISO.
          </p>
        </div>

        <div>
          <div className="flex items-baseline justify-between gap-3">
            <label htmlFor="geo-threshold" className="text-sm font-medium">
              Seuil du filtre
            </label>
            <span className="text-sm tabular-nums text-ink-3">
              {settings.threshold.toFixed(2)}
            </span>
          </div>
          <input
            id="geo-threshold"
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={settings.threshold}
            onChange={(e) => update({ threshold: Number(e.target.value) })}
            style={
              { '--fill': `${((settings.threshold - 0.01) / 0.49) * 100}%` } as React.CSSProperties
            }
            className="range-aube h-11 w-full"
          />
          <p className="mt-1 text-xs text-ink-3">
            0,03 est la valeur de BirdNET-Analyzer. Une espèce sous ce seuil est{' '}
            <strong className="font-medium text-ink-2">retirée</strong> des résultats, pas atténuée.
          </p>
        </div>

        {loading && (
          <p className="text-sm text-ink-2" role="status" aria-live="polite">
            {progress && progress.total > 0 && !progress.fromCache
              ? `Téléchargement du modèle géographique — ${(progress.loaded / 1e6).toFixed(0)} / ${(progress.total / 1e6).toFixed(0)} Mo`
              : 'Calcul du filtre…'}
          </p>
        )}

        {error && (
          <p role="alert" className="rounded-lg border border-play/40 bg-play/10 p-3 text-sm">
            {error}
          </p>
        )}

        {!coordsValid && (
          <p className="text-sm text-ink-3">Coordonnées hors plage — le filtre est en attente.</p>
        )}

        {settings.enabled && geo.mask && (
          <p className="text-sm text-ink-2">
            {masked === 0
              ? 'Aucune espèce détectée n’est écartée par ce réglage.'
              : `${masked} espèce${masked > 1 ? 's' : ''} détectée${masked > 1 ? 's' : ''} ${masked > 1 ? 'sont écartées' : 'est écartée'} par ce réglage.`}
          </p>
        )}

        <p className="text-xs text-ink-3">
          Le modèle géographique pèse 29 Mo. Il n'est téléchargé qu'à l'activation du filtre, puis
          mis en cache.
        </p>
      </div>
    </Sheet>
  )
}

function Field({
  label,
  value,
  onChange,
  invalid,
  hint,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  invalid: boolean
  hint: string
}) {
  const id = `geo-${label.toLowerCase()}`
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        step="0.0001"
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-invalid={invalid}
        aria-describedby={`${id}-hint`}
        className={[
          'mt-1 h-11 w-full rounded-xl border bg-hover px-3 text-sm tabular-nums',
          'outline-none transition-colors duration-150 focus:border-focus',
          invalid ? 'border-play' : 'border-line',
        ].join(' ')}
      />
      <p id={`${id}-hint`} className="mt-1 text-xs text-ink-3">
        {hint}
      </p>
    </div>
  )
}

function PinIcon() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 24 24"
      className="h-4 w-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 21s7-6.4 7-11a7 7 0 1 0-14 0c0 4.6 7 11 7 11Z" />
      <circle cx="12" cy="10" r="2.5" />
    </svg>
  )
}
