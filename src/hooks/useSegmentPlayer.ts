import { useCallback, useEffect, useRef, useState } from 'react'
import { WINDOW_SECONDS } from '../lib/birdnet/constants'

/**
 * Play the 3-second segment a detection came from.
 *
 * Plays the *original* file through an `<audio>` element rather than keeping
 * decoded PCM around: the decoded 48 kHz mono buffer for a long recording is
 * tens of megabytes, and the pipeline transfers its copy into the worker anyway.
 * An object URL costs nothing and lets the browser handle seeking.
 */
export function useSegmentPlayer(file: File | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const stopTimerRef = useRef<number | null>(null)
  const [playing, setPlaying] = useState<number | null>(null)

  useEffect(() => {
    if (!file) {
      audioRef.current = null
      return
    }
    const url = URL.createObjectURL(file)
    const audio = new Audio(url)
    audioRef.current = audio
    return () => {
      audio.pause()
      URL.revokeObjectURL(url)
      audioRef.current = null
    }
  }, [file])

  const clearTimer = () => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
  }

  const stop = useCallback(() => {
    clearTimer()
    audioRef.current?.pause()
    setPlaying(null)
  }, [])

  const play = useCallback(
    (start: number, key: number) => {
      const audio = audioRef.current
      if (!audio) return

      clearTimer()
      if (playing === key) {
        audio.pause()
        setPlaying(null)
        return
      }

      audio.currentTime = start
      void audio.play().then(
        () => {
          setPlaying(key)
          // Stop at the end of the window rather than running on into the next
          // one, so what you hear is exactly what the model scored.
          stopTimerRef.current = window.setTimeout(() => {
            audio.pause()
            setPlaying(null)
          }, WINDOW_SECONDS * 1000)
        },
        () => setPlaying(null),
      )
    },
    [playing],
  )

  useEffect(() => clearTimer, [])

  return { play, stop, playing }
}
