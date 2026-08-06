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
  /**
   * Identifies the current playback attempt.
   *
   * `play()` is async, so two quick clicks can both be mid-flight; without a
   * token the first one's continuation installs a timer that later pauses the
   * *second* segment. The token also lets a stale timer recognise itself.
   */
  const playTokenRef = useRef(0)
  const [playing, setPlaying] = useState<number | null>(null)

  useEffect(() => {
    if (!file) {
      audioRef.current = null
      return
    }
    const url = URL.createObjectURL(file)
    const audio = new Audio(url)
    audioRef.current = audio

    // A window can be shorter than 3 s — the zero-padded final one always is —
    // so the fixed timer alone would keep claiming to play after the sound has
    // stopped, and the next click would be swallowed as a "pause".
    const onEnded = () => {
      playTokenRef.current++
      setPlaying(null)
    }
    audio.addEventListener('ended', onEnded)

    return () => {
      audio.removeEventListener('ended', onEnded)
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
    playTokenRef.current++
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
        playTokenRef.current++
        audio.pause()
        setPlaying(null)
        return
      }

      const token = ++playTokenRef.current
      audio.currentTime = start
      void audio.play().then(
        () => {
          if (playTokenRef.current !== token) return // superseded while starting
          setPlaying(key)
          // Stop at the end of the window rather than running on into the next
          // one, so what you hear is exactly what the model scored.
          stopTimerRef.current = window.setTimeout(() => {
            if (playTokenRef.current !== token) return
            audio.pause()
            setPlaying(null)
          }, WINDOW_SECONDS * 1000)
        },
        () => {
          if (playTokenRef.current === token) setPlaying(null)
        },
      )
    },
    [playing],
  )

  useEffect(() => clearTimer, [])

  return { play, stop, playing }
}
