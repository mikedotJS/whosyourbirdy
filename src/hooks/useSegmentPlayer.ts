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
  /**
   * Live playback position, for the spectrogram playhead.
   *
   * Driven by requestAnimationFrame rather than `timeupdate`: that event fires
   * about four times a second, which reads as a stuttering playhead against a
   * smoothly scrolling picture.
   */
  const [position, setPosition] = useState<number | null>(null)
  const frameRef = useRef<number | null>(null)

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
      stopTracking()
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

  const stopTracking = () => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = null
    }
  }

  const track = () => {
    const audio = audioRef.current
    if (!audio) return
    setPosition(audio.currentTime)
    frameRef.current = requestAnimationFrame(track)
  }

  const stop = useCallback(() => {
    playTokenRef.current++
    clearTimer()
    stopTracking()
    audioRef.current?.pause()
    setPlaying(null)
    setPosition(null)
  }, [])

  /**
   * Move the playhead without starting playback.
   *
   * Scrubbing the spectrogram is a navigation gesture, not a play gesture — it
   * should not start sound the user did not ask for. It does move the audio
   * element so the next play starts from where they looked.
   */
  const seek = useCallback((seconds: number) => {
    const audio = audioRef.current
    if (audio) audio.currentTime = seconds
    setPosition(seconds)
  }, [])

  const play = useCallback(
    (start: number, key: number) => {
      const audio = audioRef.current
      if (!audio) return

      clearTimer()
      if (playing === key) {
        playTokenRef.current++
        stopTracking()
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
          stopTracking()
          frameRef.current = requestAnimationFrame(track)
          // Stop at the end of the window rather than running on into the next
          // one, so what you hear is exactly what the model scored.
          stopTimerRef.current = window.setTimeout(() => {
            if (playTokenRef.current !== token) return
            audio.pause()
            stopTracking()
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

  useEffect(() => () => {
    clearTimer()
    stopTracking()
  }, [])

  return { play, stop, seek, playing, position }
}
