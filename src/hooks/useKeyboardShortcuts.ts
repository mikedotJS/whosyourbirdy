import { useEffect } from 'react'

export interface Shortcuts {
  /** Space. Starts or stops the segment under the playhead. */
  onTogglePlay: () => void
  /** Arrows. Seconds, already signed; Shift multiplies the step upstream. */
  onSeek: (delta: number) => void
  /** `/`. Moves focus into the species list. */
  onFocusList: () => void
  /** Escape, when no dialog is open — `<dialog>` handles its own. */
  onEscape: () => void
}

/** One analysis window. The natural unit to move by in this app. */
const STEP = 3
const BIG_STEP = 10

/**
 * Global keyboard shortcuts.
 *
 * Three rules keep this from fighting the rest of the interface:
 *
 * 1. **Never steal from a text field.** Space in an input is a space, `/` is a
 *    slash, and arrows move the caret.
 * 2. **Never double-act.** `Spectrogram` already handles arrows when it has
 *    focus and calls `preventDefault`, so a defaulted-prevented event has been
 *    dealt with by something closer to the user. React attaches its listeners on
 *    the root container, which is inside `window`, so by the time this bubbling
 *    listener runs the flag is already set.
 * 3. **Stand down under a modal.** A `<dialog>` is a mode; scrubbing the
 *    timeline from behind an open sheet would act on something the user cannot
 *    see. Escape is left alone there too — the platform closes the dialog.
 */
export function useKeyboardShortcuts({
  onTogglePlay,
  onSeek,
  onFocusList,
  onEscape,
}: Shortcuts): void {
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return
      if (event.metaKey || event.ctrlKey || event.altKey) return

      const target = event.target as HTMLElement | null
      if (target?.isContentEditable) return
      const tag = target?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      const modal = document.querySelector('dialog[open]') !== null
      // Space is the native activation key for a button or a link. Claiming it
      // globally would have made every focused species row and every occurrence
      // chip unusable from the keyboard — the shortcut would have eaten the
      // press and played something else instead of activating what has focus.
      const activatable = tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY'

      switch (event.key) {
        case ' ':
        case 'Spacebar': // older WebKit
          if (modal || activatable) return
          event.preventDefault() // otherwise the page scrolls under the shortcut
          onTogglePlay()
          return
        case 'ArrowLeft':
          if (modal) return
          event.preventDefault()
          onSeek(event.shiftKey ? -BIG_STEP : -STEP)
          return
        case 'ArrowRight':
          if (modal) return
          event.preventDefault()
          onSeek(event.shiftKey ? BIG_STEP : STEP)
          return
        case '/':
          if (modal) return
          event.preventDefault() // Firefox opens quick-find on `/`
          onFocusList()
          return
        case 'Escape':
          if (modal) return
          onEscape()
          return
        default:
      }
    }

    window.addEventListener('keydown', handle)
    return () => window.removeEventListener('keydown', handle)
  }, [onTogglePlay, onSeek, onFocusList, onEscape])
}
