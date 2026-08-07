import { useCallback, useEffect, useRef } from 'react'

interface Props {
  open: boolean
  onClose: () => void
  title: string
  /** Right-hand side of the header row — a score, a count, an action. */
  aside?: React.ReactNode
  children: React.ReactNode
}

/** Fraction of the sheet's height a drag must cover to count as a dismissal. */
const DISMISS_FRACTION = 0.25
/** …or this speed, so a short flick also closes it. Pixels per millisecond. */
const DISMISS_VELOCITY = 0.5

/**
 * Bottom sheet.
 *
 * A real `<dialog>` opened with `showModal()`, not a positioned div. That single
 * choice hands over the focus trap, the inert background, Escape-to-close, and
 * top-layer stacking — four things that are easy to write badly and that no
 * amount of care in application code does as well as the platform. What is left
 * here is the geometry (in `index.css`) and the drag-to-dismiss gesture, which
 * the platform does not provide.
 *
 * Focus returns to whatever opened it, because `showModal()`/`close()` restore
 * it on their own.
 */
export function Sheet({ open, onClose, title, aside, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null)
  const dragRef = useRef<{ id: number; y0: number; t0: number; dy: number } | null>(null)

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    // `open` as a *prop* is not the same as the `open` attribute: setting the
    // attribute directly opens a non-modal dialog with no backdrop and no focus
    // trap, which is the failure mode this component exists to avoid.
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  // Escape and the backdrop both fire `close`; route them through the same
  // callback so the parent's state cannot drift out of sync with the element.
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    const handle = () => {
      dialog.style.transform = ''
      onClose()
    }
    dialog.addEventListener('close', handle)
    return () => dialog.removeEventListener('close', handle)
  }, [onClose])

  // Clicking outside the sheet closes it. `<dialog>` gives the backdrop no
  // element of its own, so the test is whether the point landed in the dialog's
  // own box.
  const handleBackdrop = useCallback(
    (event: React.MouseEvent<HTMLDialogElement>) => {
      if (event.target !== event.currentTarget) return
      const box = event.currentTarget.getBoundingClientRect()
      const inside =
        event.clientX >= box.left &&
        event.clientX <= box.right &&
        event.clientY >= box.top &&
        event.clientY <= box.bottom
      if (!inside) onClose()
    },
    [onClose],
  )

  const onPointerDown = (event: React.PointerEvent) => {
    // The whole header is the grab area, and it can also hold a control — the
    // geo sheet puts its on/off switch there. Capturing the pointer swallows
    // that control's click entirely, so a drag must not start on one. Found by
    // a checkbox that reported "clicking did not change its state".
    if ((event.target as HTMLElement).closest('button, input, a, select, textarea, label')) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = { id: event.pointerId, y0: event.clientY, t0: event.timeStamp, dy: 0 }
    ref.current?.classList.remove('sheet-settling')
  }

  const onPointerMove = (event: React.PointerEvent) => {
    const drag = dragRef.current
    const dialog = ref.current
    if (!drag || !dialog || drag.id !== event.pointerId) return
    // Downward only. Letting it travel up would lift the sheet off its edge and
    // reveal the gap underneath.
    drag.dy = Math.max(0, event.clientY - drag.y0)
    dialog.style.transform = `translateY(${drag.dy}px)`
  }

  const onPointerUp = (event: React.PointerEvent) => {
    const drag = dragRef.current
    const dialog = ref.current
    dragRef.current = null
    if (!drag || !dialog) return
    event.currentTarget.releasePointerCapture(event.pointerId)

    const elapsed = Math.max(1, event.timeStamp - drag.t0)
    const velocity = drag.dy / elapsed
    const far = drag.dy > dialog.getBoundingClientRect().height * DISMISS_FRACTION

    dialog.classList.add('sheet-settling')
    if (far || velocity > DISMISS_VELOCITY) {
      onClose()
    } else {
      // Snap back. The class is left on for the length of the transition only.
      dialog.style.transform = ''
      window.setTimeout(() => dialog.classList.remove('sheet-settling'), 220)
    }
  }

  return (
    <dialog ref={ref} className="sheet" onClick={handleBackdrop} aria-label={title}>
      {/*
        The grab area is the whole header, not just the 36px handle: a 4px-tall
        pill is a fine affordance and a terrible target.
      */}
      <div
        className="flex shrink-0 cursor-grab touch-none select-none flex-col items-stretch gap-2 px-4 pt-2.5 pb-3 active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <span aria-hidden className="mx-auto h-1 w-9 rounded-full bg-line-strong" />
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="min-w-0 truncate text-base font-medium">{title}</h2>
          {aside}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-4">{children}</div>
    </dialog>
  )
}
