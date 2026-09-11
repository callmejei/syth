import { useEffect, useRef } from 'react'

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * Dialog keyboard behaviour: Escape to close, focus moved in on open, focus
 * returned to whatever opened it on close, and Tab cycled within the dialog.
 *
 * The focus return matters more than it looks: without it, closing a dialog
 * drops focus back to the top of the document, so a keyboard user has to tab
 * through the whole page again to get back to where they were.
 */
export function useDialog(onClose) {
  const ref = useRef(null)

  useEffect(() => {
    const previouslyFocused = document.activeElement

    const node = ref.current
    if (node) {
      const first = node.querySelector(FOCUSABLE)
      // Prefer the first field over the close button.
      const target = node.querySelector('input, select, textarea') || first
      target?.focus()
    }

    function onKeyDown(event) {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
        return
      }
      if (event.key !== 'Tab' || !ref.current) return

      const focusable = Array.from(ref.current.querySelectorAll(FOCUSABLE)).filter(
        (element) => element.offsetParent !== null,
      )
      if (!focusable.length) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    // The page behind a modal must not scroll under it.
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
      if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus()
    }
  }, [onClose])

  return ref
}

export default useDialog
