import { useTheme } from './theme'

/**
 * OCBC DataCraft logo.
 *
 * Two variants exist because the wordmark's "DATA" is set in the brand ink
 * (#182130). On the dark canvas that is all but invisible, so a recoloured
 * asset swaps in. The reds are identical in both files; only the ink moves.
 */
export function Logo({ variant = 'full', height = 32, className = '', showTagline = true }) {
  const { theme } = useTheme()
  const dark = theme === 'dark'

  if (variant === 'mark') {
    return (
      <img
        src="/brand/logo-mark.png"
        alt="OCBC DataCraft"
        height={height}
        style={{ height }}
        className={`w-auto select-none ${className}`}
        draggable={false}
      />
    )
  }

  return (
    <img
      src={dark ? '/brand/logo-full-dark.png' : '/brand/logo-full.png'}
      alt="OCBC DataCraft, powered by GDO"
      height={height}
      style={{ height }}
      className={`w-auto select-none ${className}`}
      draggable={false}
    />
  )
}

/**
 * Compact lockup for the sidebar: the mark plus a typeset wordmark. Scaling the
 * full artwork down to sidebar height makes "POWERED BY GDO" illegible, so the
 * wordmark is set in type at that size instead.
 */
export function LogoLockup({ className = '' }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <img
        src="/brand/logo-mark.png"
        alt=""
        aria-hidden="true"
        className="h-9 w-auto select-none"
        draggable={false}
      />
      <div className="leading-none">
        <div className="font-display text-[12px] font-extrabold leading-none tracking-tight text-accent-text">
          OCBC
        </div>
        <div className="font-display text-[15px] font-bold leading-tight tracking-tight text-fg">
          Data<span className="text-accent-text">Craft</span>
        </div>
        {/* The artwork's own tagline is illegible below ~40px, so it is set in
            type here rather than dropped. */}
        <div className="mt-[3px] text-[8px] font-semibold uppercase leading-none tracking-[0.14em] text-fg-subtle">
          Powered by <span className="text-accent-text">GDO</span>
        </div>
      </div>
      <span className="sr-only">OCBC DataCraft, powered by GDO</span>
    </div>
  )
}

export default Logo
