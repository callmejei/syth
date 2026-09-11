/**
 * Inline SVG icon set.
 *
 * Replaces the Unicode glyphs (⌂ ▤ ▦ ⛨ ◈ ⚙ ⤓) previously used for navigation.
 * Those render from whatever font the OS substitutes, so they differ between
 * Windows, macOS and Linux, cannot inherit stroke weight, and cannot be sized
 * or themed as design tokens. One geometry, one stroke width, currentColor.
 */

const BASE = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  viewBox: '0 0 24 24',
}

/** Icons are decorative when a text label sits beside them, so they are hidden
 *  from assistive tech by default; pass a `title` when an icon stands alone. */
function Icon({ size = 20, title, children, ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      {...BASE}
      {...rest}
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  )
}

export const HomeIcon = (props) => (
  <Icon {...props}>
    <path d="M3 10.5 12 3l9 7.5" />
    <path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5" />
  </Icon>
)

export const DatabaseIcon = (props) => (
  <Icon {...props}>
    <ellipse cx="12" cy="5.5" rx="7.5" ry="3" />
    <path d="M4.5 5.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
    <path d="M4.5 11.5v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6" />
  </Icon>
)

export const SlidersIcon = (props) => (
  <Icon {...props}>
    <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h10M18 18h2" />
    <circle cx="16" cy="6" r="2" />
    <circle cx="10" cy="12" r="2" />
    <circle cx="16" cy="18" r="2" />
  </Icon>
)

export const ShieldIcon = (props) => (
  <Icon {...props}>
    <path d="M12 3l7 3v5.5c0 4.3-2.9 8.2-7 9.5-4.1-1.3-7-5.2-7-9.5V6z" />
    <path d="m9 12 2 2 4-4" />
  </Icon>
)

export const BoxIcon = (props) => (
  <Icon {...props}>
    <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
    <path d="M4 7.5 12 12l8-4.5M12 12v9" />
  </Icon>
)

export const ListIcon = (props) => (
  <Icon {...props}>
    <path d="M8 6h12M8 12h12M8 18h12" />
    <circle cx="4" cy="6" r="1" />
    <circle cx="4" cy="12" r="1" />
    <circle cx="4" cy="18" r="1" />
  </Icon>
)

export const SettingsIcon = (props) => (
  <Icon {...props}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.500 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" />
  </Icon>
)

export const UploadIcon = (props) => (
  <Icon {...props}>
    <path d="M12 16V4" />
    <path d="m7 9 5-5 5 5" />
    <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
  </Icon>
)

export const DownloadIcon = (props) => (
  <Icon {...props}>
    <path d="M12 4v12" />
    <path d="m7 11 5 5 5-5" />
    <path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2" />
  </Icon>
)

export const TrashIcon = (props) => (
  <Icon {...props}>
    <path d="M4 7h16" />
    <path d="M10 11v6M14 11v6" />
    <path d="M5 7l1 12a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1l1-12" />
    <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </Icon>
)

export const PlusIcon = (props) => (
  <Icon {...props}>
    <path d="M12 5v14M5 12h14" />
  </Icon>
)

export const CloseIcon = (props) => (
  <Icon {...props}>
    <path d="m6 6 12 12M18 6 6 18" />
  </Icon>
)

export const InfoIcon = (props) => (
  <Icon {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5" />
    <circle cx="12" cy="7.75" r="0.75" fill="currentColor" stroke="none" />
  </Icon>
)

export const ChevronIcon = ({ open = false, ...props }) => (
  <Icon {...props} style={{ transition: 'transform 150ms ease-out', transform: open ? 'rotate(90deg)' : 'none', ...props.style }}>
    <path d="m9 6 6 6-6 6" />
  </Icon>
)

export const RefreshIcon = (props) => (
  <Icon {...props}>
    <path d="M20 11a8 8 0 0 0-13.7-5.7L4 7.5" />
    <path d="M4 4v3.5h3.5" />
    <path d="M4 13a8 8 0 0 0 13.7 5.7L20 16.5" />
    <path d="M20 20v-3.5h-3.5" />
  </Icon>
)

export const PlayIcon = (props) => (
  <Icon {...props}>
    <path d="M7 5.5v13l11-6.5z" />
  </Icon>
)

export const LogoutIcon = (props) => (
  <Icon {...props}>
    <path d="M14 20H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h8" />
    <path d="M17 15l4-3-4-3" />
    <path d="M21 12H10" />
  </Icon>
)

export default Icon
