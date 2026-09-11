/**
 * Colours resolve to CSS variables so a single `data-theme` swap re-themes the
 * whole app, and so components name a role (`bg-surface`) instead of a shade
 * (`bg-white`) — the latter cannot follow a theme.
 *
 * `<alpha-value>` keeps Tailwind's opacity modifiers working (`bg-accent/25`).
 */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  darkMode: ['class', '[data-theme="dark"]'],
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        canvas: token('canvas'),
        surface: {
          DEFAULT: token('surface'),
          raised: token('surface-raised'),
        },
        sunken: token('surface-sunken'),
        line: {
          DEFAULT: token('border'),
          strong: token('border-strong'),
        },
        fg: {
          DEFAULT: token('text'),
          muted: token('text-muted'),
          subtle: token('text-subtle'),
        },
        accent: {
          DEFAULT: token('accent'),
          hover: token('accent-hover'),
          fg: token('accent-fg'),
          subtle: token('accent-subtle'),
          text: token('accent-text'),
        },
        success: {
          DEFAULT: token('success'),
          subtle: token('success-subtle'),
          text: token('success-text'),
        },
        warning: {
          DEFAULT: token('warning'),
          subtle: token('warning-subtle'),
          text: token('warning-text'),
        },
        danger: {
          DEFAULT: token('danger'),
          subtle: token('danger-subtle'),
          text: token('danger-text'),
          fg: token('danger-fg'),
        },
        info: {
          DEFAULT: token('info'),
          subtle: token('info-subtle'),
          text: token('info-text'),
        },
        violet: {
          DEFAULT: token('violet'),
          subtle: token('violet-subtle'),
          text: token('violet-text'),
        },
      },
      fontFamily: {
        sans: ['Inter Variable', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Plus Jakarta Sans Variable', 'Inter Variable', 'sans-serif'],
        mono: ['JetBrains Mono Variable', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        // A 1.2 scale from a 14px UI base: dense enough for data, still >=16px
        // for body copy.
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
        xs: ['0.75rem', { lineHeight: '1.125rem' }],
        sm: ['0.8125rem', { lineHeight: '1.25rem' }],
        base: ['0.9375rem', { lineHeight: '1.5rem' }],
        lg: ['1.0625rem', { lineHeight: '1.625rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
      },
      borderRadius: {
        xl: '0.75rem',
        '2xl': '1rem',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'none' },
        },
        'scale-in': {
          from: { opacity: '0', transform: 'scale(0.97)' },
          to: { opacity: '1', transform: 'none' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        // 150-300ms per the motion guidance; entrances ease-out.
        'fade-in': 'fade-in 200ms ease-out both',
        'scale-in': 'scale-in 180ms ease-out both',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
}
