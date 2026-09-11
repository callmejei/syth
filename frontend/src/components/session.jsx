import { createContext, useCallback, useContext, useMemo, useState } from 'react'

const KEY = 'ocbc-datacraft-session'
const SessionContext = createContext({ user: null, signIn: () => {}, signOut: () => {} })

/**
 * Demo session only.
 *
 * THIS IS NOT AUTHENTICATION. There is no auth on the API: every endpoint is
 * open, and this gate is a localStorage flag that anyone can set from the
 * console. It exists so the POC can be demonstrated with a branded sign-in
 * screen, nothing more. Before this platform touches real data it needs real
 * server-side auth, and the login screen says so on its face rather than
 * implying a security control that does not exist.
 */
export function SessionProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem(KEY)
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  })

  const signIn = useCallback((email) => {
    const name = (email || '').split('@')[0].replace(/[._-]+/g, ' ').trim()
    const profile = {
      email: email || 'analyst@ocbc.com',
      name: name
        ? name.replace(/\b\w/g, (c) => c.toUpperCase())
        : 'DataCraft Analyst',
      signedInAt: new Date().toISOString(),
    }
    localStorage.setItem(KEY, JSON.stringify(profile))
    setUser(profile)
    return profile
  }, [])

  const signOut = useCallback(() => {
    localStorage.removeItem(KEY)
    setUser(null)
  }, [])

  const value = useMemo(() => ({ user, signIn, signOut }), [user, signIn, signOut])
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export const useSession = () => useContext(SessionContext)
