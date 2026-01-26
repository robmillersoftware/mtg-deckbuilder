import { useState, useRef, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export function Layout() {
  const location = useLocation();
  const { user, isAuthenticated, isLoading, isHydrated, isFetchingUser, logout } = useAuth();
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [loadingTimedOut, setLoadingTimedOut] = useState(false);
  const [initialLoadGracePeriod, setInitialLoadGracePeriod] = useState(true);
  const menuRef = useRef<HTMLDivElement>(null);

  // Give the auth system a brief moment to start fetching after hydration
  useEffect(() => {
    if (isHydrated) {
      const timer = setTimeout(() => setInitialLoadGracePeriod(false), 500);
      return () => clearTimeout(timer);
    }
  }, [isHydrated]);

  // Show loading state when authenticated and fetching user data
  const isLoadingAuth = isHydrated && isAuthenticated && !user && (isLoading || isFetchingUser || initialLoadGracePeriod) && !loadingTimedOut;

  // Detect broken auth state: authenticated but no user, not loading, and grace period passed
  const isBrokenAuthState = isHydrated && isAuthenticated && !user && !isLoading && !isFetchingUser && !initialLoadGracePeriod && !loadingTimedOut;

  // Timeout for loading state - if stuck for 10 seconds, show recovery UI
  useEffect(() => {
    if (isAuthenticated && !user && isLoading) {
      const timeout = setTimeout(() => {
        setLoadingTimedOut(true);
      }, 10000);
      return () => clearTimeout(timeout);
    } else {
      setLoadingTimedOut(false);
    }
  }, [isAuthenticated, user, isLoading]);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const navigation = [
    { name: 'Build', href: '/' },
    { name: 'My Decks', href: '/decks', auth: true },
    { name: 'Simulate', href: '/simulate', auth: true },
    { name: 'Import', href: '/import', auth: true },
    { name: 'History', href: '/conversations', auth: true },
    { name: 'Meta', href: '/meta' },
  ];

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <span className="text-2xl">🔮</span>
              <span className="text-xl font-bold text-white">Spellbook</span>
            </Link>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-4">
              {navigation.map((item) => {
                if (item.auth && !isAuthenticated) return null;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={clsx(
                      'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                      location.pathname === item.href
                        ? 'bg-gray-800 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-800'
                    )}
                  >
                    {item.name}
                  </Link>
                );
              })}
            </nav>

            {/* User Menu */}
            <div className="flex items-center space-x-4">
              {isLoadingAuth ? (
                <div className="flex items-center space-x-2 text-sm text-gray-400">
                  <div className="w-8 h-8 rounded-full bg-gray-700 animate-pulse" />
                  <div className="hidden sm:block w-16 h-4 bg-gray-700 rounded animate-pulse" />
                </div>
              ) : isBrokenAuthState ? (
                <button
                  onClick={logout}
                  className="text-sm text-yellow-400 hover:text-yellow-300 transition-colors"
                >
                  Session expired - Log in
                </button>
              ) : isAuthenticated && user ? (
                <div ref={menuRef} className="relative">
                  <button
                    onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                    className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden">
                      {user.avatar_url ? (
                        <img
                          src={user.avatar_url}
                          alt=""
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-sm">
                          {(user.display_name || user.username || 'U')[0].toUpperCase()}
                        </span>
                      )}
                    </div>
                    <span>{user.display_name || user.username}</span>
                    <svg
                      className={clsx(
                        'w-4 h-4 transition-transform',
                        isUserMenuOpen && 'rotate-180'
                      )}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {isUserMenuOpen && (
                    <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-lg border border-gray-700 py-1 z-50">
                      <Link
                        to="/profile"
                        onClick={() => setIsUserMenuOpen(false)}
                        className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                      >
                        Profile
                      </Link>
                      <Link
                        to="/settings"
                        onClick={() => setIsUserMenuOpen(false)}
                        className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                      >
                        Settings
                      </Link>
                      <Link
                        to="/conversations"
                        onClick={() => setIsUserMenuOpen(false)}
                        className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                      >
                        Conversation History
                      </Link>
                      {user.is_superuser && (
                        <>
                          <div className="border-t border-gray-700 my-1"></div>
                          <Link
                            to="/admin"
                            onClick={() => setIsUserMenuOpen(false)}
                            className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                          >
                            Admin Dashboard
                          </Link>
                        </>
                      )}
                      <div className="border-t border-gray-700 my-1"></div>
                      <button
                        onClick={() => {
                          setIsUserMenuOpen(false);
                          logout();
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                      >
                        Logout
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    Login
                  </Link>
                  <Link
                    to="/register"
                    className="px-3 py-2 rounded-md text-sm font-medium bg-primary-600 hover:bg-primary-700 text-white transition-colors"
                  >
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Navigation Menu */}
      {isMobileMenuOpen && (
        <nav className="md:hidden bg-gray-900 border-b border-gray-800">
          <div className="max-w-7xl mx-auto px-4 py-3 space-y-1">
            {navigation.map((item) => {
              if (item.auth && !isAuthenticated) return null;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={clsx(
                    'block px-3 py-2 rounded-md text-base font-medium transition-colors',
                    location.pathname === item.href
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-300 hover:text-white hover:bg-gray-800'
                  )}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>
        </nav>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full">
        <Outlet />
      </main>
    </div>
  );
}
