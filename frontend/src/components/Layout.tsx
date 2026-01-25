import { useState, useRef, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export function Layout() {
  const location = useLocation();
  const { user, isAuthenticated, logout } = useAuth();
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

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

            {/* Navigation */}
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
              {isAuthenticated ? (
                <div ref={menuRef} className="relative">
                  <button
                    onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                    className="flex items-center space-x-2 text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden">
                      {user?.avatar_url ? (
                        <img
                          src={user.avatar_url}
                          alt=""
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-sm">
                          {(user?.display_name || user?.username || '?')[0].toUpperCase()}
                        </span>
                      )}
                    </div>
                    <span>{user?.display_name || user?.username}</span>
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
                      {user?.is_superuser && (
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

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 border-t border-gray-800 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-400">
              Spellbook - AI MTG Deck Builder
            </p>
            <div className="flex items-center space-x-4">
              <a
                href="https://scryfall.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-400 hover:text-white"
              >
                Card data by Scryfall
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
