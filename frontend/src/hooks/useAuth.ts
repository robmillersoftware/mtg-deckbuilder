import { useCallback, useEffect } from 'react';
import { useAuthStore } from '@/store/auth';
import { usePreferencesStore } from '@/store/preferences';
import { authApi, usersApi } from '@/services/api';
import toast from 'react-hot-toast';

export function useAuth() {
  const {
    user,
    isAuthenticated,
    isLoading,
    isHydrated,
    isFetchingUser,
    setUser,
    setTokens,
    logout: storeLogout,
    setLoading,
    setFetchingUser,
  } = useAuthStore();

  // Fetch user and preferences on mount if we have tokens
  // Only runs after hydration completes and prevents concurrent fetches
  useEffect(() => {
    const fetchUserAndPreferences = async () => {
      // Wait for hydration before checking auth state
      if (!isHydrated) return;

      // Prevent concurrent fetches
      if (isFetchingUser) return;

      // Only fetch if authenticated but no user data
      if (isAuthenticated && !user) {
        setFetchingUser(true);
        setLoading(true);
        try {
          const response = await usersApi.getMe();
          setUser(response.data);
          // Also load preferences
          try {
            const prefsResponse = await usersApi.getPreferences();
            usePreferencesStore.getState().setPreferences(prefsResponse.data);
          } catch (prefsError) {
            console.error('Failed to load preferences:', prefsError);
          }
        } catch (error) {
          console.error('Failed to fetch user:', error);
          // Only logout if it's an auth error, not a network error
          const status = (error as { response?: { status?: number } })?.response?.status;
          if (status === 401 || status === 403) {
            storeLogout();
          }
        } finally {
          setLoading(false);
          setFetchingUser(false);
        }
      }
    };

    fetchUserAndPreferences();
  }, [isHydrated, isAuthenticated, user, isFetchingUser, setUser, storeLogout, setLoading, setFetchingUser]);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setFetchingUser(true); // Prevent useEffect from racing with login
    try {
      const response = await authApi.login(email, password);
      const { access_token, refresh_token } = response.data;
      setTokens(access_token, refresh_token);

      // Fetch user data
      const userResponse = await usersApi.getMe();
      setUser(userResponse.data);

      // Load user preferences
      try {
        const prefsResponse = await usersApi.getPreferences();
        usePreferencesStore.getState().setPreferences(prefsResponse.data);
      } catch (prefsError) {
        console.error('Failed to load preferences:', prefsError);
      }

      toast.success('Welcome back!');
      return true;
    } catch (error: unknown) {
      console.error('Login failed:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Login failed';
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
      setFetchingUser(false);
    }
  }, [setTokens, setUser, setLoading, setFetchingUser]);

  const register = useCallback(async (
    email: string,
    password: string,
    onSuccess?: () => void
  ) => {
    setLoading(true);
    try {
      await authApi.register(email, password);
      toast.success('Account created! Please check your email to verify.');
      // Call onSuccess callback after toast to ensure navigation happens
      if (onSuccess) {
        onSuccess();
      }
      return true;
    } catch (error: unknown) {
      console.error('Registration failed:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Registration failed';
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [setLoading]);

  const logout = useCallback(() => {
    storeLogout();
    toast.success('Logged out');
  }, [storeLogout]);

  const verifyEmail = useCallback(async (token: string) => {
    try {
      await authApi.verifyEmail(token);
      toast.success('Email verified successfully!');
      return true;
    } catch (error) {
      console.error('Email verification failed:', error);
      toast.error('Email verification failed');
      return false;
    }
  }, []);

  const forgotPassword = useCallback(async (email: string) => {
    try {
      await authApi.forgotPassword(email);
      toast.success('Password reset email sent');
      return true;
    } catch (error) {
      console.error('Forgot password failed:', error);
      toast.error('Failed to send reset email');
      return false;
    }
  }, []);

  const resetPassword = useCallback(async (token: string, newPassword: string, newPasswordConfirm: string) => {
    try {
      await authApi.resetPassword(token, newPassword, newPasswordConfirm);
      toast.success('Password reset successfully!');
      return true;
    } catch (error) {
      console.error('Password reset failed:', error);
      toast.error('Password reset failed');
      return false;
    }
  }, []);

  const refreshUser = useCallback(async () => {
    if (!isAuthenticated) return;
    try {
      const response = await usersApi.getMe();
      setUser(response.data);
    } catch (error) {
      console.error('Failed to refresh user:', error);
    }
  }, [isAuthenticated, setUser]);

  return {
    user,
    isAuthenticated,
    isLoading,
    isHydrated,
    isFetchingUser,
    login,
    register,
    logout,
    verifyEmail,
    forgotPassword,
    resetPassword,
    refreshUser,
  };
}
