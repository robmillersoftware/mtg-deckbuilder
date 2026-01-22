import { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { authApi } from '@/services/api';

type VerificationStatus = 'loading' | 'success' | 'error';

export function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState<VerificationStatus>('loading');
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    const token = searchParams.get('token');
    if (token) {
      verifyToken(token);
    } else {
      setStatus('error');
      setErrorMessage('No verification token provided');
    }
  }, [searchParams]);

  const verifyToken = async (token: string) => {
    try {
      await authApi.verifyEmail(token);
      setStatus('success');
    } catch (error: unknown) {
      setStatus('error');
      const message =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Verification failed. The token may be invalid or expired.';
      setErrorMessage(message);
    }
  };

  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="max-w-md w-full">
        {status === 'loading' && (
          <div className="bg-gray-900 rounded-lg p-8 text-center">
            <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <h1 className="text-xl font-bold text-white mb-2">Verifying Your Email</h1>
            <p className="text-gray-400">Please wait...</p>
          </div>
        )}

        {status === 'success' && (
          <div className="bg-gray-900 rounded-lg p-8 text-center">
            <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-white mb-2">Email Verified!</h1>
            <p className="text-gray-400 mb-6">
              Your email has been verified successfully. You can now log in to your account.
            </p>
            <Link
              to="/login"
              className="inline-block px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
            >
              Go to Login
            </Link>
          </div>
        )}

        {status === 'error' && (
          <div className="bg-gray-900 rounded-lg p-8 text-center">
            <div className="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-white mb-2">Verification Failed</h1>
            <p className="text-gray-400 mb-6">{errorMessage}</p>
            <div className="space-y-3">
              <Link
                to="/register"
                className="block px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
              >
                Register Again
              </Link>
              <Link
                to="/login"
                className="block px-6 py-3 bg-gray-800 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
              >
                Go to Login
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
