import { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/hooks/useAuth';
import { usersApi } from '@/services/api';
import toast from 'react-hot-toast';

export function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const [displayName, setDisplayName] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isDirty, setIsDirty] = useState(false);

  // Email change state
  const [showEmailChange, setShowEmailChange] = useState(false);
  const [newEmail, setNewEmail] = useState('');
  const [emailChangeLoading, setEmailChangeLoading] = useState(false);

  // Password change state
  const [showPasswordChange, setShowPasswordChange] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordChangeLoading, setPasswordChangeLoading] = useState(false);

  // Avatar upload ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (user) {
      setDisplayName(user.display_name || '');
      setAvatarUrl(user.avatar_url || '');
    }
  }, [user]);

  const handleDisplayNameChange = (value: string) => {
    setDisplayName(value);
    setIsDirty(true);
  };

  const handleAvatarUrlChange = (value: string) => {
    setAvatarUrl(value);
    setIsDirty(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!displayName.trim()) {
      toast.error('Display name cannot be empty');
      return;
    }

    setIsLoading(true);
    try {
      await usersApi.updateProfile({
        display_name: displayName,
        avatar_url: avatarUrl || undefined,
      });
      await refreshUser();
      setIsDirty(false);
      toast.success('Profile updated successfully');
    } catch (error) {
      console.error('Failed to update profile:', error);
      toast.error('Failed to update profile');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      toast.error('Please upload a valid image file (JPEG, PNG, GIF, or WebP)');
      return;
    }

    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      toast.error('File size must be less than 5MB');
      return;
    }

    setIsLoading(true);
    try {
      const result = await usersApi.uploadAvatar(file);
      setAvatarUrl(result.avatar_url || '');
      await refreshUser();
      toast.success('Avatar uploaded successfully');
    } catch (error) {
      console.error('Failed to upload avatar:', error);
      toast.error('Failed to upload avatar');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEmailChangeRequest = async () => {
    if (!newEmail.trim()) {
      toast.error('Please enter a valid email address');
      return;
    }

    setEmailChangeLoading(true);
    try {
      await usersApi.requestEmailChange(newEmail);
      toast.success('Verification email sent to your new email address');
      setShowEmailChange(false);
      setNewEmail('');
    } catch (error: unknown) {
      console.error('Failed to request email change:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to request email change';
      toast.error(message);
    } finally {
      setEmailChangeLoading(false);
    }
  };

  const handlePasswordChange = async () => {
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error('Please fill in all password fields');
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error('New passwords do not match');
      return;
    }

    if (newPassword.length < 8) {
      toast.error('New password must be at least 8 characters');
      return;
    }

    setPasswordChangeLoading(true);
    try {
      await usersApi.changePassword(currentPassword, newPassword);
      toast.success('Password changed successfully');
      setShowPasswordChange(false);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (error: unknown) {
      console.error('Failed to change password:', error);
      const message = (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to change password';
      toast.error(message);
    } finally {
      setPasswordChangeLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Please log in to view your profile</div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">Profile Settings</h1>

      <div className="bg-gray-900 rounded-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Avatar Preview */}
          <div className="flex items-center space-x-6">
            <div className="relative">
              <div className="w-20 h-20 rounded-full bg-gray-700 flex items-center justify-center overflow-hidden">
                {avatarUrl ? (
                  <img
                    src={avatarUrl}
                    alt="Avatar"
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                  />
                ) : (
                  <span className="text-3xl text-gray-400">
                    {(displayName || user.username || '?')[0].toUpperCase()}
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="absolute bottom-0 right-0 w-7 h-7 bg-primary-600 rounded-full flex items-center justify-center hover:bg-primary-700 transition-colors"
                title="Upload avatar"
              >
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/gif,image/webp"
                onChange={handleAvatarUpload}
                className="hidden"
              />
            </div>
            <div>
              <h2 className="text-lg font-medium text-white">{displayName || user.username}</h2>
              <p className="text-sm text-gray-400">@{user.username}</p>
            </div>
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Email
            </label>
            <div className="flex space-x-2">
              <input
                type="email"
                value={user.email}
                disabled
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 cursor-not-allowed"
              />
              <button
                type="button"
                onClick={() => setShowEmailChange(!showEmailChange)}
                className="px-4 py-2 text-sm text-primary-400 hover:text-primary-300 transition-colors"
              >
                Change
              </button>
            </div>

            {showEmailChange && (
              <div className="mt-3 p-4 bg-gray-800 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-3">
                  A verification link will be sent to your new email address.
                </p>
                <div className="flex space-x-2">
                  <input
                    type="email"
                    value={newEmail}
                    onChange={(e) => setNewEmail(e.target.value)}
                    placeholder="Enter new email address"
                    className="flex-1 px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
                  />
                  <button
                    type="button"
                    onClick={handleEmailChangeRequest}
                    disabled={emailChangeLoading || !newEmail.trim()}
                    className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {emailChangeLoading ? 'Sending...' : 'Send Verification'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Username (Read-only) */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Username
            </label>
            <input
              type="text"
              value={user.username}
              disabled
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 cursor-not-allowed"
            />
          </div>

          {/* Display Name */}
          <div>
            <label htmlFor="displayName" className="block text-sm font-medium text-gray-300 mb-2">
              Display Name
            </label>
            <input
              id="displayName"
              type="text"
              value={displayName}
              onChange={(e) => handleDisplayNameChange(e.target.value)}
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
              placeholder="Your display name"
            />
          </div>

          {/* Avatar URL */}
          <div>
            <label htmlFor="avatarUrl" className="block text-sm font-medium text-gray-300 mb-2">
              Avatar URL
            </label>
            <input
              id="avatarUrl"
              type="url"
              value={avatarUrl}
              onChange={(e) => handleAvatarUrlChange(e.target.value)}
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
              placeholder="https://example.com/avatar.jpg"
            />
            <p className="mt-1 text-xs text-gray-500">
              Enter a URL to an image for your avatar
            </p>
          </div>

          {/* Account Status */}
          <div className="pt-4 border-t border-gray-800">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Account Status</h3>
            <div className="flex flex-wrap gap-2">
              <span
                className={`px-3 py-1 text-xs rounded-full ${
                  user.is_verified
                    ? 'bg-green-900/50 text-green-300'
                    : 'bg-yellow-900/50 text-yellow-300'
                }`}
              >
                {user.is_verified ? 'Email Verified' : 'Email Not Verified'}
              </span>
              <span
                className={`px-3 py-1 text-xs rounded-full ${
                  user.is_active
                    ? 'bg-green-900/50 text-green-300'
                    : 'bg-red-900/50 text-red-300'
                }`}
              >
                {user.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>

          {/* Password Change */}
          <div className="pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-300">Password</h3>
              <button
                type="button"
                onClick={() => setShowPasswordChange(!showPasswordChange)}
                className="px-4 py-2 text-sm text-primary-400 hover:text-primary-300 transition-colors"
              >
                {showPasswordChange ? 'Cancel' : 'Change Password'}
              </button>
            </div>

            {showPasswordChange && (
              <div className="mt-3 p-4 bg-gray-800 rounded-lg border border-gray-700 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Current Password
                  </label>
                  <input
                    type="password"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    New Password
                  </label>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
                    placeholder="At least 8 characters"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-1">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-primary-500"
                  />
                </div>
                <button
                  type="button"
                  onClick={handlePasswordChange}
                  disabled={passwordChangeLoading || !currentPassword || !newPassword || !confirmPassword}
                  className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {passwordChangeLoading ? 'Changing...' : 'Change Password'}
                </button>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <div className="flex justify-end pt-4">
            <button
              type="submit"
              disabled={isLoading || !isDirty}
              className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
