import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

import { Layout } from '@/components/Layout';
import { HomePage } from '@/pages/Home';
import { LoginPage } from '@/pages/Login';
import { RegisterPage } from '@/pages/Register';
import { ForgotPasswordPage } from '@/pages/ForgotPassword';
import { ResetPasswordPage } from '@/pages/ResetPassword';
import { DecksPage } from '@/pages/Decks';
import { DeckViewPage } from '@/pages/DeckView';
import { SharedDeckViewPage } from '@/pages/SharedDeckView';
import { DeckImportPage } from '@/pages/DeckImport';
import { MetaPage } from '@/pages/Meta';
import { ProfilePage } from '@/pages/Profile';
import { SettingsPage } from '@/pages/Settings';
import { ConversationsPage } from '@/pages/Conversations';
import { AdminPage } from '@/pages/Admin';
import { VerifyEmailPage } from '@/pages/VerifyEmail';
import { SimulationPage } from '@/pages/Simulation';
import { GuidedBuilderPage } from '@/pages/GuidedBuilder';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="login" element={<LoginPage />} />
            <Route path="register" element={<RegisterPage />} />
            <Route path="forgot-password" element={<ForgotPasswordPage />} />
            <Route path="reset-password" element={<ResetPasswordPage />} />
            <Route path="verify-email" element={<VerifyEmailPage />} />
            <Route path="decks" element={<DecksPage />} />
            <Route path="deck/:id" element={<DeckViewPage />} />
            <Route path="deck/shared/:shareToken" element={<SharedDeckViewPage />} />
            <Route path="meta" element={<MetaPage />} />
            <Route path="profile" element={<ProfilePage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="conversations" element={<ConversationsPage />} />
            <Route path="import" element={<DeckImportPage />} />
            <Route path="admin" element={<AdminPage />} />
            <Route path="simulate" element={<SimulationPage />} />
            <Route path="build" element={<GuidedBuilderPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'bg-gray-800 text-white',
          duration: 3000,
        }}
      />
    </QueryClientProvider>
  );
}

export default App;
