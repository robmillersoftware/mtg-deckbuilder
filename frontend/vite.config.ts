import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Get API URL, handling empty strings
const apiUrl = process.env.VITE_API_URL?.trim() || 'http://backend:8000'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
      },
    },
  },
})
