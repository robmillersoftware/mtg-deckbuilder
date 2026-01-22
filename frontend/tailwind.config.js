/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // MTG colors
        mtg: {
          white: '#F8F6D8',
          blue: '#0E68AB',
          black: '#150B00',
          red: '#D3202A',
          green: '#00733E',
          gold: '#D4AF37',
          artifact: '#C0C0C0',
          colorless: '#CAC5C0',
        },
        // App colors
        primary: {
          50: '#f0f7ff',
          100: '#e0efff',
          200: '#b9dfff',
          300: '#7cc4ff',
          400: '#36a7ff',
          500: '#0c8cf1',
          600: '#006dce',
          700: '#0057a7',
          800: '#044a89',
          900: '#0a3e71',
        },
      },
      fontFamily: {
        beleren: ['Beleren', 'serif'],
      },
    },
  },
  plugins: [],
}
