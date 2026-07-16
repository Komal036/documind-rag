/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f4f6f8",
          100: "#e4e8ec",
          200: "#c8d1d9",
          400: "#8493a3",
          600: "#465165",
          800: "#242c3a",
          900: "#151a24",
        },
        accent: {
          50: "#eefcf6",
          100: "#d3f7e6",
          200: "#a2eecd",
          400: "#37c98f",
          500: "#1eab73",
          600: "#158a5c",
          700: "#116d49",
          900: "#0a3e2a",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
