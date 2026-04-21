/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ieeeBlue: "#ADD8E6",
        ieeeGreen: "#90EE90"
      }
    }
  },
  plugins: []
};

