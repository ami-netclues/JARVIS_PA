import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     port: 5173,
//     proxy: {
//       // Local dev convenience: frontend -> backend
//       "/api": "http://localhost:8001",
//       "/health": "http://localhost:8001"
//     }
//   }
// });

//BELOW IS SYSTEM GENERATED BUILD
//BELOW IS SYSTEM GENERATED BUILD

export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.lottie"],
  build: {
    outDir: "dist", 
  },
});

