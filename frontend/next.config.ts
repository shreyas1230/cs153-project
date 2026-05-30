import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a fully static site into `out/` for Cloudflare Pages (no SSR runtime).
  // The app is client-only and talks to the backend via NEXT_PUBLIC_API_URL.
  output: "export",
};

export default nextConfig;
