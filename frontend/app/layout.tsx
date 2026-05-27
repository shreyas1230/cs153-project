import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Claim Conflict Detector",
  description: "Surface conflicts and terminology mismatches across scientific papers",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
