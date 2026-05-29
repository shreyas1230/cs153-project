import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClaimLens — Discover what papers actually disagree about",
  description:
    "Upload scientific PDFs and ask a question. Surface real contradictions, hidden support, and apparent conflicts caused by terminology drift.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-white text-slate-900 antialiased">{children}</body>
    </html>
  );
}
